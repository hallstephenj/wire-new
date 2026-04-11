import os
import time
import logging
import asyncio
import httpx
from jose import jwt, jwk, JWTError
from fastapi import Request, HTTPException
from wire.db import get_conn

log = logging.getLogger("wire.auth")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_JWKS_URL = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"

# In-memory JWKS cache — refreshed every hour or on kid miss
_jwks_cache: dict = {"keys": [], "fetched_at": 0.0}
_JWKS_TTL = 3600
_jwks_lock = asyncio.Lock()


async def _fetch_jwks(force: bool = False) -> list:
    now = time.time()
    if not force and _jwks_cache["keys"] and (now - _jwks_cache["fetched_at"]) < _JWKS_TTL:
        return _jwks_cache["keys"]
    async with _jwks_lock:
        # Re-check after acquiring lock (another coroutine may have fetched already)
        now = time.time()
        if not force and _jwks_cache["keys"] and (now - _jwks_cache["fetched_at"]) < _JWKS_TTL:
            return _jwks_cache["keys"]
        t_jwks = time.perf_counter()
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(_JWKS_URL)
            resp.raise_for_status()
        jwks_ms = (time.perf_counter() - t_jwks) * 1000
        log.info(f"JWKS fetch: {jwks_ms:.0f}ms")
        keys = resp.json().get("keys", [])
        _jwks_cache["keys"] = keys
        _jwks_cache["fetched_at"] = time.time()
        return keys


async def _verify_token(token: str) -> dict:
    """Verify a Supabase JWT using the project's JWKS endpoint (ES256)."""
    header = jwt.get_unverified_header(token)
    kid = header.get("kid")
    alg = header.get("alg", "ES256")

    keys = await _fetch_jwks()
    key_data = next((k for k in keys if k.get("kid") == kid), None)

    # kid not in cache → maybe rotated, refetch once
    if key_data is None:
        keys = await _fetch_jwks(force=True)
        key_data = next((k for k in keys if k.get("kid") == kid), None)

    if key_data is None:
        raise JWTError(f"No matching key for kid={kid}")

    public_key = jwk.construct(key_data, algorithm=alg)
    return jwt.decode(token, public_key.to_dict(), algorithms=[alg],
                      options={"verify_aud": False})


def _get_token(request: Request) -> str | None:
    import json as _json
    # Supabase JS v2 may store session as JSON in sb-*-auth-token cookie
    for name, value in request.cookies.items():
        if name.endswith("-auth-token"):
            try:
                session = _json.loads(value)
                if isinstance(session, dict) and session.get("access_token"):
                    return session["access_token"]
            except Exception:
                pass
    # Plain cookie set by our login page
    token = request.cookies.get("sb-access-token")
    if token:
        return token
    # Authorization header fallback
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


async def get_current_user(request: Request):
    """Returns user profile dict or None if not logged in."""
    token = _get_token(request)
    if not token:
        return None
    try:
        t0 = time.perf_counter()
        payload = await _verify_token(token)
        jwt_ms = (time.perf_counter() - t0) * 1000
        if jwt_ms > 200:
            log.warning(f"Slow JWT verify: {jwt_ms:.0f}ms")

        user_id = payload.get("sub")
        if not user_id:
            return None

        t1 = time.perf_counter()
        conn = get_conn()
        profile = conn.execute(
            "SELECT * FROM user_profiles WHERE id=?", (user_id,)
        ).fetchone()
        conn.close()
        db_ms = (time.perf_counter() - t1) * 1000
        if db_ms > 200:
            log.warning(f"Slow auth DB lookup: {db_ms:.0f}ms")

        if not profile:
            return {"id": user_id, "is_admin": 0, "subscription_status": "free",
                    "onboarding_completed": 0, "email": payload.get("email")}
        return dict(profile)
    except Exception:
        return None


async def require_user(request: Request):
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    return user


async def require_admin(request: Request):
    user = await get_current_user(request)
    if not user or not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin required")
    return user


async def require_pro(request: Request):
    user = await get_current_user(request)
    if not user or user.get("subscription_status") not in ("pro",):
        raise HTTPException(status_code=403, detail="Pro required")
    return user
