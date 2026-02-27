import re
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
import feedparser
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from wire.db import get_conn
from wire.ingest import _poll_single_feed, is_gnews_blocked
from wire.rewrite import urgent_rewrite
from wire.events import push as ev

log = logging.getLogger("wire.reference")

BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

# ── Shared HTTP client for reference site scraping ────────────────────────
_ref_client: httpx.AsyncClient | None = None


def init_ref_http_client():
    """Create the shared HTTP client for reference checks. Call once at app startup."""
    global _ref_client
    _ref_client = httpx.AsyncClient(
        timeout=20,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )


async def close_ref_http_client():
    """Close the shared HTTP client. Call at app shutdown."""
    global _ref_client
    if _ref_client:
        await _ref_client.aclose()
        _ref_client = None


def _get_ref_client() -> httpx.AsyncClient:
    """Return the shared reference client."""
    if _ref_client is not None:
        return _ref_client
    return httpx.AsyncClient(timeout=20, follow_redirects=True)


# ── Site-specific parsers ─────────────────────────────────────────────

def parse_techmeme(html: str, max_headlines: int = 20) -> list[dict]:
    """Extract top headlines from Techmeme's front page."""
    soup = BeautifulSoup(html, "html.parser")
    headlines = []
    seen = set()

    for col_id in ("topcol1", "topcol2"):
        col = soup.find(id=col_id)
        if not col:
            continue
        for item in col.select(".item"):
            ed = item.select_one(".ed")
            if not ed:
                continue
            link = ed.select_one("strong a")
            if not link:
                continue
            title = link.get_text(strip=True)
            href = link.get("href", "")
            if not title or not href or title in seen:
                continue
            seen.add(title)
            headlines.append({"headline": title, "url": href})
            if len(headlines) >= max_headlines:
                return headlines

    return headlines


def parse_drudge(html: str, max_headlines: int = 30) -> list[dict]:
    """Extract top headlines from Drudge Report."""
    soup = BeautifulSoup(html, "html.parser")
    headlines = []
    seen = set()

    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        title = link.get_text(strip=True)
        if not title or len(title) < 10:
            continue

        # Skip internal/archive links
        parsed = urlparse(href)
        if not parsed.scheme or not parsed.netloc:
            continue
        if "drudgereport.com" in parsed.netloc:
            continue
        if any(skip in href.lower() for skip in ["archive", "mailto:", "javascript:"]):
            continue

        # Drudge headlines are typically upper-case or mixed-case links
        if title in seen:
            continue
        seen.add(title)
        headlines.append({"headline": title, "url": href})
        if len(headlines) >= max_headlines:
            break

    return headlines


def parse_apnews(html: str, max_headlines: int = 20) -> list[dict]:
    """Extract headlines from AP News front page."""
    soup = BeautifulSoup(html, "html.parser")
    headlines = []
    seen = set()

    # AP uses various headline patterns; look for links with substantial text
    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        title = link.get_text(strip=True)
        if not title or len(title) < 20:
            continue

        # Only AP article links
        if not href.startswith("/article/") and "apnews.com/article/" not in href:
            continue

        if href.startswith("/"):
            href = "https://apnews.com" + href

        if title in seen:
            continue
        seen.add(title)
        headlines.append({"headline": title, "url": href})
        if len(headlines) >= max_headlines:
            break

    return headlines


def parse_apnews_rss(xml: str, max_headlines: int = 20) -> list[dict]:
    """Fallback: parse AP's RSS feed."""
    feed = feedparser.parse(xml)
    headlines = []
    for entry in feed.entries[:max_headlines]:
        title = getattr(entry, "title", None)
        link = getattr(entry, "link", None)
        if title and link:
            headlines.append({"headline": title, "url": link})
    return headlines


def parse_googlenews_rss(xml: str, max_headlines: int = 20) -> list[dict]:
    """Parse Google News RSS feed for top editorial picks."""
    feed = feedparser.parse(xml)
    headlines = []
    for entry in feed.entries[:max_headlines]:
        title = getattr(entry, "title", None)
        link = getattr(entry, "link", None)
        if not title or not link:
            continue
        # Strip source suffix from Google News titles
        if " - " in title:
            title = title.rsplit(" - ", 1)[0]
        headlines.append({"headline": title, "url": link})
    return headlines


PARSERS = {
    "techmeme": parse_techmeme,
    "drudge": parse_drudge,
    "apnews": parse_apnews,
    "googlenews": parse_googlenews_rss,
}


# ── Fetch & parse ─────────────────────────────────────────────────────

async def fetch_site(url: str, parser_key: str, max_headlines: int = 20) -> list[dict]:
    """Fetch a reference site and parse its headlines."""
    headers = {"User-Agent": BROWSER_UA}
    client = _get_ref_client()

    try:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
    except Exception as e:
        # For AP News, try RSS fallback
        if parser_key == "apnews":
            log.warning(f"AP News HTML fetch failed ({e}), trying RSS fallback")
            try:
                resp = await client.get(
                    "https://rsshub.app/apnews/topics/apf-topnews",
                    headers=headers,
                )
                resp.raise_for_status()
                return parse_apnews_rss(resp.text, max_headlines)
            except Exception as e2:
                log.error(f"AP News RSS fallback also failed: {e2}")
                return []
        raise

    parser = PARSERS.get(parser_key)
    if not parser:
        log.error(f"Unknown parser: {parser_key}")
        return []

    return parser(resp.text, max_headlines)


# ── Similarity check against existing clusters ────────────────────────

# ── Cached reference matcher ──────────────────────────────────────────
# Builds a TF-IDF index once per check_references run, reuses for all
# headline matching within that run.

_ref_match_cache = {"vectorizer": None, "matrix": None, "rows": [], "built_at": 0}
_REF_CACHE_TTL = 30  # seconds


def _build_ref_match_cache():
    """Build/refresh the reference matching TF-IDF index."""
    import time as _time
    from datetime import timedelta
    from wire.config import load_config
    now_mono = _time.monotonic()
    if _ref_match_cache["vectorizer"] and (now_mono - _ref_match_cache["built_at"]) < _REF_CACHE_TTL:
        return

    cfg = load_config()
    lookback = cfg["clustering"]["lookback_hours"]
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(hours=lookback)).isoformat()

    conn = get_conn()
    rows = conn.execute(
        "SELECT id, rewritten_headline FROM story_clusters WHERE last_updated > ? AND expires_at > ? ORDER BY source_count DESC LIMIT 2000",
        (cutoff, now.isoformat())
    ).fetchall()
    conn.close()

    if not rows:
        _ref_match_cache.update({"vectorizer": None, "matrix": None, "rows": [], "built_at": now_mono})
        return

    headlines = [r["rewritten_headline"] or "" for r in rows]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform(headlines)
    _ref_match_cache.update({"vectorizer": vectorizer, "matrix": matrix, "rows": list(rows), "built_at": now_mono})


def _invalidate_ref_cache():
    """Force rebuild on next match attempt."""
    _ref_match_cache["built_at"] = 0


def _find_matching_cluster(headline: str, threshold: float = 0.30) -> str | None:
    """Check if a headline matches any existing cluster via cached TF-IDF index."""
    _build_ref_match_cache()

    if not _ref_match_cache["rows"] or _ref_match_cache["vectorizer"] is None:
        return None

    try:
        new_vec = _ref_match_cache["vectorizer"].transform([headline])
        sims = cosine_similarity(new_vec, _ref_match_cache["matrix"])[0]
        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])
        if best_sim >= threshold:
            return _ref_match_cache["rows"][best_idx]["id"]
    except Exception as e:
        log.warning(f"TF-IDF similarity check error: {e}")

    return None


# ── Main orchestrator ─────────────────────────────────────────────────

async def _probe_gnews():
    """Quick probe to detect if Google News is accessible (vs consent wall)."""
    from wire.ingest import _gnews_blocked
    import wire.ingest as _ingest_mod
    if _ingest_mod._gnews_blocked:
        return
    try:
        client = _get_ref_client()
        resp = await client.head(
            "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
            headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        if "consent.google" in str(resp.url):
            _ingest_mod._gnews_blocked = True
            log.warning("Google News consent wall detected — gap-fill searches disabled")
    except Exception:
        pass


async def check_references(on_progress=None) -> dict:
    """
    Main reference check: scrape editorial front pages, compare against
    existing clusters, fill gaps via Google News search.
    """
    # Detect consent wall early so gap-fill searches are skipped
    await _probe_gnews()

    conn = get_conn()
    sites = conn.execute(
        "SELECT * FROM reference_sites WHERE enabled = 1 ORDER BY id"
    ).fetchall()
    conn.close()

    if not sites:
        log.info("No enabled reference sites, skipping check")
        return {"sites_checked": 0, "headlines_found": 0, "gaps_found": 0, "gaps_filled": 0}

    # Pre-build the matching index once for the entire reference check
    _build_ref_match_cache()

    now = datetime.now(timezone.utc).isoformat()
    total_found = 0
    total_gaps = 0
    total_filled = 0

    for site_idx, site in enumerate(sites):
        site_found = 0
        site_gaps = 0

        try:
            headlines = await fetch_site(site["url"], site["parser"], site["max_headlines"])
            site_found = len(headlines)
            total_found += site_found
            log.info(f"Reference check: {site['name']} — {site_found} headlines")
        except Exception as e:
            log.error(f"Reference check failed for {site['name']}: {e}")
            conn = get_conn()
            conn.execute(
                "INSERT INTO reference_check_log (site_id, site_name, headline, source_url, status, detail, checked_at) VALUES (?,?,?,?,?,?,?)",
                (site["id"], site["name"], None, site["url"], "error", str(e), now)
            )
            conn.commit()
            conn.close()
            if on_progress:
                on_progress(site_idx + 1, len(sites))
            continue

        # Collect log entries, then write them in one batch at the end
        # to avoid holding a connection open during _poll_single_feed calls
        log_entries = []

        for hl in headlines:
            headline_text = hl["headline"]
            headline_url = hl["url"]

            # Check if this headline matches an existing cluster
            matched_cluster = _find_matching_cluster(headline_text)

            if matched_cluster:
                log_entries.append(
                    (site["id"], site["name"], headline_text, headline_url, "matched", matched_cluster, None, now)
                )
            else:
                # Gap detected — try to fill via Google News search
                site_gaps += 1
                total_gaps += 1

                # Skip Google News gap-fill if consent wall is active (EU servers)
                if is_gnews_blocked():
                    log_entries.append(
                        (site["id"], site["name"], headline_text, headline_url, "gap_unfilled", None, "Google News blocked (consent wall)", now)
                    )
                    continue

                try:
                    # Search Google News RSS for this headline
                    search_query = headline_text[:80].replace(" ", "+")
                    search_url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
                    ingested = await _poll_single_feed(search_url, "Google News", "general")

                    if ingested > 0:
                        # Invalidate cache since new clusters may have been created
                        _invalidate_ref_cache()
                        # Re-check if a cluster now covers this headline
                        new_match = _find_matching_cluster(headline_text, threshold=0.25)
                        if new_match:
                            total_filled += 1
                            log_entries.append(
                                (site["id"], site["name"], headline_text, headline_url, "gap_filled", new_match, f"Ingested {ingested} items via search", now)
                            )
                        else:
                            log_entries.append(
                                (site["id"], site["name"], headline_text, headline_url, "gap_unfilled", None, f"Searched but no cluster match (ingested {ingested})", now)
                            )
                    else:
                        log_entries.append(
                            (site["id"], site["name"], headline_text, headline_url, "gap_unfilled", None, "No results from search", now)
                        )
                except Exception as e:
                    log.warning(f"Gap fill search failed for '{headline_text[:40]}': {e}")
                    log_entries.append(
                        (site["id"], site["name"], headline_text, headline_url, "error", None, f"Search failed: {e}", now)
                    )

        # Write all log entries and update site stats in one batch
        conn = get_conn()
        for entry in log_entries:
            conn.execute(
                "INSERT INTO reference_check_log (site_id, site_name, headline, source_url, status, matched_cluster_id, detail, checked_at) VALUES (?,?,?,?,?,?,?,?)",
                entry
            )
        conn.execute(
            "UPDATE reference_sites SET last_checked=?, last_found=?, last_gaps=?, updated_at=? WHERE id=?",
            (now, site_found, site_gaps, now, site["id"])
        )
        conn.commit()
        conn.close()

        if on_progress:
            on_progress(site_idx + 1, len(sites))

    summary = {
        "sites_checked": len(sites),
        "headlines_found": total_found,
        "gaps_found": total_gaps,
        "gaps_filled": total_filled,
    }
    log.info(f"Reference check complete: {summary}")
    ev("reference_check", **summary)
    return summary


async def run_reference_check(on_progress=None) -> dict:
    """Wrapper that runs check_references and triggers urgent_rewrite if gaps were filled."""
    summary = await check_references(on_progress=on_progress)

    if summary.get("gaps_filled", 0) > 0:
        try:
            await urgent_rewrite()
        except Exception as e:
            log.warning(f"Urgent rewrite after reference check failed: {e}")

    return summary
