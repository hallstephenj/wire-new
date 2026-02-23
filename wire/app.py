import json
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from wire.db import init_db, get_conn
from wire.config import load_config, set_override, get_overrides
from wire.ingest import poll_feeds, search_sweep, backfill_48h, invalidate_filter_cache
from wire.rewrite import rewrite_pending
from wire.cluster import assign_cluster, merge_existing_clusters
from wire.events import snapshot as events_snapshot
from wire.scores import all_scores, get_score
from wire.reference import run_reference_check
from wire.algorithm import get_active_version, build_source_quality_case, build_reputable_sources_sql, build_hot_score_sql, list_versions, set_active_version

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("wire")

scheduler = AsyncIOScheduler()

boot_state = {
    "phase": "reference_check",   # reference_check → polling → clustering → rewriting → ready
    "detail": "",
    "clusters": 0,
    "pending": 0,
    # Per-phase progress (0.0 – 1.0)
    "reference_progress": 0,
    "polling_progress": 0,
    "clustering_progress": 0,
    "rewriting_progress": 0,
}

_boot_task = None

async def startup_backfill():
    """Walk through 3 boot phases: polling, clustering, rewriting."""
    conn = get_conn()
    cluster_count = conn.execute("SELECT COUNT(*) as c FROM story_clusters").fetchone()["c"]
    conn.close()

    if cluster_count >= 100:
        log.info(f"Startup: {cluster_count} clusters found, skipping backfill")
        boot_state["phase"] = "ready"
        boot_state["clusters"] = cluster_count
        boot_state["reference_progress"] = 1
        boot_state["polling_progress"] = 1
        boot_state["clustering_progress"] = 1
        boot_state["rewriting_progress"] = 1
        return

    log.info(f"Startup: only {cluster_count} clusters, running backfill...")

    # Phase 0: reference check — scrape editorial front pages
    boot_state["phase"] = "reference_check"
    boot_state["detail"] = ""
    log.info("Boot phase 0 — reference check")

    def ref_progress(done, total):
        boot_state["reference_progress"] = done / total

    try:
        await run_reference_check(on_progress=ref_progress)
    except Exception as e:
        log.warning(f"Boot reference check error: {e}")
    boot_state["reference_progress"] = 1

    # Phase 1: polling — 3 passes of (poll_feeds + search_sweep)
    # Progress is tracked per-feed across all 3 passes
    boot_state["phase"] = "polling"
    num_passes = 3
    for i in range(num_passes):
        boot_state["detail"] = f"Pass {i+1}/{num_passes}"
        log.info(f"Boot phase 1 — polling pass {i+1}/{num_passes}")

        def poll_progress(done, total, _pass=i):
            # Each pass has poll_feeds portion (0-0.8) + search_sweep portion (0.8-1.0)
            pass_frac = (_pass + (done / total) * 0.8) / num_passes
            boot_state["polling_progress"] = min(pass_frac, 0.99)

        def search_progress(done, total, _pass=i):
            pass_frac = (_pass + 0.8 + (done / total) * 0.2) / num_passes
            boot_state["polling_progress"] = min(pass_frac, 0.99)

        try:
            await poll_feeds(on_progress=poll_progress)
        except Exception as e:
            log.warning(f"Boot poll error: {e}")
        try:
            await search_sweep(on_progress=search_progress)
        except Exception as e:
            log.warning(f"Boot search error: {e}")
    boot_state["polling_progress"] = 1

    # Phase 2: clustering — backfill_48h with per-feed progress
    boot_state["phase"] = "clustering"
    boot_state["detail"] = ""
    log.info("Boot phase 2 — clustering (backfill_48h)")

    def cluster_progress(done, total):
        boot_state["clustering_progress"] = done / total
        # Periodically update cluster count
        if done % 10 == 0 or done == total:
            try:
                c = get_conn()
                boot_state["clusters"] = c.execute("SELECT COUNT(*) as c FROM story_clusters").fetchone()["c"]
                c.close()
            except Exception:
                pass

    try:
        await backfill_48h(on_progress=cluster_progress)
    except Exception as e:
        log.warning(f"Boot clustering error: {e}")
    conn = get_conn()
    boot_state["clusters"] = conn.execute("SELECT COUNT(*) as c FROM story_clusters").fetchone()["c"]
    conn.close()
    boot_state["clustering_progress"] = 1
    log.info(f"Boot clustering done: {boot_state['clusters']} clusters")

    # Phase 3: rewriting — track pending countdown
    boot_state["phase"] = "rewriting"
    boot_state["detail"] = ""
    initial_pending = None
    for i in range(20):
        conn = get_conn()
        pending = conn.execute("""
            SELECT COUNT(*) as c FROM story_clusters sc
            WHERE sc.source_count >= 2
            AND EXISTS (
                SELECT 1 FROM raw_items ri
                WHERE ri.cluster_id = sc.id
                AND ri.original_headline = sc.rewritten_headline
            )
        """).fetchone()["c"]
        conn.close()
        boot_state["pending"] = pending
        if initial_pending is None:
            initial_pending = pending if pending > 0 else 1
        if pending == 0:
            break
        boot_state["rewriting_progress"] = 1 - (pending / initial_pending)
        log.info(f"Boot rewrite pass {i+1}: {pending} pending...")
        try:
            await rewrite_pending()
        except Exception as e:
            log.warning(f"Boot rewrite error: {e}")

    conn = get_conn()
    boot_state["clusters"] = conn.execute("SELECT COUNT(*) as c FROM story_clusters").fetchone()["c"]
    conn.close()
    boot_state["pending"] = 0
    boot_state["rewriting_progress"] = 1

    # Merge duplicate clusters before going live
    try:
        conn = get_conn()
        merged = merge_existing_clusters(conn)
        conn.close()
        if merged:
            log.info(f"Boot merge: combined {merged} duplicate clusters")
    except Exception as e:
        log.warning(f"Boot merge error: {e}")

    boot_state["phase"] = "ready"
    conn = get_conn()
    boot_state["clusters"] = conn.execute("SELECT COUNT(*) as c FROM story_clusters").fetchone()["c"]
    conn.close()
    log.info(f"Boot complete: {boot_state['clusters']} clusters")


async def _start_scheduler_after_boot():
    """Wait for boot to finish, then start scheduled jobs."""
    global _boot_task, scheduler
    if _boot_task:
        await _boot_task
    cfg = load_config()
    # Create a fresh scheduler if the old one was shut down
    if not scheduler.running:
        scheduler = AsyncIOScheduler()
    scheduler.add_job(poll_feeds, "interval", minutes=cfg["polling"]["rss_interval_minutes"], id="rss", next_run_time=datetime.now(timezone.utc))
    scheduler.add_job(search_sweep, "interval", minutes=cfg["polling"]["search_interval_minutes"], id="search", next_run_time=datetime.now(timezone.utc))
    scheduler.add_job(rewrite_pending, "interval", minutes=2, id="rewrite", next_run_time=datetime.now(timezone.utc))
    scheduler.add_job(cleanup_expired, "interval", hours=cfg["polling"]["cleanup_interval_hours"], id="cleanup")
    scheduler.add_job(run_reference_check, "cron", hour=6, id="reference_check")
    scheduler.add_job(deduplicate_clusters, "interval", hours=1, id="dedup")
    if not scheduler.running:
        scheduler.start()
    log.info("Scheduler started (post-boot)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _boot_task
    init_db()

    # Boot backfill in background, scheduler deferred until boot completes
    _boot_task = asyncio.create_task(startup_backfill())
    asyncio.create_task(_start_scheduler_after_boot())

    log.info("DINWIRE started")
    yield
    scheduler.shutdown(wait=False)

app = FastAPI(lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Cache templates in memory at import time
_TEMPLATES_DIR = Path(__file__).parent / "templates"
_INDEX_HTML = (_TEMPLATES_DIR / "index.html").read_text()
_RIVER_HTML = (_TEMPLATES_DIR / "river.html").read_text()

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_INDEX_HTML)

@app.get("/api/boot-status")
async def boot_status():
    return boot_state

@app.get("/api/stories")
async def get_stories(
    sort: str = Query("hot", pattern="^(hot|new)$"),
    category: str = Query("all", pattern="^(all|markets|tech|politics|world|general)$"),
    since: str | None = None,
    limit: int = Query(20, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    algo = get_active_version()

    where = ["sc.expires_at > ?"]
    params = [now]

    # Hide curated-hidden stories from homepage
    where.append("(co.hidden IS NULL OR co.hidden = 0)")

    if category == "all":
        # Exclude "general" bucket from the all view
        if algo.get("general_exclusion", True):
            where.append("COALESCE(co.category_override, sc.category) != 'general'")
    else:
        where.append("COALESCE(co.category_override, sc.category) = ?")
        params.append(category)

    if since:
        where.append("sc.last_updated > ?")
        params.append(since)

    # Sort precedence: BREAKING > pinned > boosted > normal > demoted
    sort_prefix = """CASE WHEN COALESCE(co.breaking, 0) = 1 THEN 0
                          WHEN COALESCE(co.pinned, 0) = 1 THEN 1
                          WHEN COALESCE(co.boost, 0) > 0 THEN 2
                          WHEN COALESCE(co.boost, 0) < 0 THEN 4
                          ELSE 3 END"""
    # Deprioritize "world" in the ALL view
    world_deprio = "CASE WHEN COALESCE(co.category_override, sc.category) = 'world' THEN 1 ELSE 0 END" if (category == "all" and algo.get("world_deprio", True)) else "0+0"
    hot_score = build_hot_score_sql(algo)
    reputable_sources_sql = build_reputable_sources_sql(algo)
    min_reputable = algo["min_reputable_sources"]
    if sort == "hot":
        reputable_filter = f"""(SELECT COUNT(DISTINCT cs2.source_name) FROM cluster_sources cs2
            WHERE cs2.cluster_id = sc.id AND cs2.source_name IN {reputable_sources_sql}) >= {min_reputable}"""
        # Scoop-boosted clusters bypass the min reputable sources filter
        if algo.get("scoop_enabled"):
            where.append(f"({reputable_filter} OR (COALESCE(co.boost, 0) > 0 AND co.scoop_boosted_at IS NOT NULL))")
        else:
            where.append(reputable_filter)
        order = f"{sort_prefix}, {world_deprio}, {hot_score} DESC, sc.published_at DESC"
    else:
        order = f"{sort_prefix}, sc.published_at DESC"
    where_clause = " AND ".join(where)

    rows = conn.execute(f"""
        SELECT sc.id, COALESCE(co.headline_override, sc.rewritten_headline) as rewritten_headline,
               sc.primary_url, sc.primary_source,
               COALESCE(co.category_override, sc.category) as category,
               sc.source_count, sc.first_seen, sc.last_updated, sc.published_at,
               COALESCE(co.breaking, 0) as breaking,
               (SELECT ri2.original_headline FROM raw_items ri2 WHERE ri2.cluster_id = sc.id LIMIT 1) as original_headline
        FROM story_clusters sc
        LEFT JOIN curation_overrides co ON co.cluster_id = sc.id
        WHERE {where_clause}
        ORDER BY {order}
        LIMIT ? OFFSET ?
    """, (*params, limit, offset)).fetchall()

    cluster_ids = [r["id"] for r in rows]

    # Batch-fetch sources and items for all clusters in 2 queries instead of 2*N
    sources_by_cluster = {}
    items_by_cluster = {}
    if cluster_ids:
        ph = ",".join("?" for _ in cluster_ids)
        all_sources = conn.execute(
            f"SELECT cluster_id, source_name, MIN(source_url) as source_url FROM cluster_sources WHERE cluster_id IN ({ph}) GROUP BY cluster_id, source_name",
            cluster_ids
        ).fetchall()
        for s in all_sources:
            sources_by_cluster.setdefault(s["cluster_id"], []).append(s)

        all_items = conn.execute(
            f"SELECT cluster_id, original_headline, source_url, source_name, published_at FROM raw_items WHERE cluster_id IN ({ph}) ORDER BY published_at DESC",
            cluster_ids
        ).fetchall()
        for i in all_items:
            items_by_cluster.setdefault(i["cluster_id"], []).append(i)

    stories = []
    for r in rows:
        cid = r["id"]
        other_sources = [{"name": s["source_name"], "url": s["source_url"]}
                         for s in sources_by_cluster.get(cid, []) if s["source_url"] != r["primary_url"]]
        items = [{"headline": i["original_headline"], "url": i["source_url"], "source": i["source_name"], "published_at": i["published_at"]}
                 for i in items_by_cluster.get(cid, [])]
        stories.append({
            "id": cid,
            "headline": r["rewritten_headline"],
            "original_headline": r["original_headline"],
            "url": r["primary_url"],
            "source": r["primary_source"],
            "category": r["category"],
            "source_count": r["source_count"],
            "first_seen": r["first_seen"],
            "published_at": r["published_at"],
            "last_updated": r["last_updated"],
            "breaking": bool(r["breaking"]),
            "other_sources": other_sources,
            "items": items,
        })

    total = conn.execute(f"SELECT COUNT(*) as c FROM story_clusters sc LEFT JOIN curation_overrides co ON co.cluster_id = sc.id WHERE {where_clause}", params).fetchone()["c"]
    conn.close()

    return {"stories": stories, "updated_at": now, "total": total}

@app.get("/api/health")
async def health():
    conn = get_conn()
    clusters = conn.execute("SELECT COUNT(*) as c FROM story_clusters").fetchone()["c"]
    items = conn.execute("SELECT COUNT(*) as c FROM raw_items").fetchone()["c"]
    conn.close()
    return {"status": "ok", "clusters": clusters, "raw_items": items}

@app.get("/cp", response_class=HTMLResponse)
async def river():
    return HTMLResponse(_RIVER_HTML)

@app.get("/api/river/events")
async def river_events(limit: int = Query(100, ge=1, le=500), event_type: str | None = None):
    return {"events": events_snapshot(limit=limit, event_type=event_type)}

@app.get("/api/river/status")
async def river_status():
    cfg = load_config()
    conn = get_conn()
    clusters = conn.execute("SELECT COUNT(*) as c FROM story_clusters WHERE expires_at > ?",
                            (datetime.now(timezone.utc).isoformat(),)).fetchone()["c"]
    raw_items = conn.execute("SELECT COUNT(*) as c FROM raw_items").fetchone()["c"]
    pending = conn.execute("""
        SELECT COUNT(*) as c FROM story_clusters sc
        WHERE sc.expires_at > ?
        AND sc.source_count >= 2
        AND EXISTS (
            SELECT 1 FROM raw_items ri
            WHERE ri.cluster_id = sc.id
            AND ri.original_headline = sc.rewritten_headline
        )
    """, (datetime.now(timezone.utc).isoformat(),)).fetchone()["c"]
    active_filters = conn.execute("SELECT COUNT(*) as c FROM content_filters WHERE enabled = 1").fetchone()["c"]
    cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    filtered_24h = conn.execute("SELECT COUNT(*) as c FROM filtered_items WHERE filtered_at > ?", (cutoff_24h,)).fetchone()["c"]
    reference_sites_count = conn.execute("SELECT COUNT(*) as c FROM reference_sites WHERE enabled = 1").fetchone()["c"]
    last_ref_check = conn.execute("SELECT MAX(last_checked) as lc FROM reference_sites").fetchone()["lc"]
    reference_gaps_24h = conn.execute("SELECT COUNT(*) as c FROM reference_check_log WHERE status IN ('gap_filled','gap_unfilled') AND checked_at > ?", (cutoff_24h,)).fetchone()["c"]
    conn.close()

    jobs = {}
    for job_id in ("rss", "search", "rewrite", "cleanup", "reference_check", "dedup"):
        job = scheduler.get_job(job_id)
        if job:
            nrt = job.next_run_time
            jobs[job_id] = {
                "next_run": nrt.isoformat() if nrt else None,
                "interval": str(job.trigger),
            }
        else:
            jobs[job_id] = None

    return {
        "jobs": jobs,
        "db": {"clusters": clusters, "raw_items": raw_items, "pending_rewrites": pending,
               "active_filters": active_filters, "filtered_24h": filtered_24h,
               "reference_sites": reference_sites_count, "last_reference_check": last_ref_check,
               "reference_gaps_24h": reference_gaps_24h},
        "config": {
            "similarity_threshold": cfg["clustering"]["similarity_threshold"],
            "lookback_hours": cfg["clustering"]["lookback_hours"],
            "model": cfg["ai"]["model"],
            "batch_size": cfg["ai"]["batch_size"],
            "rss_interval_minutes": cfg["polling"]["rss_interval_minutes"],
            "search_interval_minutes": cfg["polling"]["search_interval_minutes"],
        },
    }

@app.get("/api/river/pipeline")
async def river_pipeline(limit: int = Query(50, ge=1, le=200)):
    conn = get_conn()
    rows = conn.execute("""
        SELECT ri.source_name, ri.original_headline, ri.category as feed_category,
               ri.ingested_at, ri.cluster_id,
               sc.rewritten_headline, sc.category as cluster_category, sc.source_count
        FROM raw_items ri
        LEFT JOIN story_clusters sc ON sc.id = ri.cluster_id
        ORDER BY ri.ingested_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    items = []
    for r in rows:
        is_rewritten = (r["rewritten_headline"] and r["rewritten_headline"] != r["original_headline"])
        items.append({
            "source": r["source_name"],
            "headline": r["original_headline"],
            "feed_category": r["feed_category"],
            "cluster_category": r["cluster_category"],
            "cluster_id": r["cluster_id"][:8] if r["cluster_id"] else None,
            "source_count": r["source_count"],
            "rewrite_status": "REWRITTEN" if is_rewritten else "PENDING",
            "rewritten_headline": r["rewritten_headline"] if is_rewritten else None,
            "ingested_at": r["ingested_at"],
        })
    return {"items": items}

@app.post("/api/recluster")
async def recluster():
    """Dedupe raw_items and rebuild all clusters from scratch."""
    conn = get_conn()
    try:
        # Dedupe raw_items
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("""
            DELETE FROM raw_items WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM raw_items GROUP BY original_headline, source_name
            )
        """)
        conn.execute("DELETE FROM cluster_sources")
        conn.execute("DELETE FROM story_clusters")
        conn.execute("UPDATE raw_items SET cluster_id = NULL")
        conn.commit()
        conn.execute("PRAGMA foreign_keys=ON")

        items = conn.execute("""
            SELECT id, original_headline, source_url, source_name, category, published_at
            FROM raw_items ORDER BY published_at ASC NULLS LAST
        """).fetchall()

        for i, item in enumerate(items):
            cid = assign_cluster(conn, item["original_headline"], item["source_url"],
                                 item["source_name"], item["category"], published_at=item["published_at"])
            conn.execute("UPDATE raw_items SET cluster_id=? WHERE id=?", (cid, item["id"]))
            if (i + 1) % 200 == 0:
                conn.commit()
        conn.commit()

        total = conn.execute("SELECT COUNT(*) FROM story_clusters").fetchone()[0]
        by_cat = conn.execute("SELECT category, COUNT(*) FROM story_clusters GROUP BY category").fetchall()
        return {"status": "ok", "clusters": total, "items": len(items),
                "categories": {r[0]: r[1] for r in by_cat}}
    finally:
        conn.close()

@app.get("/api/scores")
async def scores():
    data = all_scores()
    return {"scores": [{"source": name, **breakdown} for name, breakdown in data.items()]}

import time as _time
_categories_cache = {"data": None, "loaded_at": 0}
_CATEGORIES_CACHE_TTL = 30

def _get_valid_categories():
    now = _time.time()
    if _categories_cache["data"] and (now - _categories_cache["loaded_at"]) < _CATEGORIES_CACHE_TTL:
        return _categories_cache["data"]
    conn = get_conn()
    rows = conn.execute("SELECT name FROM categories WHERE enabled=1").fetchall()
    conn.close()
    cats = {r["name"] for r in rows} if rows else {"tech", "markets", "politics", "world", "general"}
    _categories_cache["data"] = cats
    _categories_cache["loaded_at"] = now
    return cats

def _invalidate_categories_cache():
    _categories_cache["data"] = None
    _categories_cache["loaded_at"] = 0

def _log_curation(conn, action, cluster_id, detail):
    """Insert a curation log entry, auto-including the cluster headline."""
    if cluster_id and "headline" not in detail:
        row = conn.execute("SELECT COALESCE(co.headline_override, sc.rewritten_headline) as h FROM story_clusters sc LEFT JOIN curation_overrides co ON co.cluster_id = sc.id WHERE sc.id=?", (cluster_id,)).fetchone()
        if row:
            detail["headline"] = row["h"]
    conn.execute(
        "INSERT INTO curation_log (action, cluster_id, detail, created_at) VALUES (?, ?, ?, ?)",
        (action, cluster_id, json.dumps(detail), datetime.now(timezone.utc).isoformat())
    )

@app.post("/api/river/edit-headline")
async def edit_headline(request: Request):
    body = await request.json()
    cluster_id = body["cluster_id"]
    headline = body["headline"].strip()
    if not headline:
        return JSONResponse({"error": "headline required"}, status_code=400)

    conn = get_conn()
    old = conn.execute("SELECT rewritten_headline FROM story_clusters WHERE id=?", (cluster_id,)).fetchone()
    if not old:
        conn.close()
        return JSONResponse({"error": "cluster not found"}, status_code=404)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("UPDATE story_clusters SET rewritten_headline=? WHERE id=?", (headline, cluster_id))
    conn.execute("""
        INSERT INTO curation_overrides (cluster_id, headline_override, locked, updated_at)
        VALUES (?, ?, 1, ?)
        ON CONFLICT(cluster_id) DO UPDATE SET headline_override=?, locked=1, updated_at=?
    """, (cluster_id, headline, now, headline, now))
    _log_curation(conn, "edit_headline", cluster_id, {"before": old["rewritten_headline"], "after": headline})
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.post("/api/river/edit-category")
async def edit_category(request: Request):
    body = await request.json()
    cluster_id = body["cluster_id"]
    category = body["category"].strip().lower()
    valid = _get_valid_categories()
    if category not in valid:
        return JSONResponse({"error": f"invalid category, must be one of: {', '.join(sorted(valid))}"}, status_code=400)

    conn = get_conn()
    old = conn.execute("SELECT category FROM story_clusters WHERE id=?", (cluster_id,)).fetchone()
    if not old:
        conn.close()
        return JSONResponse({"error": "cluster not found"}, status_code=404)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("UPDATE story_clusters SET category=? WHERE id=?", (category, cluster_id))
    conn.execute("""
        INSERT INTO curation_overrides (cluster_id, category_override, locked, updated_at)
        VALUES (?, ?, 1, ?)
        ON CONFLICT(cluster_id) DO UPDATE SET category_override=?, locked=1, updated_at=?
    """, (cluster_id, category, now, category, now))
    _log_curation(conn, "edit_category", cluster_id, {"before": old["category"], "after": category})
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.post("/api/river/pin")
async def pin_cluster(request: Request):
    body = await request.json()
    cluster_id = body["cluster_id"]
    pinned = int(body.get("pinned", 1))
    rank = int(body.get("rank", 0))

    conn = get_conn()
    if not conn.execute("SELECT 1 FROM story_clusters WHERE id=?", (cluster_id,)).fetchone():
        conn.close()
        return JSONResponse({"error": "cluster not found"}, status_code=404)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO curation_overrides (cluster_id, pinned, pin_rank, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(cluster_id) DO UPDATE SET pinned=?, pin_rank=?, updated_at=?
    """, (cluster_id, pinned, rank, now, pinned, rank, now))
    action = "pin" if pinned else "unpin"
    _log_curation(conn, action, cluster_id, {"pinned": pinned, "rank": rank})
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.post("/api/river/hide")
async def hide_cluster(request: Request):
    body = await request.json()
    cluster_id = body["cluster_id"]
    hidden = int(body.get("hidden", 1))

    conn = get_conn()
    if not conn.execute("SELECT 1 FROM story_clusters WHERE id=?", (cluster_id,)).fetchone():
        conn.close()
        return JSONResponse({"error": "cluster not found"}, status_code=404)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO curation_overrides (cluster_id, hidden, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(cluster_id) DO UPDATE SET hidden=?, updated_at=?
    """, (cluster_id, hidden, now, hidden, now))
    action = "hide" if hidden else "unhide"
    _log_curation(conn, action, cluster_id, {"hidden": hidden})
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.post("/api/river/lock")
async def lock_cluster(request: Request):
    body = await request.json()
    cluster_id = body["cluster_id"]
    locked = int(body.get("locked", 1))

    conn = get_conn()
    if not conn.execute("SELECT 1 FROM story_clusters WHERE id=?", (cluster_id,)).fetchone():
        conn.close()
        return JSONResponse({"error": "cluster not found"}, status_code=404)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO curation_overrides (cluster_id, locked, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(cluster_id) DO UPDATE SET locked=?, updated_at=?
    """, (cluster_id, locked, now, locked, now))
    action = "lock" if locked else "unlock"
    _log_curation(conn, action, cluster_id, {"locked": locked})
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.post("/api/river/merge")
async def merge_clusters(request: Request):
    body = await request.json()
    source_id = body["source_cluster_id"]
    target_id = body["target_cluster_id"]

    conn = get_conn()
    source = conn.execute("SELECT * FROM story_clusters WHERE id=?", (source_id,)).fetchone()
    target = conn.execute("SELECT * FROM story_clusters WHERE id=?", (target_id,)).fetchone()
    if not source or not target:
        conn.close()
        return JSONResponse({"error": "cluster not found"}, status_code=404)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("UPDATE raw_items SET cluster_id=? WHERE cluster_id=?", (target_id, source_id))
    sources = conn.execute("SELECT source_name, source_url, added_at FROM cluster_sources WHERE cluster_id=?", (source_id,)).fetchall()
    for s in sources:
        conn.execute("""
            INSERT OR IGNORE INTO cluster_sources (cluster_id, source_name, source_url, added_at)
            VALUES (?, ?, ?, ?)
        """, (target_id, s["source_name"], s["source_url"], s["added_at"]))
    conn.execute("DELETE FROM cluster_sources WHERE cluster_id=?", (source_id,))
    new_count = conn.execute("SELECT COUNT(DISTINCT source_name) as c FROM cluster_sources WHERE cluster_id=?", (target_id,)).fetchone()["c"]
    conn.execute("UPDATE story_clusters SET source_count=?, last_updated=? WHERE id=?", (new_count, now, target_id))
    conn.execute("DELETE FROM curation_overrides WHERE cluster_id=?", (source_id,))
    conn.execute("DELETE FROM story_clusters WHERE id=?", (source_id,))
    _log_curation(conn, "merge", target_id, {"source_cluster_id": source_id, "target_cluster_id": target_id, "new_source_count": new_count})
    conn.commit()
    conn.close()
    return {"status": "ok", "new_source_count": new_count}

@app.post("/api/river/split")
async def split_cluster(request: Request):
    body = await request.json()
    cluster_id = body["cluster_id"]
    item_ids = body["item_ids"]

    if not item_ids:
        return JSONResponse({"error": "item_ids required"}, status_code=400)

    conn = get_conn()
    original = conn.execute("SELECT * FROM story_clusters WHERE id=?", (cluster_id,)).fetchone()
    if not original:
        conn.close()
        return JSONResponse({"error": "cluster not found"}, status_code=404)

    placeholders = ",".join("?" for _ in item_ids)
    items = conn.execute(f"SELECT * FROM raw_items WHERE id IN ({placeholders}) AND cluster_id=?",
                         (*item_ids, cluster_id)).fetchall()
    if not items:
        conn.close()
        return JSONResponse({"error": "no matching items found in cluster"}, status_code=400)

    import uuid
    now = datetime.now(timezone.utc).isoformat()
    new_id = str(uuid.uuid4())

    first = items[0]
    conn.execute("""
        INSERT INTO story_clusters (id, rewritten_headline, primary_url, primary_source, category, source_count, first_seen, last_updated, expires_at, published_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (new_id, first["original_headline"], first["source_url"], first["source_name"],
          original["category"], len(items), now, now, original["expires_at"], first["published_at"]))

    conn.execute(f"UPDATE raw_items SET cluster_id=? WHERE id IN ({placeholders})", (new_id, *item_ids))

    for item in items:
        conn.execute("""
            INSERT OR IGNORE INTO cluster_sources (cluster_id, source_name, source_url, added_at)
            VALUES (?, ?, ?, ?)
        """, (new_id, item["source_name"], item["source_url"], now))

    # Remove sources from original that no longer have items there
    for item in items:
        remaining = conn.execute("SELECT 1 FROM raw_items WHERE cluster_id=? AND source_name=?", (cluster_id, item["source_name"])).fetchone()
        if not remaining:
            conn.execute("DELETE FROM cluster_sources WHERE cluster_id=? AND source_name=?", (cluster_id, item["source_name"]))

    new_old_count = conn.execute("SELECT COUNT(DISTINCT source_name) as c FROM cluster_sources WHERE cluster_id=?", (cluster_id,)).fetchone()["c"]
    conn.execute("UPDATE story_clusters SET source_count=?, last_updated=? WHERE id=?", (max(new_old_count, 1), now, cluster_id))

    _log_curation(conn, "split", cluster_id, {"new_cluster_id": new_id, "item_ids": item_ids, "new_cluster_size": len(items)})
    conn.commit()
    conn.close()
    return {"status": "ok", "new_cluster_id": new_id}

@app.post("/api/river/config")
async def update_config(request: Request):
    body = await request.json()
    key = body["key"]
    value = body["value"]

    allowed = {
        "clustering.similarity_threshold": float,
        "clustering.lookback_hours": int,
        "ai.model": str,
        "ai.batch_size": int,
        "polling.rss_interval_minutes": int,
        "polling.search_interval_minutes": int,
    }

    if key not in allowed:
        return JSONResponse({"error": f"key not allowed, must be one of: {', '.join(sorted(allowed))}"}, status_code=400)

    try:
        typed_value = allowed[key](value)
    except (ValueError, TypeError):
        return JSONResponse({"error": f"invalid value type for {key}"}, status_code=400)

    old_config = load_config()
    keys = key.split(".")
    old_val = old_config
    for k in keys:
        old_val = old_val.get(k)

    set_override(key, typed_value)

    conn = get_conn()
    _log_curation(conn, "config_change", None, {"key": key, "before": old_val, "after": typed_value})
    conn.commit()
    conn.close()

    if key in ("polling.rss_interval_minutes", "polling.search_interval_minutes"):
        job_map = {
            "polling.rss_interval_minutes": "rss",
            "polling.search_interval_minutes": "search",
        }
        job_id = job_map[key]
        job = scheduler.get_job(job_id)
        if job:
            job.reschedule(trigger="interval", minutes=typed_value)
            log.info(f"Rescheduled {job_id} to {typed_value} minutes")

    return {"status": "ok", "key": key, "value": typed_value}

@app.get("/api/river/algorithm")
async def get_algorithm():
    algo = get_active_version()
    return {"active": algo["name"], "versions": list_versions()}

@app.put("/api/river/algorithm")
async def put_algorithm(request: Request):
    body = await request.json()
    version = body.get("version", "").strip()
    if not version:
        return JSONResponse({"error": "version required"}, status_code=400)
    try:
        old = get_active_version()["name"]
        set_active_version(version)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    conn = get_conn()
    _log_curation(conn, "algorithm_change", None, {"before": old, "after": version})
    conn.commit()
    conn.close()
    return {"status": "ok", "version": version}

@app.get("/api/river/curation-log")
async def get_curation_log(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, action, cluster_id, detail, created_at FROM curation_log ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, offset)
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) as c FROM curation_log").fetchone()["c"]
    conn.close()
    return {
        "entries": [{"id": r["id"], "action": r["action"], "cluster_id": r["cluster_id"],
                      "detail": json.loads(r["detail"]) if r["detail"] else None,
                      "created_at": r["created_at"]} for r in rows],
        "total": total,
    }

@app.get("/api/river/stories")
async def river_stories(
    category: str = Query("all", pattern="^(all|markets|tech|politics|world|general)$"),
    sort: str = Query("hot", pattern="^(hot|new)$"),
    show_hidden: bool = Query(False),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Control panel story browser — mirrors homepage query with curation overlays."""
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    algo = get_active_version()

    where = ["sc.expires_at > ?"]
    params = [now]

    if not show_hidden:
        where.append("(co.hidden IS NULL OR co.hidden = 0)")

    if category == "all":
        if algo.get("general_exclusion", True):
            where.append("COALESCE(co.category_override, sc.category) != 'general'")
    else:
        where.append("COALESCE(co.category_override, sc.category) = ?")
        params.append(category)

    # Same sort precedence as homepage: BREAKING > pinned > boosted > normal > demoted
    sort_prefix = """CASE WHEN COALESCE(co.breaking, 0) = 1 THEN 0
                          WHEN COALESCE(co.pinned, 0) = 1 THEN 1
                          WHEN COALESCE(co.boost, 0) > 0 THEN 2
                          WHEN COALESCE(co.boost, 0) < 0 THEN 4
                          ELSE 3 END"""
    # Deprioritize "world" in the ALL view
    world_deprio = "CASE WHEN COALESCE(co.category_override, sc.category) = 'world' THEN 1 ELSE 0 END" if (category == "all" and algo.get("world_deprio", True)) else "0+0"
    hot_score = build_hot_score_sql(algo)
    reputable_sources_sql = build_reputable_sources_sql(algo)
    min_reputable = algo["min_reputable_sources"]
    if sort == "hot":
        reputable_filter = f"""(SELECT COUNT(DISTINCT cs2.source_name) FROM cluster_sources cs2
            WHERE cs2.cluster_id = sc.id AND cs2.source_name IN {reputable_sources_sql}) >= {min_reputable}"""
        if algo.get("scoop_enabled"):
            where.append(f"({reputable_filter} OR (COALESCE(co.boost, 0) > 0 AND co.scoop_boosted_at IS NOT NULL))")
        else:
            where.append(reputable_filter)
        order = f"{sort_prefix}, {world_deprio}, {hot_score} DESC, sc.published_at DESC"
    else:
        order = f"{sort_prefix}, sc.published_at DESC"

    where_clause = " AND ".join(where)

    rows = conn.execute(f"""
        SELECT sc.id, COALESCE(co.headline_override, sc.rewritten_headline) as headline,
               sc.rewritten_headline as auto_headline,
               sc.primary_url, sc.primary_source,
               COALESCE(co.category_override, sc.category) as category,
               sc.category as auto_category,
               sc.source_count, sc.first_seen, sc.last_updated, sc.published_at,
               COALESCE(co.pinned, 0) as pinned,
               COALESCE(co.hidden, 0) as hidden,
               COALESCE(co.locked, 0) as locked,
               COALESCE(co.breaking, 0) as breaking,
               COALESCE(co.boost, 0) as boost,
               co.note,
               co.pin_rank,
               (SELECT ri2.original_headline FROM raw_items ri2 WHERE ri2.cluster_id = sc.id LIMIT 1) as original_headline
        FROM story_clusters sc
        LEFT JOIN curation_overrides co ON co.cluster_id = sc.id
        WHERE {where_clause}
        ORDER BY {order}
        LIMIT ? OFFSET ?
    """, (*params, limit, offset)).fetchall()

    # Batch-fetch sources and items for all clusters (avoid N+1)
    cluster_ids = [r["id"] for r in rows]
    sources_by_cluster = {}
    items_by_cluster = {}
    if cluster_ids:
        ph = ",".join("?" for _ in cluster_ids)
        all_sources = conn.execute(
            f"SELECT cluster_id, source_name, source_url FROM cluster_sources WHERE cluster_id IN ({ph})",
            cluster_ids
        ).fetchall()
        for s in all_sources:
            sources_by_cluster.setdefault(s["cluster_id"], []).append(s)
        all_items = conn.execute(
            f"SELECT cluster_id, id, original_headline, source_url, source_name, published_at FROM raw_items WHERE cluster_id IN ({ph}) ORDER BY published_at DESC",
            cluster_ids
        ).fetchall()
        for i in all_items:
            items_by_cluster.setdefault(i["cluster_id"], []).append(i)

    stories = []
    for r in rows:
        sources = sources_by_cluster.get(r["id"], [])
        items = items_by_cluster.get(r["id"], [])
        other_sources = [{"name": s["source_name"], "url": s["source_url"]} for s in sources if s["source_url"] != r["primary_url"]]
        stories.append({
            "id": r["id"],
            "headline": r["headline"],
            "original_headline": r["original_headline"],
            "auto_headline": r["auto_headline"],
            "url": r["primary_url"],
            "source": r["primary_source"],
            "category": r["category"],
            "auto_category": r["auto_category"],
            "source_count": r["source_count"],
            "first_seen": r["first_seen"],
            "published_at": r["published_at"],
            "last_updated": r["last_updated"],
            "pinned": bool(r["pinned"]),
            "hidden": bool(r["hidden"]),
            "locked": bool(r["locked"]),
            "breaking": bool(r["breaking"]),
            "boost": r["boost"],
            "note": r["note"],
            "other_sources": other_sources,
            "items": [{"id": i["id"], "headline": i["original_headline"], "url": i["source_url"],
                        "source": i["source_name"], "published_at": i["published_at"]} for i in items],
        })

    total = conn.execute(f"SELECT COUNT(*) as c FROM story_clusters sc LEFT JOIN curation_overrides co ON co.cluster_id = sc.id WHERE {where_clause}", params).fetchone()["c"]
    conn.close()
    return {"stories": stories, "total": total}


@app.post("/api/river/swap-primary")
async def swap_primary(request: Request):
    body = await request.json()
    cluster_id = body["cluster_id"]
    source_url = body["source_url"]

    conn = get_conn()
    cluster = conn.execute("SELECT * FROM story_clusters WHERE id=?", (cluster_id,)).fetchone()
    if not cluster:
        conn.close()
        return JSONResponse({"error": "cluster not found"}, status_code=404)

    source = conn.execute("SELECT source_name, source_url FROM cluster_sources WHERE cluster_id=? AND source_url=?",
                          (cluster_id, source_url)).fetchone()
    if not source:
        conn.close()
        return JSONResponse({"error": "source not found in cluster"}, status_code=404)

    old_url = cluster["primary_url"]
    old_source = cluster["primary_source"]
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("UPDATE story_clusters SET primary_url=?, primary_source=?, last_updated=? WHERE id=?",
                 (source["source_url"], source["source_name"], now, cluster_id))
    _log_curation(conn, "swap_primary", cluster_id, {"old_url": old_url, "old_source": old_source, "new_url": source["source_url"], "new_source": source["source_name"]})
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.post("/api/river/flag")
async def flag_cluster(request: Request):
    body = await request.json()
    cluster_id = body["cluster_id"]
    flag_type = body["flag_type"]
    detail = body.get("detail", "")

    valid_flags = {"miscategorized", "bad_headline", "bad_merge", "duplicate", "stale"}
    if flag_type not in valid_flags:
        return JSONResponse({"error": f"invalid flag_type, must be one of: {', '.join(sorted(valid_flags))}"}, status_code=400)

    conn = get_conn()
    if not conn.execute("SELECT 1 FROM story_clusters WHERE id=?", (cluster_id,)).fetchone():
        conn.close()
        return JSONResponse({"error": "cluster not found"}, status_code=404)

    _log_curation(conn, f"flag_{flag_type}", cluster_id, {"flag_type": flag_type, "detail": detail})
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.post("/api/river/breaking")
async def set_breaking(request: Request):
    body = await request.json()
    cluster_id = body["cluster_id"]
    breaking = int(body.get("breaking", 1))

    conn = get_conn()
    if not conn.execute("SELECT 1 FROM story_clusters WHERE id=?", (cluster_id,)).fetchone():
        conn.close()
        return JSONResponse({"error": "cluster not found"}, status_code=404)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO curation_overrides (cluster_id, breaking, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(cluster_id) DO UPDATE SET breaking=?, updated_at=?
    """, (cluster_id, breaking, now, breaking, now))
    action = "breaking_on" if breaking else "breaking_off"
    _log_curation(conn, action, cluster_id, {"breaking": breaking})
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.post("/api/river/not-news")
async def not_news_cluster(request: Request):
    body = await request.json()
    cluster_id = body["cluster_id"]

    conn = get_conn()
    if not conn.execute("SELECT 1 FROM story_clusters WHERE id=?", (cluster_id,)).fetchone():
        conn.close()
        return JSONResponse({"error": "cluster not found"}, status_code=404)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO curation_overrides (cluster_id, hidden, updated_at)
        VALUES (?, 1, ?)
        ON CONFLICT(cluster_id) DO UPDATE SET hidden=1, updated_at=?
    """, (cluster_id, now, now))
    _log_curation(conn, "not_news", cluster_id, {"hidden": 1})
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.post("/api/river/boost")
async def boost_cluster(request: Request):
    body = await request.json()
    cluster_id = body["cluster_id"]
    boost = int(body.get("boost", 0))

    if boost not in (-1, 0, 1):
        return JSONResponse({"error": "boost must be -1, 0, or 1"}, status_code=400)

    conn = get_conn()
    if not conn.execute("SELECT 1 FROM story_clusters WHERE id=?", (cluster_id,)).fetchone():
        conn.close()
        return JSONResponse({"error": "cluster not found"}, status_code=404)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO curation_overrides (cluster_id, boost, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(cluster_id) DO UPDATE SET boost=?, updated_at=?
    """, (cluster_id, boost, now, boost, now))
    action = "boost" if boost > 0 else ("demote" if boost < 0 else "unboost")
    _log_curation(conn, action, cluster_id, {"boost": boost})
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.post("/api/river/note")
async def set_note(request: Request):
    body = await request.json()
    cluster_id = body["cluster_id"]
    note = body.get("note", "").strip()

    conn = get_conn()
    if not conn.execute("SELECT 1 FROM story_clusters WHERE id=?", (cluster_id,)).fetchone():
        conn.close()
        return JSONResponse({"error": "cluster not found"}, status_code=404)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO curation_overrides (cluster_id, note, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(cluster_id) DO UPDATE SET note=?, updated_at=?
    """, (cluster_id, note or None, now, note or None, now))
    _log_curation(conn, "note", cluster_id, {"note": note})
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.post("/api/river/set-expiry")
async def set_expiry(request: Request):
    body = await request.json()
    cluster_id = body["cluster_id"]
    expires_at = body["expires_at"]

    conn = get_conn()
    cluster = conn.execute("SELECT expires_at FROM story_clusters WHERE id=?", (cluster_id,)).fetchone()
    if not cluster:
        conn.close()
        return JSONResponse({"error": "cluster not found"}, status_code=404)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("UPDATE story_clusters SET expires_at=? WHERE id=?", (expires_at, cluster_id))
    conn.execute("""
        INSERT INTO curation_overrides (cluster_id, expiry_override, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(cluster_id) DO UPDATE SET expiry_override=?, updated_at=?
    """, (cluster_id, expires_at, now, expires_at, now))
    _log_curation(conn, "set_expiry", cluster_id, {"old_expires_at": cluster["expires_at"], "new_expires_at": expires_at})
    conn.commit()
    conn.close()
    return {"status": "ok"}


import re as _re

# ── Content Filter CRUD ──────────────────────────────────────────────────

@app.get("/api/river/filters")
async def list_filters(
    filter_type: str | None = None,
    enabled: int | None = None,
):
    conn = get_conn()
    where = []
    params = []
    if filter_type is not None:
        where.append("filter_type = ?")
        params.append(filter_type)
    if enabled is not None:
        where.append("enabled = ?")
        params.append(enabled)
    where_clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = conn.execute(f"SELECT * FROM content_filters{where_clause} ORDER BY id", params).fetchall()
    conn.close()
    return {"filters": [dict(r) for r in rows]}


@app.post("/api/river/filters")
async def create_filter(request: Request):
    body = await request.json()
    name = body.get("name", "").strip()
    pattern = body.get("pattern", "").strip()
    filter_type = body.get("filter_type", "not_news").strip()
    if not name or not pattern:
        return JSONResponse({"error": "name and pattern required"}, status_code=400)
    try:
        _re.compile(pattern)
    except _re.error as e:
        return JSONResponse({"error": f"invalid regex: {e}"}, status_code=400)

    now = datetime.now(timezone.utc).isoformat()
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO content_filters (name, filter_type, pattern, enabled, created_at, updated_at) VALUES (?,?,?,1,?,?)",
        (name, filter_type, pattern, now, now)
    )
    filter_id = cur.lastrowid
    conn.commit()
    conn.close()
    invalidate_filter_cache()
    return {"status": "ok", "id": filter_id}


@app.put("/api/river/filters/{filter_id}")
async def update_filter(filter_id: int, request: Request):
    body = await request.json()
    conn = get_conn()
    existing = conn.execute("SELECT * FROM content_filters WHERE id=?", (filter_id,)).fetchone()
    if not existing:
        conn.close()
        return JSONResponse({"error": "filter not found"}, status_code=404)

    name = body.get("name", existing["name"]).strip()
    pattern = body.get("pattern", existing["pattern"]).strip()
    filter_type = body.get("filter_type", existing["filter_type"]).strip()
    enabled = body.get("enabled", existing["enabled"])

    if "pattern" in body:
        try:
            _re.compile(pattern)
        except _re.error as e:
            conn.close()
            return JSONResponse({"error": f"invalid regex: {e}"}, status_code=400)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE content_filters SET name=?, filter_type=?, pattern=?, enabled=?, updated_at=? WHERE id=?",
        (name, filter_type, pattern, int(enabled), now, filter_id)
    )
    conn.commit()
    conn.close()
    invalidate_filter_cache()
    return {"status": "ok"}


@app.delete("/api/river/filters/{filter_id}")
async def delete_filter(filter_id: int):
    conn = get_conn()
    existing = conn.execute("SELECT * FROM content_filters WHERE id=?", (filter_id,)).fetchone()
    if not existing:
        conn.close()
        return JSONResponse({"error": "filter not found"}, status_code=404)
    conn.execute("DELETE FROM content_filters WHERE id=?", (filter_id,))
    conn.commit()
    conn.close()
    invalidate_filter_cache()
    return {"status": "ok"}


@app.post("/api/river/filters/test")
async def test_filter(request: Request):
    body = await request.json()
    pattern = body.get("pattern", "").strip()
    if not pattern:
        return JSONResponse({"error": "pattern required"}, status_code=400)
    try:
        compiled = _re.compile(pattern, _re.I)
    except _re.error as e:
        return JSONResponse({"error": f"invalid regex: {e}"}, status_code=400)

    conn = get_conn()
    rows = conn.execute(
        "SELECT original_headline, source_name FROM raw_items ORDER BY ingested_at DESC LIMIT 500"
    ).fetchall()
    conn.close()

    matches = []
    for r in rows:
        if compiled.search(r["original_headline"]):
            matches.append({"headline": r["original_headline"], "source": r["source_name"]})
    return {"matches": matches, "tested": len(rows)}


@app.get("/api/river/filtered-items")
async def get_filtered_items(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    search: str | None = None,
):
    conn = get_conn()
    where = []
    params = []
    if search:
        where.append("headline LIKE ?")
        params.append(f"%{search}%")
    where_clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = conn.execute(
        f"SELECT * FROM filtered_items{where_clause} ORDER BY filtered_at DESC LIMIT ? OFFSET ?",
        (*params, limit, offset)
    ).fetchall()
    total = conn.execute(f"SELECT COUNT(*) as c FROM filtered_items{where_clause}", params).fetchone()["c"]
    conn.close()
    return {"items": [dict(r) for r in rows], "total": total}


# ── Reference Site CRUD + Check ───────────────────────────────────────

@app.get("/api/river/references")
async def list_references():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM reference_sites ORDER BY id").fetchall()
    conn.close()
    return {"sites": [dict(r) for r in rows]}


@app.post("/api/river/references")
async def create_reference(request: Request):
    body = await request.json()
    name = body.get("name", "").strip()
    url = body.get("url", "").strip()
    parser = body.get("parser", "").strip()
    max_headlines = int(body.get("max_headlines", 20))

    if not name or not url or not parser:
        return JSONResponse({"error": "name, url, and parser required"}, status_code=400)

    valid_parsers = {"techmeme", "drudge", "apnews", "googlenews"}
    if parser not in valid_parsers:
        return JSONResponse({"error": f"parser must be one of: {', '.join(sorted(valid_parsers))}"}, status_code=400)

    now = datetime.now(timezone.utc).isoformat()
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO reference_sites (name, url, parser, enabled, max_headlines, created_at, updated_at) VALUES (?,?,?,1,?,?,?)",
        (name, url, parser, max_headlines, now, now)
    )
    site_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {"status": "ok", "id": site_id}


@app.put("/api/river/references/{site_id}")
async def update_reference(site_id: int, request: Request):
    body = await request.json()
    conn = get_conn()
    existing = conn.execute("SELECT * FROM reference_sites WHERE id=?", (site_id,)).fetchone()
    if not existing:
        conn.close()
        return JSONResponse({"error": "site not found"}, status_code=404)

    name = body.get("name", existing["name"])
    url = body.get("url", existing["url"])
    parser = body.get("parser", existing["parser"])
    enabled = body.get("enabled", existing["enabled"])
    max_headlines = body.get("max_headlines", existing["max_headlines"])

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE reference_sites SET name=?, url=?, parser=?, enabled=?, max_headlines=?, updated_at=? WHERE id=?",
        (name, url, parser, int(enabled), int(max_headlines), now, site_id)
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.delete("/api/river/references/{site_id}")
async def delete_reference(site_id: int):
    conn = get_conn()
    existing = conn.execute("SELECT * FROM reference_sites WHERE id=?", (site_id,)).fetchone()
    if not existing:
        conn.close()
        return JSONResponse({"error": "site not found"}, status_code=404)
    conn.execute("DELETE FROM reference_sites WHERE id=?", (site_id,))
    conn.execute("DELETE FROM reference_check_log WHERE site_id=?", (site_id,))
    conn.commit()
    conn.close()
    return {"status": "ok"}


_reference_check_running = False

@app.post("/api/river/references/check")
async def trigger_reference_check():
    global _reference_check_running
    if _reference_check_running:
        return JSONResponse({"status": "already_running"}, status_code=409)
    _reference_check_running = True

    try:
        result = await run_reference_check()
    finally:
        _reference_check_running = False
    return {"status": "ok", **result}


@app.get("/api/river/references/log")
async def reference_check_log(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    site_id: int | None = None,
    status: str | None = None,
):
    conn = get_conn()
    where = []
    params = []
    if site_id is not None:
        where.append("site_id = ?")
        params.append(site_id)
    if status is not None:
        # Support comma-separated statuses like "gap_filled,gap_unfilled"
        statuses = [s.strip() for s in status.split(",") if s.strip()]
        placeholders = ",".join("?" for _ in statuses)
        where.append(f"status IN ({placeholders})")
        params.extend(statuses)
    where_clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = conn.execute(
        f"SELECT * FROM reference_check_log{where_clause} ORDER BY checked_at DESC LIMIT ? OFFSET ?",
        (*params, limit, offset)
    ).fetchall()
    total = conn.execute(f"SELECT COUNT(*) as c FROM reference_check_log{where_clause}", params).fetchone()["c"]
    conn.close()
    return {"entries": [dict(r) for r in rows], "total": total}


# ── Feed Source CRUD ──────────────────────────────────────────────────

@app.get("/api/river/sources")
async def list_sources(category: str | None = None):
    conn = get_conn()
    where = []
    params = []
    if category:
        where.append("category = ?")
        params.append(category)
    where_clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = conn.execute(f"SELECT * FROM feed_sources{where_clause} ORDER BY category, name", params).fetchall()
    conn.close()
    sources = []
    for r in rows:
        d = dict(r)
        d["score"] = get_score(r["name"])
        sources.append(d)
    return {"sources": sources}


@app.post("/api/river/sources")
async def create_source(request: Request):
    body = await request.json()
    name = body.get("name", "").strip()
    url = body.get("url", "").strip()
    category = body.get("category", "").strip().lower()
    if not name or not url or not category:
        return JSONResponse({"error": "name, url, and category required"}, status_code=400)
    valid = _get_valid_categories()
    if category not in valid:
        return JSONResponse({"error": f"invalid category, must be one of: {', '.join(sorted(valid))}"}, status_code=400)

    now = datetime.now(timezone.utc).isoformat()
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO feed_sources (name, url, category, enabled, created_at, updated_at) VALUES (?,?,?,1,?,?)",
            (name, url, category, now, now)
        )
        source_id = cur.lastrowid
        conn.commit()
    except Exception as e:
        conn.close()
        return JSONResponse({"error": f"URL already exists: {e}"}, status_code=400)
    conn.close()
    return {"status": "ok", "id": source_id}


@app.put("/api/river/sources/{source_id}")
async def update_source(source_id: int, request: Request):
    body = await request.json()
    conn = get_conn()
    existing = conn.execute("SELECT * FROM feed_sources WHERE id=?", (source_id,)).fetchone()
    if not existing:
        conn.close()
        return JSONResponse({"error": "source not found"}, status_code=404)

    name = body.get("name", existing["name"])
    url = body.get("url", existing["url"])
    category = body.get("category", existing["category"])
    enabled = body.get("enabled", existing["enabled"])

    if "category" in body:
        valid = _get_valid_categories()
        if category not in valid:
            conn.close()
            return JSONResponse({"error": f"invalid category"}, status_code=400)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE feed_sources SET name=?, url=?, category=?, enabled=?, updated_at=? WHERE id=?",
        (name, url, category, int(enabled), now, source_id)
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.delete("/api/river/sources/{source_id}")
async def delete_source(source_id: int):
    conn = get_conn()
    existing = conn.execute("SELECT * FROM feed_sources WHERE id=?", (source_id,)).fetchone()
    if not existing:
        conn.close()
        return JSONResponse({"error": "source not found"}, status_code=404)
    conn.execute("DELETE FROM feed_sources WHERE id=?", (source_id,))
    conn.commit()
    conn.close()
    return {"status": "ok"}


# ── Category CRUD ────────────────────────────────────────────────────

@app.get("/api/river/categories")
async def list_categories():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM categories ORDER BY sort_order, name").fetchall()
    conn.close()
    return {"categories": [dict(r) for r in rows]}


@app.post("/api/river/categories")
async def create_category(request: Request):
    body = await request.json()
    name = body.get("name", "").strip().lower()
    if not name:
        return JSONResponse({"error": "name required"}, status_code=400)

    now = datetime.now(timezone.utc).isoformat()
    conn = get_conn()
    max_order = conn.execute("SELECT MAX(sort_order) as m FROM categories").fetchone()["m"] or 0
    try:
        cur = conn.execute(
            "INSERT INTO categories (name, sort_order, enabled, created_at, updated_at) VALUES (?,?,1,?,?)",
            (name, max_order + 1, now, now)
        )
        cat_id = cur.lastrowid
        conn.commit()
    except Exception:
        conn.close()
        return JSONResponse({"error": "category already exists"}, status_code=400)
    conn.close()
    _invalidate_categories_cache()
    return {"status": "ok", "id": cat_id}


@app.put("/api/river/categories/{cat_id}")
async def update_category(cat_id: int, request: Request):
    body = await request.json()
    conn = get_conn()
    existing = conn.execute("SELECT * FROM categories WHERE id=?", (cat_id,)).fetchone()
    if not existing:
        conn.close()
        return JSONResponse({"error": "category not found"}, status_code=404)

    name = body.get("name", existing["name"]).strip().lower()
    sort_order = body.get("sort_order", existing["sort_order"])
    enabled = body.get("enabled", existing["enabled"])

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE categories SET name=?, sort_order=?, enabled=?, updated_at=? WHERE id=?",
        (name, int(sort_order), int(enabled), now, cat_id)
    )
    conn.commit()
    conn.close()
    _invalidate_categories_cache()
    return {"status": "ok"}


@app.delete("/api/river/categories/{cat_id}")
async def delete_category(cat_id: int):
    conn = get_conn()
    existing = conn.execute("SELECT * FROM categories WHERE id=?", (cat_id,)).fetchone()
    if not existing:
        conn.close()
        return JSONResponse({"error": "category not found"}, status_code=404)
    refs = conn.execute("SELECT COUNT(*) as c FROM feed_sources WHERE category=?", (existing["name"],)).fetchone()["c"]
    if refs > 0:
        conn.close()
        return JSONResponse({"error": f"cannot delete: {refs} source(s) still use this category"}, status_code=400)
    conn.execute("DELETE FROM categories WHERE id=?", (cat_id,))
    conn.commit()
    conn.close()
    _invalidate_categories_cache()
    return {"status": "ok"}


_backfill_running = False

@app.post("/api/river/backfill")
async def river_backfill():
    global _backfill_running
    if _backfill_running:
        return JSONResponse({"status": "already_running"}, status_code=409)
    _backfill_running = True
    async def _run():
        global _backfill_running
        try:
            await backfill_48h()
            await rewrite_pending()
        finally:
            _backfill_running = False
    asyncio.ensure_future(_run())
    return {"status": "started"}

@app.post("/api/river/reboot")
async def river_reboot():
    """Wipe the database and re-run the full boot sequence."""
    global _boot_task

    # Don't allow reboot if one is already running
    if boot_state["phase"] != "ready":
        return JSONResponse({"status": "already_booting"}, status_code=409)

    # Stop scheduler jobs
    for job_id in ("rss", "search", "rewrite", "cleanup", "reference_check", "dedup"):
        job = scheduler.get_job(job_id)
        if job:
            job.remove()
    try:
        scheduler.shutdown(wait=False)
    except Exception:
        pass

    # Wipe all tables (preserve curation_log and content_filters as permanent data)
    conn = get_conn()
    conn.execute("DELETE FROM curation_overrides")
    conn.execute("DELETE FROM cluster_sources")
    conn.execute("DELETE FROM raw_items")
    conn.execute("DELETE FROM story_clusters")
    conn.execute("DELETE FROM filtered_items")
    conn.execute("DELETE FROM reference_check_log")
    conn.execute("UPDATE reference_sites SET last_checked=NULL, last_found=0, last_gaps=0")
    conn.commit()
    conn.close()
    log.info("Reboot: database wiped (filters and reference sites preserved)")

    # Reset boot state
    boot_state["phase"] = "reference_check"
    boot_state["detail"] = ""
    boot_state["clusters"] = 0
    boot_state["pending"] = 0
    boot_state["reference_progress"] = 0
    boot_state["polling_progress"] = 0
    boot_state["clustering_progress"] = 0
    boot_state["rewriting_progress"] = 0

    # Re-run boot sequence + scheduler
    _boot_task = asyncio.create_task(startup_backfill())
    asyncio.create_task(_start_scheduler_after_boot())

    return {"status": "rebooting"}


async def cleanup_expired():
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("DELETE FROM curation_overrides WHERE cluster_id IN (SELECT id FROM story_clusters WHERE expires_at < ?)", (now,))
    conn.execute("DELETE FROM cluster_sources WHERE cluster_id IN (SELECT id FROM story_clusters WHERE expires_at < ?)", (now,))
    conn.execute("DELETE FROM raw_items WHERE cluster_id IN (SELECT id FROM story_clusters WHERE expires_at < ?)", (now,))
    conn.execute("DELETE FROM story_clusters WHERE expires_at < ?", (now,))
    # Clean up filtered_items older than 7 days
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    conn.execute("DELETE FROM filtered_items WHERE filtered_at < ?", (cutoff,))
    conn.execute("DELETE FROM reference_check_log WHERE checked_at < ?", (cutoff,))
    conn.commit()
    conn.close()
    log.info("Cleaned up expired stories, old filtered items, and old reference logs")


async def deduplicate_clusters():
    """Hourly job: merge duplicate clusters + expire stale scoop boosts."""
    try:
        conn = get_conn()
        merged = merge_existing_clusters(conn)
        if merged:
            log.info(f"Dedup job: merged {merged} duplicate clusters")

        # Expire scoop boosts that haven't generated enough follow-on coverage
        algo = get_active_version()
        if algo.get("scoop_enabled"):
            boost_hours = algo.get("scoop_boost_hours", 4)
            min_sources = algo.get("scoop_min_sources_after", 2)
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=boost_hours)).isoformat()
            stale_scoops = conn.execute("""
                SELECT co.cluster_id, sc.source_count, sc.rewritten_headline
                FROM curation_overrides co
                JOIN story_clusters sc ON sc.id = co.cluster_id
                WHERE co.scoop_boosted_at IS NOT NULL
                  AND co.scoop_boosted_at < ?
                  AND co.boost = 1
                  AND sc.source_count < ?
            """, (cutoff, min_sources)).fetchall()
            for row in stale_scoops:
                conn.execute(
                    "UPDATE curation_overrides SET boost=0, updated_at=? WHERE cluster_id=?",
                    (datetime.now(timezone.utc).isoformat(), row["cluster_id"])
                )
                log.info(f"Scoop expired: {row['rewritten_headline'][:60]} (only {row['source_count']} source(s))")
            if stale_scoops:
                conn.commit()

        conn.close()
    except Exception as e:
        log.warning(f"Dedup job error: {e}")
