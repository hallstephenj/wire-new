import logging
import asyncio
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from wire.db import init_db, get_conn
from wire.config import load_config
from wire.ingest import poll_feeds, search_sweep
from wire.rewrite import rewrite_pending
from wire.events import snapshot as events_snapshot
from wire.scores import all_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("wire")

scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    cfg = load_config()

    scheduler.add_job(poll_feeds, "interval", minutes=cfg["polling"]["rss_interval_minutes"], id="rss", next_run_time=datetime.now(timezone.utc))
    scheduler.add_job(search_sweep, "interval", minutes=cfg["polling"]["search_interval_minutes"], id="search", next_run_time=datetime.now(timezone.utc))
    scheduler.add_job(rewrite_pending, "interval", minutes=2, id="rewrite", next_run_time=datetime.now(timezone.utc))
    scheduler.add_job(cleanup_expired, "interval", hours=cfg["polling"]["cleanup_interval_hours"], id="cleanup")
    scheduler.start()
    log.info("WIRE started")
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html_path.read_text())

@app.get("/api/stories")
async def get_stories(
    sort: str = Query("hot", pattern="^(hot|new)$"),
    category: str = Query("all", pattern="^(all|markets|tech|politics|world|general)$"),
    since: str | None = None,
    limit: int = Query(100, ge=1, le=500),
):
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()

    where = ["sc.expires_at > ?"]
    params = [now]

    if category == "all":
        # Exclude "general" bucket from the all view
        where.append("sc.category != 'general'")
    else:
        where.append("sc.category = ?")
        params.append(category)

    if since:
        where.append("sc.last_updated > ?")
        params.append(since)

    if sort == "hot":
        age_cutoff = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        order = f"CASE WHEN sc.published_at > '{age_cutoff}' THEN 0 ELSE 1 END, sc.source_count DESC, sc.published_at DESC"
    else:
        order = "sc.published_at DESC"
    where_clause = " AND ".join(where)

    rows = conn.execute(f"""
        SELECT sc.id, sc.rewritten_headline, sc.primary_url, sc.primary_source,
               sc.category, sc.source_count, sc.first_seen, sc.last_updated, sc.published_at,
               (SELECT ri2.original_headline FROM raw_items ri2 WHERE ri2.cluster_id = sc.id LIMIT 1) as original_headline
        FROM story_clusters sc
        WHERE {where_clause}
        ORDER BY {order}
        LIMIT ?
    """, (*params, limit)).fetchall()

    stories = []
    for r in rows:
        sources = conn.execute(
            "SELECT source_name, MIN(source_url) as source_url FROM cluster_sources WHERE cluster_id=? AND source_url != ? GROUP BY source_name LIMIT 5",
            (r["id"], r["primary_url"])
        ).fetchall()
        items = conn.execute(
            "SELECT original_headline, source_url, source_name, published_at FROM raw_items WHERE cluster_id=? ORDER BY published_at DESC",
            (r["id"],)
        ).fetchall()
        stories.append({
            "id": r["id"],
            "headline": r["rewritten_headline"],
            "original_headline": r["original_headline"],
            "url": r["primary_url"],
            "source": r["primary_source"],
            "category": r["category"],
            "source_count": r["source_count"],
            "first_seen": r["first_seen"],
            "published_at": r["published_at"],
            "last_updated": r["last_updated"],
            "other_sources": [{"name": s["source_name"], "url": s["source_url"]} for s in sources],
            "items": [{"headline": i["original_headline"], "url": i["source_url"], "source": i["source_name"], "published_at": i["published_at"]} for i in items],
        })

    total = conn.execute(f"SELECT COUNT(*) as c FROM story_clusters sc WHERE {where_clause}", params).fetchone()["c"]
    conn.close()

    return {"stories": stories, "updated_at": now, "total": total}

@app.get("/api/health")
async def health():
    conn = get_conn()
    clusters = conn.execute("SELECT COUNT(*) as c FROM story_clusters").fetchone()["c"]
    items = conn.execute("SELECT COUNT(*) as c FROM raw_items").fetchone()["c"]
    conn.close()
    return {"status": "ok", "clusters": clusters, "raw_items": items}

@app.get("/river", response_class=HTMLResponse)
async def river():
    html_path = Path(__file__).parent / "templates" / "river.html"
    return HTMLResponse(html_path.read_text())

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
    conn.close()

    jobs = {}
    for job_id in ("rss", "search", "rewrite", "cleanup"):
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
        "db": {"clusters": clusters, "raw_items": raw_items, "pending_rewrites": pending},
        "config": {
            "similarity_threshold": cfg["clustering"]["similarity_threshold"],
            "lookback_hours": cfg["clustering"]["lookback_hours"],
            "model": cfg["ai"]["model"],
            "batch_size": cfg["ai"]["batch_size"],
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

@app.get("/api/scores")
async def scores():
    data = all_scores()
    return {"scores": [{"source": name, **breakdown} for name, breakdown in data.items()]}

async def cleanup_expired():
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("DELETE FROM cluster_sources WHERE cluster_id IN (SELECT id FROM story_clusters WHERE expires_at < ?)", (now,))
    conn.execute("DELETE FROM raw_items WHERE cluster_id IN (SELECT id FROM story_clusters WHERE expires_at < ?)", (now,))
    conn.execute("DELETE FROM story_clusters WHERE expires_at < ?", (now,))
    conn.commit()
    conn.close()
    log.info("Cleaned up expired stories")
