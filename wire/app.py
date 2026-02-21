import logging
import asyncio
from datetime import datetime, timezone
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

    order = "sc.source_count DESC, sc.published_at DESC" if sort == "hot" else "sc.published_at DESC"
    where_clause = " AND ".join(where)

    rows = conn.execute(f"""
        SELECT sc.id, sc.rewritten_headline, sc.primary_url, sc.primary_source,
               sc.category, sc.source_count, sc.first_seen, sc.last_updated, sc.published_at
        FROM story_clusters sc
        WHERE {where_clause}
        ORDER BY {order}
        LIMIT ?
    """, (*params, limit)).fetchall()

    stories = []
    for r in rows:
        sources = conn.execute(
            "SELECT source_name, source_url FROM cluster_sources WHERE cluster_id=? AND source_url != ? LIMIT 5",
            (r["id"], r["primary_url"])
        ).fetchall()
        stories.append({
            "id": r["id"],
            "headline": r["rewritten_headline"],
            "url": r["primary_url"],
            "source": r["primary_source"],
            "category": r["category"],
            "source_count": r["source_count"],
            "first_seen": r["first_seen"],
            "published_at": r["published_at"],
            "last_updated": r["last_updated"],
            "other_sources": [{"name": s["source_name"], "url": s["source_url"]} for s in sources]
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

async def cleanup_expired():
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("DELETE FROM cluster_sources WHERE cluster_id IN (SELECT id FROM story_clusters WHERE expires_at < ?)", (now,))
    conn.execute("DELETE FROM raw_items WHERE cluster_id IN (SELECT id FROM story_clusters WHERE expires_at < ?)", (now,))
    conn.execute("DELETE FROM story_clusters WHERE expires_at < ?", (now,))
    conn.commit()
    conn.close()
    log.info("Cleaned up expired stories")
