import uuid
import logging
from datetime import datetime, timezone, timedelta
from time import mktime

import feedparser
import httpx

from wire.db import get_conn
from wire.config import load_feeds, load_config
from wire.cluster import assign_cluster

log = logging.getLogger("wire.ingest")

def parse_published(entry) -> str:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()

async def poll_feeds():
    feeds_cfg = load_feeds()
    log.info("Polling RSS feeds...")
    count = 0
    for category, feeds in feeds_cfg["feeds"].items():
        for feed in feeds:
            try:
                n = await _poll_single_feed(feed["url"], feed["name"], category)
                count += n
            except Exception as e:
                log.warning(f"Feed error {feed['name']}: {e}")
    log.info(f"Ingested {count} new items")

async def _poll_single_feed(url: str, name: str, category: str) -> int:
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": "WIRE/1.0"})
        resp.raise_for_status()

    feed = feedparser.parse(resp.text)
    conn = get_conn()
    count = 0

    for entry in feed.entries[:50]:
        link = getattr(entry, "link", None)
        title = getattr(entry, "title", None)
        if not link or not title:
            continue

        # Exact URL dedup
        existing = conn.execute("SELECT id FROM raw_items WHERE source_url=?", (link,)).fetchone()
        if existing:
            continue

        item_id = str(uuid.uuid4())
        published = parse_published(entry)
        now = datetime.now(timezone.utc).isoformat()

        cluster_id = assign_cluster(conn, title, link, name, category)

        conn.execute(
            "INSERT INTO raw_items (id, source_url, source_name, original_headline, published_at, ingested_at, feed_url, category, cluster_id) VALUES (?,?,?,?,?,?,?,?,?)",
            (item_id, link, name, title, published, now, url, category, cluster_id)
        )
        count += 1

    conn.commit()
    conn.close()
    return count

async def search_sweep():
    cfg = load_config()
    queries = cfg.get("search", {}).get("queries", [])
    log.info("Running search sweep...")
    # Use Google News RSS as search proxy
    count = 0
    for query in queries:
        try:
            cat = _query_to_category(query)
            url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
            n = await _poll_single_feed(url, "Google News", cat)
            count += n
        except Exception as e:
            log.warning(f"Search sweep error for '{query}': {e}")
    log.info(f"Search sweep ingested {count} new items")

def _query_to_category(query: str) -> str:
    q = query.lower()
    if "stock" in q or "market" in q:
        return "markets"
    if "tech" in q:
        return "tech"
    if "politic" in q:
        return "politics"
    if "world" in q:
        return "world"
    return "world"
