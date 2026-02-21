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

# ── Content-based category classification ─────────────────────────────────
# When a general-purpose source (NYT, Reuters, etc.) is in the tech feed,
# we need to verify the headline is actually tech-related.

TECH_KEYWORDS = {
    "ai", "artificial intelligence", "algorithm", "android", "api", "app",
    "apple", "autonomous", "bitcoin", "blockchain", "browser", "bug", "byte",
    "chatbot", "chatgpt", "chip", "chipmaker", "claude", "cloud", "code",
    "computer", "computing", "crypto", "cryptocurrency", "cyber", "data",
    "database", "deepfake", "developer", "digital", "drone", "elon musk",
    "encryption", "ethernet", "facebook", "firmware", "gemini", "github",
    "google", "gpu", "hack", "hardware", "intel", "internet", "ios",
    "iphone", "laptop", "linux", "llm", "machine learning", "malware",
    "meta", "microsoft", "model", "neural", "nvidia", "openai", "open source",
    "password", "phishing", "pixel", "platform", "privacy", "processor",
    "programming", "quantum", "ransomware", "robot", "robotics", "saas",
    "samsung", "semiconductor", "server", "silicon", "smartphone", "software",
    "spacex", "startup", "streaming", "tech", "technology", "tesla", "tiktok",
    "twitter", "uber", "venture capital", "virtual reality", "vpn", "web",
    "wifi", "windows", "x.com", "youtube", "zero-day",
}

MARKETS_KEYWORDS = {
    "stock", "shares", "earnings", "revenue", "profit", "quarterly", "ipo",
    "nasdaq", "dow", "s&p", "fed", "interest rate", "inflation", "gdp",
    "bond", "yield", "treasury", "investor", "dividend", "market cap",
    "bull", "bear", "rally", "crash", "trading", "wall street", "forex",
    "commodity", "oil price", "gold price", "etf", "hedge fund",
}

POLITICS_KEYWORDS = {
    "congress", "senate", "house", "president", "white house", "democrat",
    "republican", "gop", "election", "vote", "ballot", "legislation",
    "bill", "law", "regulation", "executive order", "supreme court",
    "impeach", "campaign", "lobby", "partisan", "bipartisan", "caucus",
    "filibuster", "amendment",
}

WORLD_KEYWORDS = {
    "war", "conflict", "troops", "military", "nato", "united nations",
    "embassy", "diplomat", "sanctions", "ceasefire", "refugee", "humanitarian",
    "earthquake", "tsunami", "hurricane", "flood", "famine",
}

# Sources that publish across many topics — these need content filtering
GENERAL_SOURCES = {
    "Bloomberg", "New York Times", "Reuters", "Financial Times",
    "Washington Post", "The Guardian", "CNBC", "BBC", "BBC World",
    "Axios", "Forbes", "Fortune", "NBC News", "Business Insider",
    "The Atlantic", "Semafor", "South China Morning Post", "Nikkei Asia",
    "Nature", "Google News", "Al Jazeera", "Fox Business",
}


def classify_headline(title: str, feed_category: str, source_name: str) -> str:
    """
    For general-purpose sources, verify the headline matches the feed category.
    Specialist sources (TechCrunch, Ars Technica, etc.) keep their feed category.
    """
    if source_name not in GENERAL_SOURCES:
        return feed_category

    title_lower = title.lower()

    def score(keywords):
        return sum(1 for kw in keywords if kw in title_lower)

    scores = {
        "tech": score(TECH_KEYWORDS),
        "markets": score(MARKETS_KEYWORDS),
        "politics": score(POLITICS_KEYWORDS),
        "world": score(WORLD_KEYWORDS),
    }

    best = max(scores, key=scores.get)

    # If the best category has at least 1 keyword match, use it
    if scores[best] > 0:
        return best

    # No keyword matches at all — keep the feed category
    # (better to show it somewhere than drop it)
    return feed_category

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

        cluster_id = assign_cluster(conn, title, link, name, category, published_at=published)

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
