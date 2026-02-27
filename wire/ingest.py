import re
import uuid
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from time import mktime

import feedparser
import httpx

from wire.db import get_conn
from wire.config import load_feeds, load_config
from wire.cluster import assign_cluster
from wire.events import push as ev
from wire.rewrite import urgent_rewrite

log = logging.getLogger("wire.ingest")

# ── Shared HTTP client (connection pooling) ───────────────────────────────
# Initialized once at app startup via init_http_client(), reused for all
# feed polling, search sweeps, and Google News URL resolution.

_shared_client: httpx.AsyncClient | None = None


def init_http_client():
    """Create the shared HTTP client. Call once at app startup."""
    global _shared_client
    _shared_client = httpx.AsyncClient(
        timeout=15,
        follow_redirects=True,
        headers={"User-Agent": "DINWIRE/1.0"},
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    )


async def close_http_client():
    """Close the shared HTTP client. Call at app shutdown."""
    global _shared_client
    if _shared_client:
        await _shared_client.aclose()
        _shared_client = None


def _get_client() -> httpx.AsyncClient:
    """Return the shared client, or create a fallback if not initialized."""
    if _shared_client is not None:
        return _shared_client
    # Fallback for testing or direct script usage
    return httpx.AsyncClient(
        timeout=15, follow_redirects=True, headers={"User-Agent": "DINWIRE/1.0"}
    )


# Normalize common source name variants to a canonical form
SOURCE_ALIASES = {
    "The Wall Street Journal": "Wall Street Journal",
    "WSJ": "Wall Street Journal",
    "WSJ Markets": "Wall Street Journal",
    "The New York Times": "New York Times",
    "NYT": "New York Times",
    "The Washington Post": "Washington Post",
    "WaPo": "Washington Post",
    "The Verge": "The Verge",
    "The Guardian": "The Guardian",
    "The Atlantic": "The Atlantic",
    "The Hill": "The Hill",
    "The Block": "The Block",
    "The Record": "The Record",
    "The Information": "The Information",
    "Reuters Business": "Reuters",
    "Reuters Politics": "Reuters",
    "Reuters World": "Reuters",
    "AP Politics": "AP",
    "AP Top News": "AP",
    "SCMP": "South China Morning Post",
}

def _normalize_source(name: str) -> str:
    return SOURCE_ALIASES.get(name, name)


# ── Google News URL unwrapping ────────────────────────────────────────────
# Google News RSS entries have redirect URLs (news.google.com/rss/articles/...).
# We resolve them to actual article URLs via HEAD requests with an LRU cache.
# If the server is behind a GDPR consent wall (e.g. hosted in EU), resolution
# will always fail — we detect this once and skip all future attempts.

_gnews_blocked = False  # set True after first consent-wall hit

# Bounded cache for resolved Google News URLs
_gnews_url_cache: dict[str, str] = {}
_GNEWS_CACHE_MAX = 500


def is_gnews_blocked() -> bool:
    """Return True if Google News is blocked by a consent wall."""
    return _gnews_blocked

# Suppress noisy httpx INFO logs for redirect chains
logging.getLogger("httpx").setLevel(logging.WARNING)


async def _unwrap_google_news_url(url: str, client: httpx.AsyncClient) -> str:
    """Resolve a Google News redirect URL to the actual article URL.
    Returns original URL if resolution fails or consent wall detected."""
    global _gnews_blocked

    if "news.google.com" not in url:
        return url

    # After first consent-wall hit, skip all future HTTP attempts
    if _gnews_blocked:
        return url

    if url in _gnews_url_cache:
        return _gnews_url_cache[url]

    try:
        resp = await asyncio.wait_for(
            client.head(url, follow_redirects=True, headers={
                "Accept-Language": "en-US,en;q=0.9",
            }),
            timeout=8,
        )
        resolved = str(resp.url)
        # Only cache if we actually resolved to the real article —
        # reject Google domains (consent walls, redirects back to gnews)
        if "google.com" not in resolved and "google.nl" not in resolved:
            # Bounded cache: evict oldest half when full
            if len(_gnews_url_cache) >= _GNEWS_CACHE_MAX:
                keys = list(_gnews_url_cache)[:_GNEWS_CACHE_MAX // 2]
                for k in keys:
                    del _gnews_url_cache[k]
            _gnews_url_cache[url] = resolved
            return resolved
        elif "consent.google" in resolved:
            _gnews_blocked = True
            log.warning("Google News consent wall detected — skipping URL resolution for this session")
    except Exception:
        pass

    return url

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


# ── DB-backed content filter system ────────────────────────────────────────
# Replaces the old hardcoded _NOT_NEWS_PATTERNS with cached DB lookups.

import time as _time

_filter_cache = {"filters": [], "loaded_at": 0, "pattern_hash": None}
_FILTER_CACHE_TTL = 60  # seconds

def _load_filters():
    """Load enabled filters from DB with TTL cache.

    Compiled regexes persist across refreshes — only recompiled when
    the underlying DB patterns actually change.
    """
    now = _time.time()
    if _filter_cache["filters"] and (now - _filter_cache["loaded_at"]) < _FILTER_CACHE_TTL:
        return _filter_cache["filters"]

    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, filter_type, pattern FROM content_filters WHERE enabled = 1"
    ).fetchall()
    conn.close()

    # Check if patterns changed since last compile
    new_hash = hash(tuple((r["id"], r["pattern"]) for r in rows))
    if new_hash == _filter_cache["pattern_hash"] and _filter_cache["filters"]:
        # Patterns unchanged — just refresh the TTL, keep compiled regexes
        _filter_cache["loaded_at"] = now
        return _filter_cache["filters"]

    filters = []
    for r in rows:
        try:
            compiled = re.compile(r["pattern"], re.I)
            filters.append({
                "id": r["id"],
                "name": r["name"],
                "filter_type": r["filter_type"],
                "pattern": r["pattern"],
                "regex": compiled,
            })
        except re.error:
            log.warning(f"Invalid filter regex (id={r['id']}): {r['pattern']}")

    _filter_cache["filters"] = filters
    _filter_cache["loaded_at"] = now
    _filter_cache["pattern_hash"] = new_hash
    return filters


def _check_filters(title):
    """Return matching filter dict or None."""
    filters = _load_filters()
    for f in filters:
        if f["regex"].search(title):
            return f
    return None


def invalidate_filter_cache():
    """Clear the filter cache so next check reloads from DB."""
    _filter_cache["filters"] = []
    _filter_cache["loaded_at"] = 0


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

def _load_feeds_from_db():
    """Load enabled feed sources from DB, falling back to YAML if table is empty."""
    conn = get_conn()
    try:
        rows = conn.execute("SELECT name, url, category FROM feed_sources WHERE enabled=1").fetchall()
    except Exception:
        rows = []
    conn.close()
    if rows:
        return [(r["category"], {"name": r["name"], "url": r["url"]}) for r in rows]
    # Fallback to YAML
    feeds_cfg = load_feeds()
    return [(cat, f) for cat, feeds in feeds_cfg["feeds"].items() for f in feeds]

async def poll_feeds(on_progress=None):
    log.info("Polling RSS feeds...")
    ev("ingest_start", job="rss")
    all_feeds = _load_feeds_from_db()
    total = len(all_feeds)
    completed = 0
    count = 0
    sem = asyncio.Semaphore(8)

    async def _poll_one(category, feed):
        nonlocal completed, count
        async with sem:
            try:
                n = await _poll_single_feed(feed["url"], feed["name"], category)
                count += n
            except Exception as e:
                log.warning(f"Feed error {feed['name']}: {e}")
            completed += 1
            if on_progress:
                on_progress(completed, total)

    await asyncio.gather(*[_poll_one(cat, f) for cat, f in all_feeds])
    ev("ingest_done", job="rss", items=count)
    log.info(f"Ingested {count} new items")
    # Immediately rewrite any high-coverage clusters that need it
    if count > 0:
        try:
            await urgent_rewrite()
        except Exception as e:
            log.warning(f"Urgent rewrite after poll failed: {e}")

async def _poll_single_feed(url: str, name: str, category: str) -> int:
    client = _get_client()
    resp = await client.get(url)
    resp.raise_for_status()

    feed = feedparser.parse(resp.text)
    conn = get_conn()
    count = 0

    for entry in feed.entries[:50]:
        link = getattr(entry, "link", None)
        title = getattr(entry, "title", None)
        if not link or not title:
            continue

        # Extract real source from Google News entries
        item_name = name
        if name == "Google News":
            # feedparser exposes <source> element; title also ends with " - Source"
            if hasattr(entry, "source") and hasattr(entry.source, "title"):
                item_name = entry.source.title
                title = title.rsplit(" - " + item_name, 1)[0] if title.endswith(" - " + item_name) else title
            elif " - " in title:
                title, item_name = title.rsplit(" - ", 1)

        item_name = _normalize_source(item_name)

        # Resolve Google News redirect URLs to actual article URLs
        if "news.google.com" in link:
            link = await _unwrap_google_news_url(link, client)

        # Exact URL dedup
        existing = conn.execute("SELECT id FROM raw_items WHERE source_url=?", (link,)).fetchone()
        if existing:
            continue

        # Headline + source dedup (catches Google News duplicates with different URLs)
        existing = conn.execute(
            "SELECT id FROM raw_items WHERE original_headline=? AND source_name=?",
            (title, item_name)
        ).fetchone()
        if existing:
            continue

        # Filter out product reviews, deals, affiliate/sponsored content
        matched_filter = _check_filters(title)
        if matched_filter:
            log.debug(f"Filtered ({matched_filter['name']}): {title}")
            now_f = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO filtered_items (headline, source_name, source_url, feed_url, category, filter_id, filter_name, filter_pattern, filtered_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (title, item_name, link, url, category, matched_filter["id"], matched_filter["name"], matched_filter["pattern"], now_f)
            )
            continue

        item_id = str(uuid.uuid4())
        published = parse_published(entry)
        now = datetime.now(timezone.utc).isoformat()

        cluster_id = assign_cluster(conn, title, link, item_name, category, published_at=published)

        conn.execute(
            "INSERT INTO raw_items (id, source_url, source_name, original_headline, published_at, ingested_at, feed_url, category, cluster_id) VALUES (?,?,?,?,?,?,?,?,?)",
            (item_id, link, item_name, title, published, now, url, category, cluster_id)
        )
        count += 1

    conn.commit()
    conn.close()
    return count

async def search_sweep(on_progress=None):
    cfg = load_config()
    queries = cfg.get("search", {}).get("queries", [])
    log.info("Running search sweep...")
    ev("ingest_start", job="search")
    # Use Google News RSS as search proxy — skip if consent wall active
    count = 0
    if _gnews_blocked:
        log.info("Skipping search sweep (Google News consent wall active)")
        if on_progress:
            on_progress(len(queries), len(queries))
    else:
        for idx, query in enumerate(queries):
            try:
                cat = _query_to_category(query)
                url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
                n = await _poll_single_feed(url, "Google News", cat)
                count += n
            except Exception as e:
                log.warning(f"Search sweep error for '{query}': {e}")
            if on_progress:
                on_progress(idx + 1, len(queries))
    ev("ingest_done", job="search", items=count)
    log.info(f"Search sweep ingested {count} new items")
    if count > 0:
        try:
            await urgent_rewrite()
        except Exception as e:
            log.warning(f"Urgent rewrite after search failed: {e}")

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


# ── Deep backfill via Google News search ───────────────────────────────────

BACKFILL_QUERIES = {
    "tech": [
        "AI artificial intelligence",
        "cybersecurity breach hack",
        "Apple Google Microsoft",
        "startup funding venture capital",
        "semiconductor chip nvidia",
        "OpenAI Anthropic AI",
        "social media regulation",
        "tech layoffs",
        "Meta Facebook Instagram",
        "Amazon AWS cloud",
        "Tesla Elon Musk",
        "TikTok ByteDance",
        "data breach privacy",
        "antitrust big tech",
        "robotics automation",
        "space SpaceX launch",
    ],
    "markets": [
        "stock market today",
        "earnings report quarterly",
        "IPO stock offering",
        "federal reserve interest rate",
        "cryptocurrency bitcoin ethereum",
        "oil prices energy",
        "S&P 500 Dow Jones Nasdaq",
        "inflation CPI economic data",
        "merger acquisition deal",
        "bonds treasury yield",
        "housing market mortgage rates",
        "jobs report unemployment",
        "retail sales consumer spending",
        "bank earnings financial",
    ],
    "politics": [
        "congress legislation bill",
        "supreme court ruling",
        "Trump tariffs trade",
        "election 2026",
        "government shutdown",
        "White House executive order",
        "DOGE Elon Musk government",
        "immigration border policy",
        "Pentagon military defense",
        "FBI DOJ investigation",
        "sanctions foreign policy",
        "state governor legislation",
    ],
    "world": [
        "breaking news today",
        "breaking news this week",
        "Ukraine Russia war",
        "China Taiwan",
        "Middle East conflict",
        "NATO military",
        "climate change summit",
        "Israel Gaza Hamas",
        "Europe EU policy",
        "India Modi economy",
        "Africa coup conflict",
        "Latin America Brazil Mexico",
        "North Korea missile nuclear",
        "humanitarian crisis refugees",
        "earthquake hurricane disaster",
        "pandemic health WHO",
    ],
}


async def backfill_48h(on_progress=None):
    """Deep sweep via Google News to backfill the last 72 hours of stories."""
    log.info("Starting deep backfill sweep...")
    ev("ingest_start", job="backfill")
    total = 0
    completed = 0
    # Build flat list of all operations for progress tracking
    # Use time-scoped queries to reach back further than default RSS window
    ops = []
    if not _gnews_blocked:
        for category, queries in BACKFILL_QUERIES.items():
            for query in queries:
                for time_scope in ["when:1d", "when:2d", "when:3d"]:
                    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}+{time_scope}&hl=en-US&gl=US&ceid=US:en"
                    ops.append((url, "Google News", category))
    else:
        log.info("Skipping Google News backfill queries (consent wall active)")
    db_feeds = _load_feeds_from_db()
    for category, feed in db_feeds:
        ops.append((feed["url"], feed["name"], category))

    sem = asyncio.Semaphore(8)
    total_ops = len(ops)
    _progress_batch = max(1, total_ops // 20)  # update progress ~20 times

    async def _backfill_one(url, name, category):
        nonlocal total, completed
        async with sem:
            try:
                n = await _poll_single_feed(url, name, category)
                total += n
            except Exception as e:
                log.warning(f"Backfill error for '{name}': {e}")
            completed += 1
            if on_progress and completed % _progress_batch == 0:
                on_progress(completed, total_ops)

    await asyncio.gather(*[_backfill_one(u, n, c) for u, n, c in ops])
    if on_progress:
        on_progress(total_ops, total_ops)
    ev("ingest_done", job="backfill", items=total)
    log.info(f"Backfill complete: {total} new items")
