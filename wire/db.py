import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from wire.config import load_config, load_feeds

_DB_PATH = None

def get_db_path():
    global _DB_PATH
    if _DB_PATH is None:
        cfg = load_config()
        _DB_PATH = Path(__file__).resolve().parent.parent / cfg["database"]["path"]
        # Ensure parent directory exists
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _DB_PATH

def get_conn():
    conn = sqlite3.connect(str(get_db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")   # 64MB page cache
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS story_clusters (
        id TEXT PRIMARY KEY,
        rewritten_headline TEXT,
        primary_url TEXT,
        primary_source TEXT,
        category TEXT,
        source_count INTEGER DEFAULT 1,
        first_seen TEXT,
        last_updated TEXT,
        expires_at TEXT,
        published_at TEXT
    );
    CREATE TABLE IF NOT EXISTS raw_items (
        id TEXT PRIMARY KEY,
        source_url TEXT,
        source_name TEXT,
        original_headline TEXT,
        published_at TEXT,
        ingested_at TEXT,
        feed_url TEXT,
        category TEXT,
        cluster_id TEXT REFERENCES story_clusters(id)
    );
    CREATE TABLE IF NOT EXISTS cluster_sources (
        cluster_id TEXT REFERENCES story_clusters(id),
        source_name TEXT,
        source_url TEXT,
        added_at TEXT,
        PRIMARY KEY (cluster_id, source_url)
    );
    CREATE INDEX IF NOT EXISTS idx_csources_cluster_name ON cluster_sources(cluster_id, source_name);
    CREATE INDEX IF NOT EXISTS idx_clusters_category ON story_clusters(category);
    CREATE INDEX IF NOT EXISTS idx_clusters_expires ON story_clusters(expires_at);
    CREATE INDEX IF NOT EXISTS idx_clusters_source_count ON story_clusters(source_count DESC);
    CREATE INDEX IF NOT EXISTS idx_clusters_last_updated ON story_clusters(last_updated DESC);
    CREATE INDEX IF NOT EXISTS idx_raw_source_url ON raw_items(source_url);
    CREATE INDEX IF NOT EXISTS idx_raw_headline_source ON raw_items(original_headline, source_name);
    CREATE INDEX IF NOT EXISTS idx_raw_cluster_published ON raw_items(cluster_id, published_at DESC);
    CREATE INDEX IF NOT EXISTS idx_raw_ingested ON raw_items(ingested_at DESC);
    CREATE INDEX IF NOT EXISTS idx_clusters_published ON story_clusters(published_at DESC);

    CREATE TABLE IF NOT EXISTS curation_overrides (
        cluster_id TEXT PRIMARY KEY REFERENCES story_clusters(id),
        headline_override TEXT,
        category_override TEXT,
        pinned INTEGER DEFAULT 0,
        hidden INTEGER DEFAULT 0,
        pin_rank INTEGER DEFAULT 0,
        locked INTEGER DEFAULT 0,
        updated_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_curation_pinned ON curation_overrides(pinned);
    CREATE INDEX IF NOT EXISTS idx_curation_hidden ON curation_overrides(hidden);
    CREATE INDEX IF NOT EXISTS idx_curation_composite ON curation_overrides(cluster_id, hidden, pinned, boost);

    CREATE TABLE IF NOT EXISTS curation_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        action TEXT NOT NULL,
        cluster_id TEXT,
        detail TEXT,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_curation_log_created ON curation_log(created_at DESC);

    CREATE TABLE IF NOT EXISTS content_filters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        filter_type TEXT DEFAULT 'not_news',
        pattern TEXT NOT NULL,
        enabled INTEGER DEFAULT 1,
        created_at TEXT,
        updated_at TEXT
    );

    CREATE TABLE IF NOT EXISTS filtered_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        headline TEXT,
        source_name TEXT,
        source_url TEXT,
        feed_url TEXT,
        category TEXT,
        filter_id INTEGER,
        filter_name TEXT,
        filter_pattern TEXT,
        filtered_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_filtered_items_at ON filtered_items(filtered_at DESC);
    CREATE INDEX IF NOT EXISTS idx_filtered_items_filter ON filtered_items(filter_id);

    CREATE TABLE IF NOT EXISTS reference_sites (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        url TEXT NOT NULL,
        parser TEXT NOT NULL,
        enabled INTEGER DEFAULT 1,
        max_headlines INTEGER DEFAULT 20,
        last_checked TEXT,
        last_found INTEGER DEFAULT 0,
        last_gaps INTEGER DEFAULT 0,
        created_at TEXT,
        updated_at TEXT
    );

    CREATE TABLE IF NOT EXISTS reference_check_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        site_id INTEGER,
        site_name TEXT,
        headline TEXT,
        source_url TEXT,
        status TEXT,
        matched_cluster_id TEXT,
        detail TEXT,
        checked_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_ref_log_checked ON reference_check_log(checked_at DESC);
    CREATE INDEX IF NOT EXISTS idx_ref_log_site ON reference_check_log(site_id);

    CREATE TABLE IF NOT EXISTS feed_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        url TEXT NOT NULL UNIQUE,
        category TEXT NOT NULL,
        enabled INTEGER DEFAULT 1,
        created_at TEXT,
        updated_at TEXT
    );

    CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        sort_order INTEGER DEFAULT 0,
        enabled INTEGER DEFAULT 1,
        created_at TEXT,
        updated_at TEXT
    );

    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at TEXT
    );
    """)
    # Migration: add published_at to story_clusters if missing
    cols = [r[1] for r in conn.execute("PRAGMA table_info(story_clusters)").fetchall()]
    if "published_at" not in cols:
        conn.execute("ALTER TABLE story_clusters ADD COLUMN published_at TEXT")
        # Backfill from raw_items
        conn.execute("""
            UPDATE story_clusters SET published_at = (
                SELECT MIN(ri.published_at) FROM raw_items ri WHERE ri.cluster_id = story_clusters.id
            ) WHERE published_at IS NULL
        """)

    # Migration: add breaking, boost, note, expiry_override to curation_overrides
    co_cols = [r[1] for r in conn.execute("PRAGMA table_info(curation_overrides)").fetchall()]
    if "breaking" not in co_cols:
        conn.execute("ALTER TABLE curation_overrides ADD COLUMN breaking INTEGER DEFAULT 0")
    if "boost" not in co_cols:
        conn.execute("ALTER TABLE curation_overrides ADD COLUMN boost INTEGER DEFAULT 0")
    if "note" not in co_cols:
        conn.execute("ALTER TABLE curation_overrides ADD COLUMN note TEXT")
    if "expiry_override" not in co_cols:
        conn.execute("ALTER TABLE curation_overrides ADD COLUMN expiry_override TEXT")
    if "scoop_boosted_at" not in co_cols:
        conn.execute("ALTER TABLE curation_overrides ADD COLUMN scoop_boosted_at TEXT")

    # Seed content_filters if empty
    filter_count = conn.execute("SELECT COUNT(*) as c FROM content_filters").fetchone()["c"]
    if filter_count == 0:
        now = datetime.now(timezone.utc).isoformat()
        seed_filters = [
            ("Product reviews", r'\b(review|reviewed|hands[- ]on)\b.*\b(of|with|for)\b'),
            ("Best-of lists", r'\bbest\b.{0,30}\b(for|in|of|under|deals?|picks?|buy)\b'),
            ("Top N picks", r'\b(top|our)\s+\d+\s+(picks?|favorites?|choices?|recommendations?)\b'),
            ("Buying guides", r'\bbuying guide\b'),
            ("Versus comparisons", r'\bvs\.?\s'),
            ("Sales and deals", r'\b(save|savings?|deal|deals|discount|sale|coupon|promo|offer)\b.{0,30}\b(on|at|for|now|today|this)\b'),
            ("Percentage off", r'\b\d+%?\s*off\b'),
            ("Lowest/best price", r'\b(lowest|best)\s+price\b'),
            ("Under $X", r'\bunder\s+\$\d+\b'),
            ("Just $X", r'\bjust\s+\$\d+\b'),
            ("For $X", r'\bfor\s+\$\d+\b'),
            ("Starts at $X", r'\bstarts?\s+at\s+\$\d+\b'),
            ("Shopping events", r'\b(black friday|cyber monday|prime day|clearance)\b'),
            ("Affiliate content", r'\baffiliate\b'),
            ("Sponsored content", r'\bsponsored\b'),
            ("Worth it / should you buy", r'\b(worth|still worth)\b.{0,20}\b(it|buying|paying|the price)\b'),
            ("Should you buy", r'\bshould you (buy|get|upgrade)\b'),
            ("Listicles / roundups", r'^\d+\s+(best|top|favorite|essential|must[- ]have|ways?|things?|tips?|tricks?|reasons?)\b'),
            ("Personal blog style", r'\b(here are|my favorite|i love|i tested|we tested|we picked)\b'),
            ("Marketing content", r'\b(case study|whitepaper|white paper|webinar|free download)\b'),
            ("Customization content", r'\b(watch faces?|wallpapers?|ringtones?|themes?|skins?)\b.{0,20}\b(for|on|free)\b'),
            ("How-to / tutorials", r'^(how to|a guide to|the complete guide|step[- ]by[- ]step|building a)\b'),
        ]
        for name, pattern in seed_filters:
            conn.execute(
                "INSERT INTO content_filters (name, filter_type, pattern, enabled, created_at, updated_at) VALUES (?, 'not_news', ?, 1, ?, ?)",
                (name, pattern, now, now)
            )

    # Seed reference_sites if empty
    ref_count = conn.execute("SELECT COUNT(*) as c FROM reference_sites").fetchone()["c"]
    if ref_count == 0:
        now = datetime.now(timezone.utc).isoformat()
        seed_refs = [
            ("Techmeme", "https://techmeme.com", "techmeme", 20),
            ("Drudge Report", "https://drudgereport.com", "drudge", 30),
            ("AP News", "https://apnews.com", "apnews", 20),
            ("Google News", "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en", "googlenews", 20),
        ]
        for name, url, parser, max_hl in seed_refs:
            conn.execute(
                "INSERT INTO reference_sites (name, url, parser, enabled, max_headlines, created_at, updated_at) VALUES (?,?,?,1,?,?,?)",
                (name, url, parser, max_hl, now, now)
            )

    # Seed categories if empty
    cat_count = conn.execute("SELECT COUNT(*) as c FROM categories").fetchone()["c"]
    if cat_count == 0:
        now = datetime.now(timezone.utc).isoformat()
        seed_cats = [("tech", 0), ("markets", 1), ("politics", 2), ("world", 3), ("general", 4)]
        for name, sort_order in seed_cats:
            conn.execute(
                "INSERT INTO categories (name, sort_order, enabled, created_at, updated_at) VALUES (?,?,1,?,?)",
                (name, sort_order, now, now)
            )

    # Seed settings — default to latest algorithm version
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('algorithm_version', 'v2')")
    # Upgrade existing installs still on v1
    conn.execute("UPDATE settings SET value = 'v2' WHERE key = 'algorithm_version' AND value = 'v1'")

    # Seed feed_sources from feeds.yaml if empty
    src_count = conn.execute("SELECT COUNT(*) as c FROM feed_sources").fetchone()["c"]
    if src_count == 0:
        now = datetime.now(timezone.utc).isoformat()
        try:
            feeds_cfg = load_feeds()
            for category, feeds in feeds_cfg.get("feeds", {}).items():
                for feed in feeds:
                    conn.execute(
                        "INSERT OR IGNORE INTO feed_sources (name, url, category, enabled, created_at, updated_at) VALUES (?,?,?,1,?,?)",
                        (feed["name"], feed["url"], category, now, now)
                    )
        except Exception:
            pass

    conn.commit()
    conn.close()
