import sqlite3
from pathlib import Path
from wire.config import load_config

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
    CREATE INDEX IF NOT EXISTS idx_clusters_category ON story_clusters(category);
    CREATE INDEX IF NOT EXISTS idx_clusters_expires ON story_clusters(expires_at);
    CREATE INDEX IF NOT EXISTS idx_clusters_source_count ON story_clusters(source_count DESC);
    CREATE INDEX IF NOT EXISTS idx_clusters_last_updated ON story_clusters(last_updated DESC);
    CREATE INDEX IF NOT EXISTS idx_raw_source_url ON raw_items(source_url);
    CREATE INDEX IF NOT EXISTS idx_raw_headline_source ON raw_items(original_headline, source_name);
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
    conn.commit()
    conn.close()
