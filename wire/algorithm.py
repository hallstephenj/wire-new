import time
import logging

from wire.db import get_conn

log = logging.getLogger("wire.algorithm")

# ── Version definitions ──────────────────────────────────────────────────
# Each version captures every tunable parameter of the ranking algorithm.

VERSIONS = {
    "v1": {
        "name": "v1",
        "label": "V1 — Breadth × Quality / Decay",
        "description": "Super-linear breadth (^1.5), sqrt quality boost, 4h half-life, 1.3 decay exponent. Requires 3+ reputable sources for hot ranking.",
        # ── Hot score parameters ──
        "breadth_exponent": 1.5,
        "quality_exponent": 0.5,
        "decay_halflife_hours": 4.0,
        "decay_exponent": 1.3,
        # ── Source quality multipliers (used in hot score SQL CASE) ──
        "source_quality": {
            3.0: ["Reuters", "AP"],
            2.5: ["New York Times", "Wall Street Journal", "Washington Post",
                   "BBC", "Bloomberg", "Financial Times", "The Guardian", "CNN", "NBC News", "NPR", "Al Jazeera"],
            2.0: ["Ars Technica", "ProPublica", "MIT Technology Review", "Nature",
                   "Politico", "Wired", "404 Media", "The Atlantic", "The Record"],
            1.5: ["TechCrunch", "The Verge", "Axios", "CNBC", "Fortune", "Forbes",
                   "Fox Business", "Business Insider", "The Hill", "Semafor", "South China Morning Post",
                   "Nikkei Asia", "MacRumors", "CoinDesk", "The Register"],
            1.0: ["9to5Google", "Android Authority", "Fox News", "USA Today",
                   "MarketWatch", "Yahoo Finance", "Seeking Alpha", "Motley Fool", "Sky News", "Barron's"],
            0.6: ["MSN", "AOL", "Google News"],
        },
        "source_quality_default": 0.4,
        # ── Reputable sources filter ──
        "reputable_sources": [
            "Reuters", "AP",
            "New York Times", "Wall Street Journal", "Washington Post",
            "BBC", "Bloomberg", "Financial Times", "The Guardian", "CNN", "NBC News", "NPR", "Al Jazeera",
            "Ars Technica", "ProPublica", "MIT Technology Review", "Nature",
            "Politico", "Wired", "404 Media", "The Atlantic", "The Record",
            "TechCrunch", "The Verge", "Axios", "CNBC", "Fortune", "Forbes",
            "Fox Business", "Business Insider", "The Hill", "Semafor", "South China Morning Post",
            "Nikkei Asia", "MacRumors", "CoinDesk", "The Register",
            "9to5Google", "Android Authority", "Fox News", "USA Today",
            "MarketWatch", "Yahoo Finance", "Seeking Alpha", "Motley Fool", "Sky News", "Barron's",
            "ABC News", "CBS News", "The Information", "Engadget", "TechRadar", "ZDNet",
            "PCMag", "Tom's Hardware", "Phy.org", "Space.com", "France24",
            "Euronews", "The Economist", "Foreign Policy", "Defense One",
        ],
        "min_reputable_sources": 3,
        # ── Display behavior ──
        "world_deprio": True,
        "general_exclusion": True,
        # ── Clustering thresholds ──
        "tfidf_threshold": 0.30,
        "topic_tfidf_threshold": 0.15,
        "topic_only_threshold": 1.0,
        "topic_overlap_min": 0.5,
        "keyword_tfidf_threshold": 0.10,
        "min_keyword_overlap": 3,
        "lookback_hours": 72,
        # ── Scoop boost ──
        "scoop_enabled": False,
        "scoop_sources": [],
        "scoop_patterns": [],
        "scoop_boost_hours": 4,
        "scoop_min_sources_after": 2,
    },
    "v2": {
        "name": "v2",
        "label": "V2 — Tighter Clusters, Scoop Boost",
        "description": "Tighter clustering thresholds, no world demotion, auto-boosts scoops/exclusives from top outlets for 4h (expires if no follow-on coverage).",
        # ── Hot score parameters (same as v1) ──
        "breadth_exponent": 1.5,
        "quality_exponent": 0.5,
        "decay_halflife_hours": 4.0,
        "decay_exponent": 1.3,
        # ── Source quality multipliers (same as v1) ──
        "source_quality": {
            3.0: ["Reuters", "AP"],
            2.5: ["New York Times", "Wall Street Journal", "Washington Post",
                   "BBC", "Bloomberg", "Financial Times", "The Guardian", "CNN", "NBC News", "NPR", "Al Jazeera"],
            2.0: ["Ars Technica", "ProPublica", "MIT Technology Review", "Nature",
                   "Politico", "Wired", "404 Media", "The Atlantic", "The Record"],
            1.5: ["TechCrunch", "The Verge", "Axios", "CNBC", "Fortune", "Forbes",
                   "Fox Business", "Business Insider", "The Hill", "Semafor", "South China Morning Post",
                   "Nikkei Asia", "MacRumors", "CoinDesk", "The Register"],
            1.0: ["9to5Google", "Android Authority", "Fox News", "USA Today",
                   "MarketWatch", "Yahoo Finance", "Seeking Alpha", "Motley Fool", "Sky News", "Barron's"],
            0.6: ["MSN", "AOL", "Google News"],
        },
        "source_quality_default": 0.4,
        # ── Reputable sources filter (same as v1) ──
        "reputable_sources": [
            "Reuters", "AP",
            "New York Times", "Wall Street Journal", "Washington Post",
            "BBC", "Bloomberg", "Financial Times", "The Guardian", "CNN", "NBC News", "NPR", "Al Jazeera",
            "Ars Technica", "ProPublica", "MIT Technology Review", "Nature",
            "Politico", "Wired", "404 Media", "The Atlantic", "The Record",
            "TechCrunch", "The Verge", "Axios", "CNBC", "Fortune", "Forbes",
            "Fox Business", "Business Insider", "The Hill", "Semafor", "South China Morning Post",
            "Nikkei Asia", "MacRumors", "CoinDesk", "The Register",
            "9to5Google", "Android Authority", "Fox News", "USA Today",
            "MarketWatch", "Yahoo Finance", "Seeking Alpha", "Motley Fool", "Sky News", "Barron's",
            "ABC News", "CBS News", "The Information", "Engadget", "TechRadar", "ZDNet",
            "PCMag", "Tom's Hardware", "Phy.org", "Space.com", "France24",
            "Euronews", "The Economist", "Foreign Policy", "Defense One",
        ],
        "min_reputable_sources": 3,
        # ── Display behavior ──
        "world_deprio": False,
        "general_exclusion": True,
        # ── Clustering thresholds (tighter than v1) ──
        "tfidf_threshold": 0.35,
        "topic_tfidf_threshold": 0.20,
        "topic_only_threshold": 1.0,
        "topic_overlap_min": 0.5,
        "keyword_tfidf_threshold": 0.15,
        "min_keyword_overlap": 4,
        "lookback_hours": 72,
        # ── Scoop boost ──
        "scoop_enabled": True,
        "scoop_sources": [
            "Reuters", "AP",
            "New York Times", "Wall Street Journal", "Washington Post",
            "BBC", "Bloomberg", "Financial Times", "The Guardian", "CNN", "NBC News", "NPR",
            "Axios", "Politico", "The Information", "Semafor",
            "ProPublica", "The Atlantic", "404 Media",
        ],
        "scoop_patterns": [r"(?i)\bscoop\b", r"(?i)\bexclusive\b", r"(?i)^sources?\b"],
        "scoop_boost_hours": 4,
        "scoop_min_sources_after": 2,
    },
}


# The latest version is always the last key in VERSIONS
LATEST_VERSION = list(VERSIONS.keys())[-1]


# ── SQL builder functions ────────────────────────────────────────────────

def _sql_quote(name: str) -> str:
    """Escape single quotes in a source name for SQL."""
    return name.replace("'", "''")


def build_source_quality_case(version: dict) -> str:
    """Build a SQL CASE expression for source quality multipliers."""
    lines = ["(CASE"]
    for multiplier, sources in version["source_quality"].items():
        quoted = ",".join(f"'{_sql_quote(s)}'" for s in sources)
        lines.append(f"    WHEN sc.primary_source IN ({quoted}) THEN {multiplier}")
    lines.append(f"    ELSE {version['source_quality_default']}")
    lines.append("END)")
    return "\n".join(lines)


def build_reputable_sources_sql(version: dict) -> str:
    """Build a SQL tuple string of reputable source names."""
    quoted = ",".join(f"'{_sql_quote(s)}'" for s in version["reputable_sources"])
    return f"({quoted})"


def build_hot_score_sql(version: dict) -> str:
    """Build the full hot score SQL expression."""
    quality_case = build_source_quality_case(version)
    return f"""(
        POWER(sc.source_count, {version['breadth_exponent']}) * POWER({quality_case}, {version['quality_exponent']})
        / POWER(1.0 + MAX(0, (julianday('now') - julianday(sc.published_at)) * 24) / {version['decay_halflife_hours']}, {version['decay_exponent']})
    )"""


# ── Active version accessor (with 30s TTL cache) ────────────────────────

_cache = {"version_name": None, "loaded_at": 0}
_CACHE_TTL = 30


def get_active_version() -> dict:
    """Read the active algorithm version from the DB settings table. 30s TTL cache. Falls back to latest."""
    now = time.time()
    if _cache["version_name"] and (now - _cache["loaded_at"]) < _CACHE_TTL:
        return VERSIONS[_cache["version_name"]]

    try:
        conn = get_conn()
        row = conn.execute("SELECT value FROM settings WHERE key = 'algorithm_version'").fetchone()
        conn.close()
        name = row["value"] if row else LATEST_VERSION
    except Exception:
        name = LATEST_VERSION

    if name not in VERSIONS:
        name = LATEST_VERSION

    _cache["version_name"] = name
    _cache["loaded_at"] = now
    return VERSIONS[name]


def set_active_version(name: str):
    """Update the active algorithm version in the DB and invalidate cache."""
    if name not in VERSIONS:
        raise ValueError(f"Unknown algorithm version: {name}")

    from datetime import datetime, timezone
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES ('algorithm_version', ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?",
        (name, now, name, now)
    )
    conn.commit()
    conn.close()

    _cache["version_name"] = name
    _cache["loaded_at"] = time.time()


def list_versions() -> list:
    """Return a list of version summaries for the API."""
    return [{"name": v["name"], "label": v["label"], "description": v["description"]}
            for v in VERSIONS.values()]
