import re
import time
import uuid
import logging
from datetime import datetime, timezone, timedelta

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from wire.config import load_config
from wire.events import push as ev
from wire.scores import get_total_score
from wire.algorithm import get_active_version

log = logging.getLogger("wire.cluster")

# ── Sentence embedding model (lazy-loaded) ────────────────────────────────
_embed_model = None
_boot_complete = False  # set True once boot finishes — defers model load


def mark_boot_complete():
    """Signal that boot is done; embedding model may now be loaded."""
    global _boot_complete
    _boot_complete = True


def _get_embed_model():
    """Lazy-load a lightweight sentence-transformer for semantic similarity.
    Deferred until after boot to avoid slowing startup."""
    global _embed_model
    if _embed_model is None:
        if not _boot_complete:
            return None
        import os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ["TQDM_DISABLE"] = "1"  # suppress weight-loading progress bars
        # Suppress verbose logging from sentence-transformers
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Loaded sentence embedding model: all-MiniLM-L6-v2")
    return _embed_model

# ── Entity / topic extraction for aggressive clustering ──────────────────

# Key entities/topics that should force clustering together
# If two headlines share any of these, they're likely the same story
TOPIC_PATTERNS = [
    # People
    r'\btrump\b', r'\bbiden\b', r'\bmusk\b', r'\bzuckerberg\b', r'\bsam altman\b',
    r'\bzelensky\b', r'\bputin\b', r'\bxi jinping\b', r'\brfk\b', r'\bcook\b',
    r'\bnadella\b', r'\bpichai\b', r'\bjony ive\b', r'\bphil spencer\b',
    # Orgs
    r'\bopenai\b', r'\banthropic\b', r'\btesla\b', r'\bapple\b', r'\bgoogle\b',
    r'\bmicrosoft\b', r'\bmeta\b', r'\bnvidia\b', r'\bamazon\b', r'\bnetflix\b',
    r'\bxbox\b', r'\bpalantir\b', r'\bcoreweave\b', r'\bapplovin\b',
    # Topics
    r'\btariff', r'\bscotus\b', r'\bsupreme court\b', r'\bartemis\b',
    r'\bautopilot\b', r'\bchatgpt\b', r'\bclaude\b', r'\bgemini\b',
    r'\bgrokipedia\b', r'\bwikipedia\b', r'\bransomware\b',
    # Government agencies
    r'\btsa\b', r'\bdhs\b', r'\bfbi\b', r'\bdoj\b', r'\bepa\b',
    r'\bfda\b', r'\bfcc\b', r'\bsec\b', r'\bfaa\b',
    # Programs
    r'\bprecheck\b', r'\bglobal entry\b', r'\bmedicare\b', r'\bmedicaid\b',
    r'\bsocial security\b',
    # Geopolitical
    r'\bukraine\b', r'\brussia\b', r'\bgazan?\b', r'\bisrael\b',
    r'\btaiwan\b', r'\bnato\b', r'\bboeing\b',
    # Events
    r'\bshutdown\b', r'\bceasefire\b', r'\bbrexit\b', r'\bimpeach',
]

COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), p) for p in TOPIC_PATTERNS]


def _extract_topics(text: str) -> set:
    """Extract key topic tags from a headline."""
    topics = set()
    for pattern, raw in COMPILED_PATTERNS:
        if pattern.search(text):
            # Normalize: strip regex chars, lowercase
            tag = re.sub(r'[\\b]', '', raw).strip().lower()
            topics.add(tag)
    return topics


def _topic_overlap_score(topics_a: set, topics_b: set) -> float:
    """Return overlap score between two topic sets. 0-1."""
    if not topics_a or not topics_b:
        return 0.0
    intersection = topics_a & topics_b
    if not intersection:
        return 0.0
    # Jaccard-ish but weighted toward having ANY overlap
    # If they share at least one specific entity, that's a strong signal
    return len(intersection) / min(len(topics_a), len(topics_b))


_STOPWORDS = frozenset({
    # Common English
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "have", "been", "from", "its",
    "they", "were", "that", "this", "with", "will", "each", "make", "like",
    "over", "such", "than", "them", "then", "into", "some", "could", "would",
    "about", "after", "which", "their", "there", "other", "being", "where",
    "these", "those", "does", "what", "when", "who", "how", "more", "also",
    # News boilerplate
    "says", "said", "report", "reports", "reported", "new", "amid", "just",
    "may", "could", "would", "should", "will", "now", "get", "gets",
    "set", "sets", "back", "way", "take", "takes", "why", "still",
})


def _extract_keywords(headline: str) -> set:
    """Extract significant non-stopword terms from a headline."""
    words = re.findall(r'[a-z]{3,}', headline.lower())
    return {w for w in words if w not in _STOPWORDS}


def _keyword_overlap_score(kw_a: set, kw_b: set) -> int:
    """Return count of shared keywords between two sets."""
    return len(kw_a & kw_b)


# ── TF-IDF cache for poll cycles ──────────────────────────────────────────
# Reuse the fitted vectorizer + matrix across multiple assign_cluster calls
# within a single poll cycle instead of rebuilding from scratch each time.

_tfidf_cache = {
    "vectorizer": None,
    "matrix": None,        # sparse matrix of existing cluster headlines
    "cluster_ids": [],     # parallel list of cluster IDs
    "headlines": [],       # parallel list of headlines
    "topic_sets": [],      # precomputed topic sets
    "keyword_sets": [],    # precomputed keyword sets
    "embeddings": None,    # sentence embeddings (numpy array or None)
    "rows": [],            # full row data (for scoop_boosted_at etc.)
    "built_at": 0,         # monotonic timestamp
    "db_cutoff": "",       # the cutoff used to build this cache
}
_TFIDF_CACHE_TTL = 10  # seconds — refreshed when stale or when clusters change


def invalidate_tfidf_cache():
    """Force cache rebuild on next assign_cluster call."""
    _tfidf_cache["built_at"] = 0


def _build_tfidf_cache(conn, cutoff: str, now_iso: str, is_ttl_refresh: bool = False):
    """Rebuild the TF-IDF cache from current clusters.

    Embeddings are only computed on TTL refreshes (stable cache), not after
    invalidation, since invalidation happens on every new cluster creation
    and re-encoding all headlines each time is O(n²) during rapid ingestion.
    """
    rows = conn.execute(
        """SELECT sc.id, sc.rewritten_headline, sc.primary_source, sc.source_count,
                  co.scoop_boosted_at
           FROM story_clusters sc
           LEFT JOIN curation_overrides co ON co.cluster_id = sc.id
           WHERE sc.last_updated > ? AND sc.expires_at > ?""",
        (cutoff, now_iso)
    ).fetchall()

    if not rows:
        _tfidf_cache["vectorizer"] = None
        _tfidf_cache["matrix"] = None
        _tfidf_cache["cluster_ids"] = []
        _tfidf_cache["headlines"] = []
        _tfidf_cache["topic_sets"] = []
        _tfidf_cache["keyword_sets"] = []
        _tfidf_cache["embeddings"] = None
        _tfidf_cache["rows"] = []
        _tfidf_cache["built_at"] = time.monotonic()
        _tfidf_cache["db_cutoff"] = cutoff
        return

    headlines = [r["rewritten_headline"] or "" for r in rows]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform(headlines)

    _tfidf_cache["vectorizer"] = vectorizer
    _tfidf_cache["matrix"] = matrix
    _tfidf_cache["cluster_ids"] = [r["id"] for r in rows]
    _tfidf_cache["headlines"] = headlines
    _tfidf_cache["topic_sets"] = [_extract_topics(h) for h in headlines]
    _tfidf_cache["keyword_sets"] = [_extract_keywords(h) for h in headlines]
    _tfidf_cache["rows"] = list(rows)
    _tfidf_cache["built_at"] = time.monotonic()
    _tfidf_cache["db_cutoff"] = cutoff

    # Only compute embeddings on TTL refresh (cache was stable for 10s).
    # During rapid ingestion the cache is invalidated on every new cluster,
    # so skip the expensive encode to keep ingestion fast.  Pass 4 is
    # effectively disabled until the cluster set stabilises.
    if is_ttl_refresh:
        model = _get_embed_model()
        if model is not None:
            try:
                _tfidf_cache["embeddings"] = model.encode(
                    headlines, normalize_embeddings=True, show_progress_bar=False
                )
            except Exception as e:
                log.warning(f"Embedding cache build error: {e}")
                _tfidf_cache["embeddings"] = None
        else:
            _tfidf_cache["embeddings"] = None
    else:
        _tfidf_cache["embeddings"] = None


def _get_tfidf_cache(conn, cutoff: str, now_iso: str):
    """Return a fresh or cached TF-IDF index for clustering."""
    age = time.monotonic() - _tfidf_cache["built_at"]
    if age < _TFIDF_CACHE_TTL and _tfidf_cache["db_cutoff"] == cutoff:
        return _tfidf_cache
    # TTL refresh = cache expired naturally; invalidation = built_at was zeroed
    is_ttl = _tfidf_cache["built_at"] > 0
    _build_tfidf_cache(conn, cutoff, now_iso, is_ttl_refresh=is_ttl)
    return _tfidf_cache


def assign_cluster(conn, headline: str, url: str, source_name: str, category: str, published_at: str = None) -> str:
    """Find or create a cluster for this headline. Returns cluster_id."""
    cfg = load_config()
    algo = get_active_version()
    tfidf_threshold = algo.get("tfidf_threshold", cfg["clustering"]["similarity_threshold"])
    lookback = algo.get("lookback_hours", cfg["clustering"]["lookback_hours"])
    topic_tfidf_threshold = algo.get("topic_tfidf_threshold", 0.15)
    topic_only_threshold = algo.get("topic_only_threshold", 1.0)
    topic_overlap_min = algo.get("topic_overlap_min", 0.5)
    keyword_tfidf_threshold = algo.get("keyword_tfidf_threshold", 0.10)
    min_keyword_overlap = algo.get("min_keyword_overlap", 3)
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(hours=lookback)).isoformat()

    # Get cached TF-IDF index (rebuilt every _TFIDF_CACHE_TTL seconds)
    cache = _get_tfidf_cache(conn, cutoff, now.isoformat())
    rows = cache["rows"]

    if rows and cache["vectorizer"] is not None:
        cluster_headlines = cache["headlines"]
        new_topics = _extract_topics(headline)

        # ── Pass 1: TF-IDF similarity ────────────────────────────────────
        best_tfidf_idx = -1
        best_tfidf_sim = 0.0
        try:
            new_vec = cache["vectorizer"].transform([headline])
            sims = cosine_similarity(new_vec, cache["matrix"])[0]
            best_tfidf_idx = int(sims.argmax())
            best_tfidf_sim = float(sims[best_tfidf_idx])
        except Exception as e:
            log.warning(f"TF-IDF error: {e}")

        # ── Pass 2: Topic/entity overlap ─────────────────────────────────
        best_topic_idx = -1
        best_topic_score = 0.0
        best_topic_shared = 0  # number of shared topics
        if new_topics:
            for i, cluster_topics in enumerate(cache["topic_sets"]):
                overlap = _topic_overlap_score(new_topics, cluster_topics)
                shared = len(new_topics & cluster_topics)
                if overlap > best_topic_score:
                    best_topic_score = overlap
                    best_topic_idx = i
                    best_topic_shared = shared

        def _scoop_date_ok(candidate_idx):
            """Reject matches to scoop-boosted clusters if the incoming article predates the scoop."""
            scoop_at = rows[candidate_idx]["scoop_boosted_at"]
            if not scoop_at or not published_at:
                return True
            return published_at >= scoop_at

        # ── Absolute floor: reject if headline has near-zero overlap ─────
        # Even weaker paths (topic, keyword) should not fire when TF-IDF
        # is essentially noise.  This prevents stray articles from joining
        # a cluster via inflated keyword counts or a single shared topic.
        TFIDF_FLOOR = 0.08

        # ── Decision: combine both signals ───────────────────────────────
        # Strong TF-IDF match alone — headlines must be very similar
        if best_tfidf_sim >= tfidf_threshold and _scoop_date_ok(best_tfidf_idx):
            cluster_id = rows[best_tfidf_idx]["id"]
            ev("cluster_hit", headline=headline, matched=rows[best_tfidf_idx]["rewritten_headline"],
               similarity=round(best_tfidf_sim, 3), method="tfidf")
            _add_to_cluster(conn, cluster_id, source_name, url, headline, category, published_at=published_at)
            return cluster_id

        # Topic overlap with weak TF-IDF — requires 2+ shared topics to avoid
        # single broad topic (e.g. "tariff") linking unrelated stories
        if best_topic_score >= topic_overlap_min and best_topic_shared >= 2 and best_tfidf_sim >= max(topic_tfidf_threshold, TFIDF_FLOOR) and best_topic_idx >= 0 and _scoop_date_ok(best_topic_idx):
            cluster_id = rows[best_topic_idx]["id"]
            ev("cluster_hit", headline=headline, matched=rows[best_topic_idx]["rewritten_headline"],
               similarity=round(best_tfidf_sim, 3), topic_overlap=round(best_topic_score, 2), method="topic+tfidf")
            _add_to_cluster(conn, cluster_id, source_name, url, headline, category, published_at=published_at)
            return cluster_id

        # Strong topic overlap alone — requires 2+ shared entities to prevent
        # broad single-topic matches (e.g. "tariff" alone linking unrelated stories)
        if best_topic_score >= topic_only_threshold and best_topic_shared >= 2 and best_tfidf_sim >= TFIDF_FLOOR and best_topic_idx >= 0 and _scoop_date_ok(best_topic_idx):
            cluster_id = rows[best_topic_idx]["id"]
            ev("cluster_hit", headline=headline, matched=rows[best_topic_idx]["rewritten_headline"],
               topic_overlap=round(best_topic_score, 2), method="topic_only")
            _add_to_cluster(conn, cluster_id, source_name, url, headline, category, published_at=published_at)
            return cluster_id

        # ── Pass 3: Keyword overlap ──────────────────────────────────────
        # Catches developing story angles that share rare terms
        new_keywords = _extract_keywords(headline)
        if new_keywords and best_tfidf_sim >= max(keyword_tfidf_threshold, TFIDF_FLOOR):
            best_kw_idx = -1
            best_kw_count = 0
            for i, cluster_kw in enumerate(cache["keyword_sets"]):
                overlap = _keyword_overlap_score(new_keywords, cluster_kw)
                if overlap > best_kw_count:
                    best_kw_count = overlap
                    best_kw_idx = i
            if best_kw_count >= min_keyword_overlap and best_kw_idx >= 0 and _scoop_date_ok(best_kw_idx):
                cluster_id = rows[best_kw_idx]["id"]
                ev("cluster_hit", headline=headline, matched=rows[best_kw_idx]["rewritten_headline"],
                   similarity=round(best_tfidf_sim, 3), keyword_overlap=best_kw_count, method="keyword+tfidf")
                _add_to_cluster(conn, cluster_id, source_name, url, headline, category, published_at=published_at)
                return cluster_id

        # ── Pass 4: Sentence embedding similarity ───────────────────────
        # Catches semantic paraphrases that TF-IDF/topic/keyword miss.
        # Only runs when embeddings are pre-computed (cache survived a full
        # TTL cycle without invalidation).  During rapid ingestion this pass
        # is skipped; merge_existing_clusters catches stragglers afterward.
        embed_threshold = algo.get("embed_threshold", 0.72)
        cached_embeddings = cache.get("embeddings")
        if cached_embeddings is not None:
            try:
                model = _get_embed_model()
                new_embed = model.encode([headline], normalize_embeddings=True, show_progress_bar=False)
                embed_sims = (new_embed @ cached_embeddings.T)[0]
                best_embed_idx = int(embed_sims.argmax())
                best_embed_sim = float(embed_sims[best_embed_idx])

                if best_embed_sim >= embed_threshold and best_tfidf_sim >= TFIDF_FLOOR and _scoop_date_ok(best_embed_idx):
                    cluster_id = rows[best_embed_idx]["id"]
                    ev("cluster_hit", headline=headline, matched=rows[best_embed_idx]["rewritten_headline"],
                       similarity=round(best_embed_sim, 3), tfidf=round(best_tfidf_sim, 3), method="embedding")
                    _add_to_cluster(conn, cluster_id, source_name, url, headline, category, published_at=published_at)
                    return cluster_id
            except Exception as e:
                log.warning(f"Embedding pass error: {e}")

    # New cluster — invalidate cache so next call picks it up
    invalidate_tfidf_cache()
    ev("cluster_new", headline=headline, source=source_name, category=category)
    cluster_id = str(uuid.uuid4())
    expires = (now + timedelta(days=7)).isoformat()
    pub_time = published_at or now.isoformat()
    conn.execute(
        "INSERT INTO story_clusters (id, rewritten_headline, primary_url, primary_source, category, source_count, first_seen, last_updated, expires_at, published_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (cluster_id, headline, url, source_name, category, 1, now.isoformat(), now.isoformat(), expires, pub_time)
    )
    conn.execute(
        "INSERT OR IGNORE INTO cluster_sources (cluster_id, source_name, source_url, added_at) VALUES (?,?,?,?)",
        (cluster_id, source_name, url, now.isoformat())
    )

    # Auto-boost scoops/exclusives from top-tier sources
    _maybe_scoop_boost(conn, cluster_id, headline, source_name, algo, now, published_at=published_at)

    return cluster_id


def _maybe_scoop_boost(conn, cluster_id: str, headline: str, source_name: str, algo: dict, now, published_at: str = None):
    """Auto-boost clusters from top-tier sources with scoop/exclusive indicators."""
    if not algo.get("scoop_enabled"):
        return
    scoop_sources = algo.get("scoop_sources", [])
    scoop_patterns = algo.get("scoop_patterns", [])
    if source_name not in scoop_sources or not scoop_patterns:
        return
    # Check if headline matches any scoop pattern
    matched = False
    for pattern in scoop_patterns:
        if re.search(pattern, headline):
            matched = True
            break
    if not matched:
        return

    # Use the article's published_at as scoop timestamp (not processing time)
    # so follow-on coverage with later publish dates isn't incorrectly blocked
    scoop_ts = published_at or now.isoformat()
    now_iso = now.isoformat()
    conn.execute("""
        INSERT INTO curation_overrides (cluster_id, boost, scoop_boosted_at, updated_at)
        VALUES (?, 1, ?, ?)
        ON CONFLICT(cluster_id) DO UPDATE SET boost=1, scoop_boosted_at=?, updated_at=?
    """, (cluster_id, scoop_ts, now_iso, scoop_ts, now_iso))
    ev("scoop_boost", headline=headline, source=source_name, cluster_id=cluster_id[:8])
    log.info(f"Scoop boost: [{source_name}] {headline[:80]}")


def _headline_relevance(headline: str, cluster_headline: str) -> float:
    """Quick TF-IDF check: how well does this article's headline match the cluster."""
    try:
        v = TfidfVectorizer(stop_words="english")
        m = v.fit_transform([headline, cluster_headline])
        return float(cosine_similarity(m[0:1], m[1:2])[0, 0])
    except Exception:
        return 0.0


def _add_to_cluster(conn, cluster_id: str, source_name: str, url: str, headline: str, category: str, published_at: str = None):
    now = datetime.now(timezone.utc).isoformat()

    # Add source
    conn.execute(
        "INSERT OR IGNORE INTO cluster_sources (cluster_id, source_name, source_url, added_at) VALUES (?,?,?,?)",
        (cluster_id, source_name, url, now)
    )

    # Update count (distinct sources, not URLs)
    count = conn.execute("SELECT COUNT(DISTINCT source_name) as c FROM cluster_sources WHERE cluster_id=?", (cluster_id,)).fetchone()["c"]

    # Check if this source has higher priority AND its headline is relevant
    # to the cluster topic. Without the relevance check, a high-tier source's
    # tangentially related article (e.g. a live blog mentioning tariffs in passing)
    # can hijack the primary link away from an actually on-topic article.
    current = conn.execute("SELECT primary_source, rewritten_headline FROM story_clusters WHERE id=?", (cluster_id,)).fetchone()
    updates = {"source_count": count, "last_updated": now}

    if get_total_score(source_name) > get_total_score(current["primary_source"]):
        relevance = _headline_relevance(headline, current["rewritten_headline"] or "")
        if relevance >= 0.15:
            updates["primary_url"] = url
            updates["primary_source"] = source_name

    # When primary source changes, use the new primary's published_at
    # Otherwise only update if no published_at is set yet
    if "primary_source" in updates and published_at:
        updates["published_at"] = published_at
    elif published_at:
        existing_pub = conn.execute("SELECT published_at FROM story_clusters WHERE id=?", (cluster_id,)).fetchone()["published_at"]
        if existing_pub is None:
            updates["published_at"] = published_at

    set_clause = ", ".join(f"{k}=?" for k in updates)
    conn.execute(f"UPDATE story_clusters SET {set_clause} WHERE id=?", (*updates.values(), cluster_id))


def _do_merge(conn, rows, i, j, now, merged):
    """Execute the actual merge of cluster j into cluster i.

    Moves sources and raw items from the loser (j) to the winner (i).
    Raw items whose original headline has near-zero relevance to the
    winner cluster are orphaned into their own new cluster instead of
    being blindly dragged along — this prevents merge cascades from
    polluting clusters with completely unrelated articles.
    """
    winner_id = rows[i]["id"]
    loser_id = rows[j]["id"]
    winner_headline = rows[i]["rewritten_headline"] or ""

    # ── Move sources ──────────────────────────────────────────────────
    conn.execute(
        "UPDATE OR IGNORE cluster_sources SET cluster_id=? WHERE cluster_id=?",
        (winner_id, loser_id)
    )

    # ── Move raw items, but check relevance first ─────────────────────
    MERGE_ITEM_FLOOR = 0.04  # very low — just catches total garbage
    loser_items = conn.execute(
        "SELECT id, original_headline FROM raw_items WHERE cluster_id=?",
        (loser_id,)
    ).fetchall()

    for item in loser_items:
        rel = _headline_relevance(item["original_headline"] or "", winner_headline)
        if rel >= MERGE_ITEM_FLOOR:
            conn.execute("UPDATE raw_items SET cluster_id=? WHERE id=?", (winner_id, item["id"]))
        else:
            log.info(f"Merge skip: '{(item['original_headline'] or '')[:60]}' irrelevant to '{winner_headline[:60]}' (rel={rel:.3f})")
            # Orphan it — set cluster_id to NULL so it doesn't pollute
            conn.execute("UPDATE raw_items SET cluster_id=NULL WHERE id=?", (item["id"],))

    conn.execute("DELETE FROM cluster_sources WHERE cluster_id=?", (loser_id,))
    new_count = conn.execute(
        "SELECT COUNT(DISTINCT source_name) as c FROM cluster_sources WHERE cluster_id=?",
        (winner_id,)
    ).fetchone()["c"]
    conn.execute(
        "UPDATE story_clusters SET source_count=?, last_updated=? WHERE id=?",
        (new_count, now.isoformat(), winner_id)
    )
    conn.execute("DELETE FROM story_clusters WHERE id=?", (loser_id,))
    merged.add(loser_id)


def merge_existing_clusters(conn):
    """
    Hybrid merge: TF-IDF pass first (fast, high-confidence only),
    then embedding pass (semantic, catches paraphrases).
    Called at end of boot and hourly to catch duplicates that slip through.
    """
    algo = get_active_version()
    # TF-IDF pass uses a strong threshold — only near-identical wording
    # Embeddings handle the semantic/paraphrase cases in pass 2
    merge_tfidf = max(algo.get("tfidf_threshold", 0.30), 0.40)

    now = datetime.now(timezone.utc)
    rows = conn.execute(
        "SELECT id, rewritten_headline, source_count FROM story_clusters WHERE expires_at > ? ORDER BY source_count DESC",
        (now.isoformat(),)
    ).fetchall()

    if len(rows) < 2:
        return 0

    headlines = [r["rewritten_headline"] or "" for r in rows]

    # ── Pass 1: TF-IDF only (fast, high-confidence word overlap) ──────────
    try:
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        matrix = tfidf.fit_transform(headlines)
        sim_matrix = (matrix * matrix.T).toarray()
    except Exception as e:
        log.warning(f"Merge TF-IDF error: {e}")
        sim_matrix = np.zeros((len(rows), len(rows)))

    merged = set()
    merge_count = 0

    for i in range(len(rows)):
        if rows[i]["id"] in merged:
            continue
        for j in range(i + 1, len(rows)):
            if rows[j]["id"] in merged:
                continue
            if sim_matrix[i, j] >= merge_tfidf:
                log.info(f"TF-IDF merge: '{headlines[i][:60]}' ← '{headlines[j][:60]}' (sim={sim_matrix[i, j]:.3f})")
                _do_merge(conn, rows, i, j, now, merged)
                merge_count += 1

    # Commit pass 1 before re-fetching for pass 2
    if merge_count > 0:
        conn.commit()

    # ── Pass 2: Sentence embeddings (semantic, catches paraphrases) ───────
    # Re-fetch rows since pass 1 may have deleted some
    remaining_ids = {rows[i]["id"] for i in range(len(rows)) if rows[i]["id"] not in merged}
    if len(remaining_ids) < 2:
        if merge_count > 0:
            invalidate_tfidf_cache()
        conn.commit()
        log.info(f"Merged {merge_count} duplicate clusters (TF-IDF only)")
        return merge_count

    rows2 = conn.execute(
        "SELECT id, rewritten_headline, source_count FROM story_clusters WHERE expires_at > ? ORDER BY source_count DESC",
        (now.isoformat(),)
    ).fetchall()

    if len(rows2) < 2:
        if merge_count > 0:
            invalidate_tfidf_cache()
        conn.commit()
        log.info(f"Merged {merge_count} duplicate clusters (TF-IDF only)")
        return merge_count

    headlines2 = [r["rewritten_headline"] or "" for r in rows2]

    EMBED_THRESHOLD = 0.75  # semantic similarity threshold for merge

    try:
        model = _get_embed_model()
        if model is None:
            raise RuntimeError("Embedding model not available yet")
        embeddings = model.encode(headlines2, normalize_embeddings=True, show_progress_bar=False)
        embed_sim = embeddings @ embeddings.T
    except Exception as e:
        log.warning(f"Embedding merge error: {e}")
        if merge_count > 0:
            invalidate_tfidf_cache()
        conn.commit()
        log.info(f"Merged {merge_count} duplicate clusters (TF-IDF only, embedding failed)")
        return merge_count

    merged2 = set()
    for i in range(len(rows2)):
        if rows2[i]["id"] in merged2:
            continue
        for j in range(i + 1, len(rows2)):
            if rows2[j]["id"] in merged2:
                continue
            if embed_sim[i, j] >= EMBED_THRESHOLD:
                log.info(f"Embedding merge: '{headlines2[i][:60]}' ← '{headlines2[j][:60]}' (sim={embed_sim[i, j]:.3f})")
                _do_merge(conn, rows2, i, j, now, merged2)
                merge_count += 1

    if merge_count > 0:
        invalidate_tfidf_cache()
    conn.commit()
    log.info(f"Merged {merge_count} duplicate clusters (TF-IDF + embeddings)")
    return merge_count
