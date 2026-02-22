import re
import uuid
import logging
from datetime import datetime, timezone, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from wire.config import load_config
from wire.events import push as ev
from wire.scores import get_total_score

log = logging.getLogger("wire.cluster")

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


def assign_cluster(conn, headline: str, url: str, source_name: str, category: str, published_at: str = None) -> str:
    """Find or create a cluster for this headline. Returns cluster_id."""
    cfg = load_config()
    tfidf_threshold = cfg["clustering"]["similarity_threshold"]
    lookback = cfg["clustering"]["lookback_hours"]
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(hours=lookback)).isoformat()

    # Get recent clusters
    rows = conn.execute(
        "SELECT id, rewritten_headline, primary_source, source_count FROM story_clusters WHERE last_updated > ? AND expires_at > ?",
        (cutoff, now.isoformat())
    ).fetchall()

    if rows:
        cluster_headlines = [r["rewritten_headline"] or "" for r in rows]
        new_topics = _extract_topics(headline)

        # ── Pass 1: TF-IDF similarity ────────────────────────────────────
        best_tfidf_idx = -1
        best_tfidf_sim = 0.0
        try:
            all_texts = cluster_headlines + [headline]
            tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
            matrix = tfidf.fit_transform(all_texts)
            sims = cosine_similarity(matrix[-1:], matrix[:-1])[0]
            best_tfidf_idx = int(sims.argmax())
            best_tfidf_sim = float(sims[best_tfidf_idx])
        except Exception as e:
            log.warning(f"TF-IDF error: {e}")

        # ── Pass 2: Topic/entity overlap ─────────────────────────────────
        best_topic_idx = -1
        best_topic_score = 0.0
        if new_topics:
            for i, ch in enumerate(cluster_headlines):
                cluster_topics = _extract_topics(ch)
                overlap = _topic_overlap_score(new_topics, cluster_topics)
                if overlap > best_topic_score:
                    best_topic_score = overlap
                    best_topic_idx = i

        # ── Decision: combine both signals ───────────────────────────────
        # Strong TF-IDF match alone — headlines must be very similar
        if best_tfidf_sim >= tfidf_threshold:
            cluster_id = rows[best_tfidf_idx]["id"]
            ev("cluster_hit", headline=headline, matched=rows[best_tfidf_idx]["rewritten_headline"],
               similarity=round(best_tfidf_sim, 3), method="tfidf")
            _add_to_cluster(conn, cluster_id, source_name, url, headline, category, published_at=published_at)
            return cluster_id

        # Topic overlap with even weak TF-IDF is enough
        # e.g. both mention "trump" + "tariff" with tfidf > 0.15
        if best_topic_score >= 0.5 and best_tfidf_sim >= 0.15 and best_topic_idx >= 0:
            cluster_id = rows[best_topic_idx]["id"]
            ev("cluster_hit", headline=headline, matched=rows[best_topic_idx]["rewritten_headline"],
               similarity=round(best_tfidf_sim, 3), topic_overlap=round(best_topic_score, 2), method="topic+tfidf")
            _add_to_cluster(conn, cluster_id, source_name, url, headline, category, published_at=published_at)
            return cluster_id

        # Strong topic overlap alone (2+ shared entities, very high overlap)
        if best_topic_score >= 1.0 and best_topic_idx >= 0:
            cluster_id = rows[best_topic_idx]["id"]
            ev("cluster_hit", headline=headline, matched=rows[best_topic_idx]["rewritten_headline"],
               topic_overlap=round(best_topic_score, 2), method="topic_only")
            _add_to_cluster(conn, cluster_id, source_name, url, headline, category, published_at=published_at)
            return cluster_id

        # ── Pass 3: Keyword overlap ──────────────────────────────────────
        # Catches developing story angles that share 3+ rare terms
        new_keywords = _extract_keywords(headline)
        if new_keywords and best_tfidf_sim >= 0.10:
            best_kw_idx = -1
            best_kw_count = 0
            for i, ch in enumerate(cluster_headlines):
                overlap = _keyword_overlap_score(new_keywords, _extract_keywords(ch))
                if overlap > best_kw_count:
                    best_kw_count = overlap
                    best_kw_idx = i
            if best_kw_count >= 3 and best_kw_idx >= 0:
                cluster_id = rows[best_kw_idx]["id"]
                ev("cluster_hit", headline=headline, matched=rows[best_kw_idx]["rewritten_headline"],
                   similarity=round(best_tfidf_sim, 3), keyword_overlap=best_kw_count, method="keyword+tfidf")
                _add_to_cluster(conn, cluster_id, source_name, url, headline, category, published_at=published_at)
                return cluster_id

    # New cluster
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
    return cluster_id


def _add_to_cluster(conn, cluster_id: str, source_name: str, url: str, headline: str, category: str, published_at: str = None):
    now = datetime.now(timezone.utc).isoformat()

    # Add source
    conn.execute(
        "INSERT OR IGNORE INTO cluster_sources (cluster_id, source_name, source_url, added_at) VALUES (?,?,?,?)",
        (cluster_id, source_name, url, now)
    )

    # Update count (distinct sources, not URLs)
    count = conn.execute("SELECT COUNT(DISTINCT source_name) as c FROM cluster_sources WHERE cluster_id=?", (cluster_id,)).fetchone()["c"]

    # Check if this source has higher priority
    current = conn.execute("SELECT primary_source FROM story_clusters WHERE id=?", (cluster_id,)).fetchone()
    updates = {"source_count": count, "last_updated": now}

    if get_total_score(source_name) > get_total_score(current["primary_source"]):
        updates["primary_url"] = url
        updates["primary_source"] = source_name

    # Update published_at if this item is earlier
    if published_at:
        existing_pub = conn.execute("SELECT published_at FROM story_clusters WHERE id=?", (cluster_id,)).fetchone()["published_at"]
        if existing_pub is None or published_at < existing_pub:
            updates["published_at"] = published_at

    set_clause = ", ".join(f"{k}=?" for k in updates)
    conn.execute(f"UPDATE story_clusters SET {set_clause} WHERE id=?", (*updates.values(), cluster_id))


def merge_existing_clusters(conn):
    """
    One-time pass to merge existing clusters that should be together.
    Call this after improving clustering logic to clean up the backlog.
    """
    now = datetime.now(timezone.utc)
    rows = conn.execute(
        "SELECT id, rewritten_headline, source_count FROM story_clusters WHERE expires_at > ? ORDER BY source_count DESC",
        (now.isoformat(),)
    ).fetchall()

    if len(rows) < 2:
        return 0

    headlines = [r["rewritten_headline"] or "" for r in rows]
    topic_sets = [_extract_topics(h) for h in headlines]
    keyword_sets = [_extract_keywords(h) for h in headlines]

    # Build TF-IDF matrix
    try:
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        matrix = tfidf.fit_transform(headlines)
    except Exception as e:
        log.warning(f"Merge TF-IDF error: {e}")
        return 0

    merged = set()
    merge_count = 0

    for i in range(len(rows)):
        if rows[i]["id"] in merged:
            continue
        for j in range(i + 1, len(rows)):
            if rows[j]["id"] in merged:
                continue

            # Check TF-IDF
            sim = cosine_similarity(matrix[i:i+1], matrix[j:j+1])[0][0]
            topic_overlap = _topic_overlap_score(topic_sets[i], topic_sets[j])
            kw_overlap = _keyword_overlap_score(keyword_sets[i], keyword_sets[j])

            should_merge = (
                sim >= 0.35 or
                (sim >= 0.15 and topic_overlap >= 0.5) or
                (topic_overlap >= 1.0) or
                (kw_overlap >= 3 and sim >= 0.10)
            )

            if should_merge:
                # Merge j into i (i has higher source_count due to ORDER BY)
                winner_id = rows[i]["id"]
                loser_id = rows[j]["id"]

                # Move sources
                conn.execute(
                    "UPDATE OR IGNORE cluster_sources SET cluster_id=? WHERE cluster_id=?",
                    (winner_id, loser_id)
                )
                # Move raw items
                conn.execute(
                    "UPDATE raw_items SET cluster_id=? WHERE cluster_id=?",
                    (winner_id, loser_id)
                )
                # Delete orphan sources that couldn't move (duplicate key)
                conn.execute("DELETE FROM cluster_sources WHERE cluster_id=?", (loser_id,))
                # Update winner count
                new_count = conn.execute(
                    "SELECT COUNT(DISTINCT source_name) as c FROM cluster_sources WHERE cluster_id=?",
                    (winner_id,)
                ).fetchone()["c"]
                conn.execute(
                    "UPDATE story_clusters SET source_count=?, last_updated=? WHERE id=?",
                    (new_count, now.isoformat(), winner_id)
                )
                # Delete loser
                conn.execute("DELETE FROM story_clusters WHERE id=?", (loser_id,))
                merged.add(loser_id)
                merge_count += 1

    conn.commit()
    log.info(f"Merged {merge_count} duplicate clusters")
    return merge_count
