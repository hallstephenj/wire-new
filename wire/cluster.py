import uuid
import logging
from datetime import datetime, timezone, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from wire.config import load_config
from wire.events import push as ev
from wire.scores import get_total_score

log = logging.getLogger("wire.cluster")

_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
_fitted = False
_corpus_ids = []
_corpus_headlines = []

def assign_cluster(conn, headline: str, url: str, source_name: str, category: str, published_at: str = None) -> str:
    """Find or create a cluster for this headline. Returns cluster_id."""
    cfg = load_config()
    threshold = cfg["clustering"]["similarity_threshold"]
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
        all_texts = cluster_headlines + [headline]
        try:
            tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
            matrix = tfidf.fit_transform(all_texts)
            sims = cosine_similarity(matrix[-1:], matrix[:-1])[0]
            best_idx = sims.argmax()
            best_sim = sims[best_idx]

            if best_sim >= threshold:
                cluster_id = rows[best_idx]["id"]
                ev("cluster_hit", headline=headline, matched=rows[best_idx]["rewritten_headline"], similarity=round(float(best_sim), 3))
                _add_to_cluster(conn, cluster_id, source_name, url, headline, category, published_at=published_at)
                return cluster_id
        except Exception as e:
            log.warning(f"Clustering error: {e}")

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
