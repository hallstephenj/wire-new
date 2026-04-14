import re
import uuid
import logging
from datetime import datetime, timezone

from wire.db import get_conn

log = logging.getLogger("wire.group")

# Similarity range for "related but not duplicate" grouping.
# Below GROUP_SIM_MIN → unrelated. Above GROUP_SIM_MAX → should have been deduped.
GROUP_SIM_MIN = 0.12
GROUP_SIM_MAX = 0.28

# Max clusters to scan per run
_MAX_CLUSTERS = 500
# Min members to form a visible group
_MIN_GROUP_SIZE = 2
# Cap to avoid mega-groups on very common topics
_MAX_GROUP_SIZE = 6

# Demonym / adjective → canonical noun normalization applied before vectorizing.
# Prevents IRANIAN and IRAN from being treated as different tokens.
_NORM_MAP = {
    'IRANIAN': 'IRAN', 'IRANIANS': 'IRAN',
    'RUSSIAN': 'RUSSIA', 'RUSSIANS': 'RUSSIA',
    'UKRAINIAN': 'UKRAINE', 'UKRAINIANS': 'UKRAINE',
    'CHINESE': 'CHINA',
    'ISRAELI': 'ISRAEL', 'ISRAELIS': 'ISRAEL',
    'TAIWANESE': 'TAIWAN',
    'PALESTINIAN': 'PALESTINE', 'PALESTINIANS': 'PALESTINE',
    'GAZAN': 'GAZA', 'GAZANS': 'GAZA',
    'SYRIAN': 'SYRIA', 'SYRIANS': 'SYRIA',
    'KOREAN': 'KOREA', 'KOREANS': 'KOREA',
    'SAUDI': 'SAUDI',
    'AFGHAN': 'AFGHANISTAN', 'AFGHANS': 'AFGHANISTAN',
    'TURKISH': 'TURKEY', 'TURKS': 'TURKEY',
    'MEXICAN': 'MEXICO', 'MEXICANS': 'MEXICO',
    'CUBAN': 'CUBA', 'CUBANS': 'CUBA',
    'VENEZUELAN': 'VENEZUELA', 'VENEZUELANS': 'VENEZUELA',
    'PAKISTANI': 'PAKISTAN', 'PAKISTANIS': 'PAKISTAN',
}
_NORM_RE = re.compile(r'\b(' + '|'.join(re.escape(k) for k in _NORM_MAP) + r')\b')


def _normalize(headline: str) -> str:
    """Apply demonym/adjective normalization so e.g. IRANIAN → IRAN."""
    return _NORM_RE.sub(lambda m: _NORM_MAP[m.group()], headline)


def _connected_components(nodes: set, edges: set) -> list:
    """Union-find: return list of connected-component node-sets."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)

    components: dict = {}
    for node in nodes:
        root = find(node)
        components.setdefault(root, set()).add(node)

    return [c for c in components.values() if len(c) >= _MIN_GROUP_SIZE]


def _group_label(member_headlines: list) -> str:
    """Pick the best human-readable label for a group.

    Finds words that appear in the most member headlines (after normalization
    and stopword filtering) weighted by length, preferring longer proper-noun
    sequences.
    """
    _LABEL_STOP = frozenset({
        'the', 'and', 'for', 'from', 'with', 'this', 'that', 'have', 'been',
        'are', 'was', 'has', 'not', 'but', 'its', 'all', 'can', 'new', 'top',
        'after', 'over', 'into', 'amid', 'says', 'said', 'more', 'also', 'back',
        'will', 'would', 'could', 'should', 'may', 'plan', 'move', 'set', 'gets',
        'make', 'look', 'first', 'last', 'next', 'just', 'still', 'than', 'now',
        'out', 'off', 'way', 'news', 'report', 'sources', 'officials', 'begin',
        'begins', 'threat', 'threats', 'call', 'calls', 'push', 'seek', 'face',
        'gain', 'lose', 'rise', 'fall', 'lead', 'help', 'warn', 'sign', 'file',
        'hold', 'join', 'show', 'cite', 'hit', 'use', 'say', 'get', 'go', 'do',
    })

    from collections import Counter
    word_counts: Counter = Counter()
    for hl in member_headlines:
        words = re.findall(r'[a-z]{3,}', _normalize(hl).lower())
        for w in set(words):
            if w not in _LABEL_STOP:
                word_counts[w] += 1

    if not word_counts:
        return 'RELATED'

    # Score: (appearances across members) × word_length — longer specific words win
    best = max(word_counts, key=lambda w: word_counts[w] * len(w))
    return best.upper()


def assign_groups() -> int:
    """
    Group related-but-distinct clusters using TF-IDF cosine similarity on
    normalized rewritten headlines. Pairs with similarity in
    [GROUP_SIM_MIN, GROUP_SIM_MAX] are considered "related". Connected
    components of related pairs become groups.

    Runs a full reset each call — previous group_ids are cleared and reassigned.
    Returns the number of groups created.
    """
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        log.warning("sklearn not available — skipping grouping")
        return 0

    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()

    # Full reset — recompute groups from scratch each run
    conn.execute("UPDATE story_clusters SET group_id = NULL WHERE group_id IS NOT NULL")
    conn.execute("DELETE FROM cluster_groups")

    rows = conn.execute("""
        SELECT id, rewritten_headline
        FROM story_clusters
        WHERE expires_at > ?
          AND rewritten_headline IS NOT NULL
          AND source_count >= 2
        ORDER BY published_at DESC
        LIMIT ?
    """, (now, _MAX_CLUSTERS)).fetchall()

    if len(rows) < _MIN_GROUP_SIZE:
        conn.commit()
        conn.close()
        return 0

    ids = [r['id'] for r in rows]
    # Normalize demonyms before vectorizing so IRANIAN and IRAN are the same token
    headlines = [_normalize(r['rewritten_headline']) for r in rows]

    # TF-IDF on unigrams + bigrams; sublinear_tf reduces impact of repeated words
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        tfidf = vec.fit_transform(headlines)
    except ValueError:
        conn.commit()
        conn.close()
        return 0

    # Pairwise cosine similarities — 500×500 is fast
    sims = cosine_similarity(tfidf)

    # Build edges for pairs in the related-but-not-duplicate similarity band
    edges: set = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sim = float(sims[i, j])
            if GROUP_SIM_MIN <= sim <= GROUP_SIM_MAX:
                edges.add((ids[i], ids[j]))

    if not edges:
        conn.commit()
        conn.close()
        return 0

    all_nodes = {n for edge in edges for n in edge}
    components = _connected_components(all_nodes, edges)

    # Map id → headline for label computation
    headline_by_id = {r['id']: r['rewritten_headline'] for r in rows}

    groups_created = 0
    for component in components:
        if len(component) > _MAX_GROUP_SIZE:
            continue

        label = _group_label([headline_by_id[cid] for cid in component])
        group_id = str(uuid.uuid4())

        conn.execute(
            "INSERT INTO cluster_groups (id, label, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (group_id, label, now, now),
        )
        ph = ','.join('?' for _ in component)
        conn.execute(
            f"UPDATE story_clusters SET group_id = ? WHERE id IN ({ph})",
            [group_id] + list(component),
        )
        groups_created += 1

    conn.commit()
    conn.close()
    if groups_created:
        log.info(f"Grouped {groups_created} related-story groups")
    return groups_created
