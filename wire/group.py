import re
import uuid
import logging
from collections import Counter
from datetime import datetime, timezone

from wire.db import get_conn

log = logging.getLogger("wire.group")

# Uppercase tokens too generic to anchor a cluster group
_STOPWORDS = frozenset({
    'THE', 'AND', 'FOR', 'FROM', 'WITH', 'THIS', 'THAT', 'HAVE', 'BEEN',
    'ARE', 'WAS', 'HAS', 'NOT', 'BUT', 'ITS', 'ALL', 'CAN', 'NEW', 'TOP',
    'AFTER', 'OVER', 'INTO', 'AMID', 'SAYS', 'SAID', 'MORE', 'ALSO', 'BACK',
    'WILL', 'WOULD', 'COULD', 'SHOULD', 'MAY', 'PLAN', 'PLANS', 'MOVE',
    'MOVES', 'SET', 'GETS', 'MAKE', 'LOOK', 'FIRST', 'LAST', 'NEXT',
    'JUST', 'STILL', 'THAN', 'WHAT', 'WHEN', 'WHO', 'HOW', 'WHY', 'NOW',
    'OUT', 'OFF', 'WAY', 'NEWS', 'REPORT', 'SOURCES', 'OFFICIALS',
    'MILLION', 'BILLION', 'PERCENT', 'DEAL', 'BILL', 'LAW', 'VOTE',
    'HOUSE', 'SENATE', 'STATE', 'MAJOR', 'LARGE', 'HIGH', 'LOW', 'LONG',
    'COMPANY', 'MARKET', 'STOCK', 'PRICE', 'SHARE', 'SHARES',
    'AGAINST', 'BEFORE', 'DURING', 'ABOUT', 'BETWEEN', 'AMID', 'OVER',
    'YEAR', 'YEARS', 'WEEK', 'WEEKS', 'DAY', 'DAYS', 'TIME', 'TIMES',
    'CALLS', 'CALL', 'PUSH', 'PUSHES', 'SEEK', 'SEEKS', 'FACE', 'FACES',
    'SAYS', 'NEED', 'NEEDS', 'SHOW', 'SHOWS', 'CITE', 'CITES', 'HITS',
    'GAIN', 'GAINS', 'LOSE', 'LOSES', 'RISE', 'RISES', 'FALL', 'FALLS',
    'AMID', 'LEAD', 'LEADS', 'HELP', 'HELPS', 'WARN', 'WARNS', 'PLAN',
    'PLANS', 'SIGN', 'SIGNS', 'FILE', 'FILES', 'HOLD', 'HOLDS', 'JOIN',
})

# Max clusters to scan per grouping run
_MAX_CLUSTERS = 500
# Minimum group size to bother creating
_MIN_GROUP_SIZE = 2
# Cap on group size — avoids mega-groups on very common entities
_MAX_GROUP_SIZE = 6
# Max number of clusters a given ngram may appear in before it's deemed too generic
_MAX_NGRAM_FREQ = 8


def _tokenize(headline: str) -> list:
    """Extract significant uppercase tokens from a rewritten headline.
    Strips source attribution suffix (': CNBC') before tokenizing."""
    headline = re.sub(r':\s+[A-Z][A-Z\s]+$', '', headline)
    tokens = re.findall(r'[A-Z]{2,}', headline)
    return [t for t in tokens if t not in _STOPWORDS and len(t) >= 3]


def _ngrams(tokens: list, n: int) -> set:
    return {' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _headline_ngrams(headline: str) -> set:
    """Bigrams + trigrams of significant tokens from a rewritten headline."""
    tokens = _tokenize(headline)
    return _ngrams(tokens, 2) | _ngrams(tokens, 3)


def _connected_components(nodes: set, edges: set) -> list:
    """Union-find: return list of connected-component sets."""
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


def assign_groups() -> int:
    """
    Group semantically related-but-distinct clusters by shared named-entity
    bigrams/trigrams. Resets all group assignments each call.
    Returns the number of groups created.
    """
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()

    # Full reset — recompute groups from scratch
    conn.execute("UPDATE story_clusters SET group_id = NULL WHERE group_id IS NOT NULL")
    conn.execute("DELETE FROM cluster_groups")

    # Active rewritten clusters only
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

    # Ngrams per cluster
    cluster_ngrams: dict = {}
    for r in rows:
        ng = _headline_ngrams(r['rewritten_headline'])
        if ng:
            cluster_ngrams[r['id']] = ng

    # ngram → [cluster_ids] index
    ngram_index: dict = {}
    for cid, ngrams in cluster_ngrams.items():
        for ng in ngrams:
            ngram_index.setdefault(ng, []).append(cid)

    # Keep only ngrams appearing in 2..MAX_NGRAM_FREQ clusters
    useful_ngrams = {
        ng: cids for ng, cids in ngram_index.items()
        if _MIN_GROUP_SIZE <= len(cids) <= _MAX_NGRAM_FREQ
    }

    if not useful_ngrams:
        conn.commit()
        conn.close()
        return 0

    # Build edges and track which ngrams each edge shares
    edges: set = set()
    edge_ngrams: dict = {}
    for ng, cids in useful_ngrams.items():
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                a, b = cids[i], cids[j]
                key = (min(a, b), max(a, b))
                edges.add(key)
                edge_ngrams.setdefault(key, []).append(ng)

    if not edges:
        conn.commit()
        conn.close()
        return 0

    all_nodes = {n for edge in edges for n in edge}
    components = _connected_components(all_nodes, edges)

    groups_created = 0
    for component in components:
        if len(component) > _MAX_GROUP_SIZE:
            continue

        # Best label: most common shared ngram, preferring trigrams (more specific)
        all_shared = []
        cids = list(component)
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                key = (min(cids[i], cids[j]), max(cids[i], cids[j]))
                all_shared.extend(edge_ngrams.get(key, []))

        if not all_shared:
            continue

        counts = Counter(all_shared)
        label = max(counts, key=lambda ng: counts[ng] * (1 + ng.count(' ')))

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
