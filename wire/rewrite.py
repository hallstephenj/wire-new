import os
import logging
import asyncio
from datetime import datetime, timezone

import anthropic

from wire.db import get_conn
from wire.config import load_config
from wire.events import push as ev

log = logging.getLogger("wire.rewrite")

SYSTEM_PROMPT = """You are a Bloomberg terminal news wire editor. You have two jobs for each cluster of headlines about the same story:

JOB 1 — REWRITE: Synthesize the cluster of headlines into a single Bloomberg-wire-style headline.
- ALL CAPS, concise, factual, present tense, active voice
- No clickbait, no questions, no articles where possible
- Max 120 characters (shorter is better)
- If one specific publication is clearly the original source or broke the story, append ": SOURCE" at the end (e.g. "OPENAI NOW TARGETING MORE THAN $280B IN REVENUE BY 2030: CNBC")
- If multiple publications reported it independently with no clear original source, do NOT append a source
- The source attribution should use the short common name (e.g. CNBC, not "CNBC News")

JOB 2 — CATEGORIZE: Assign each cluster to exactly ONE category:
- TECH: Technology, software, hardware, AI, cybersecurity, startups, gadgets, social media platforms, gaming, space/rockets, science discoveries
- MARKETS: Stock markets, earnings, IPOs, interest rates, crypto/bitcoin, financial instruments, economic indicators, company valuations
- POLITICS: Government, legislation, elections, political parties, policy, courts/legal rulings, regulation
- WORLD: International affairs, wars/conflicts, diplomacy, disasters, non-US domestic news
- GENERAL: Everything else (sports, entertainment, lifestyle, obituaries, weather, travel, food, opinion, product deals/reviews, human interest)

IMPORTANT: Stories about tariffs, trade policy, or government regulation are POLITICS unless they specifically focus on stock price impact (then MARKETS). Stories about science, space, or medicine without a tech angle are GENERAL. Product deal roundups and "best of" lists are GENERAL. Blog meta-posts ("Adding TILs to my blog") are GENERAL.

For each numbered cluster, respond with EXACTLY this format:
1. CATEGORY | REWRITTEN HEADLINE
2. CATEGORY | REWRITTEN HEADLINE
...

Examples:
1. TECH | APPLE PLANS LOW-COST MACBOOK IN MULTIPLE COLORS FOR 2026: BLOOMBERG
2. POLITICS | TRUMP VOWS NEW TARIFFS AFTER SUPREME COURT STRIKES DOWN EMERGENCY POWERS
3. MARKETS | OPENAI NOW TARGETING MORE THAN $280B IN REVENUE BY 2030: CNBC
4. WORLD | EARTHQUAKE KILLS AT LEAST 200 IN WESTERN TURKEY"""

VALID_CATEGORIES = {"tech", "markets", "politics", "world", "general"}


def _build_rewrite_prompt(clusters):
    """Build a numbered prompt from a list of cluster dicts."""
    parts = []
    for i, c in enumerate(clusters):
        headline_lines = "\n".join(f"  - [{src}] {hl}" for hl, src in c["headlines"])
        parts.append(f"{i+1}. Cluster headlines:\n{headline_lines}")
    return "\n\n".join(parts)


def _parse_rewrite_response(text, clusters, conn):
    """Parse the model response and apply rewrites/categories. Returns (rewrites, recats)."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    rewrites = 0
    recats = 0
    line_idx = 0

    for i, cluster in enumerate(clusters):
        if line_idx >= len(lines):
            break

        line = lines[line_idx]
        line_idx += 1

        # Strip numbering prefix
        for prefix in [f"{i+1}.", f"{i+1})"]:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()

        # Parse "CATEGORY | HEADLINE"
        category = None
        rewritten = line
        if "|" in line:
            parts = line.split("|", 1)
            cat_candidate = parts[0].strip().lower()
            if cat_candidate in VALID_CATEGORIES:
                category = cat_candidate
                rewritten = parts[1].strip()

        if rewritten and len(rewritten) <= 120:
            ev("rewrite", before=cluster["current_headline"], after=rewritten, category=category or "unchanged")
            conn.execute("UPDATE story_clusters SET rewritten_headline=? WHERE id=?",
                       (rewritten, cluster["id"]))
            rewrites += 1

        if category:
            conn.execute("UPDATE story_clusters SET category=? WHERE id=?",
                       (category, cluster["id"]))
            recats += 1

    return rewrites, recats


# Max clusters per single API request (Haiku handles this easily)
_BATCH_CHUNK_SIZE = 50
# Number of concurrent API requests
_PARALLEL_REQUESTS = 3


async def _rewrite_chunk(client, model, clusters):
    """Send one chunk of clusters to the API. Returns (clusters, response_text) or (clusters, None) on error."""
    prompt = _build_rewrite_prompt(clusters)
    try:
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.messages.create(
                model=model,
                max_tokens=4000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
        )
        return clusters, resp.content[0].text.strip()
    except Exception as e:
        ev("error", source="rewrite", message=str(e))
        log.error(f"Rewrite chunk error ({len(clusters)} clusters): {e}")
        return clusters, None


async def rewrite_pending():
    """Batch rewrite and categorize clustered stories.

    Rewrites clusters with 2+ sources OR scoop-boosted clusters (any source count).
    Sends up to _PARALLEL_REQUESTS concurrent batches of _BATCH_CHUNK_SIZE clusters each.
    """
    cfg = load_config()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.debug("No ANTHROPIC_API_KEY set, skipping rewrites")
        return

    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()

    # Grab up to chunk_size * parallel_requests clusters per pass
    # Include scoop-boosted clusters even if they only have 1 source
    total_limit = _BATCH_CHUNK_SIZE * _PARALLEL_REQUESTS
    rows = conn.execute("""
        SELECT sc.id, sc.rewritten_headline, sc.source_count
        FROM story_clusters sc
        LEFT JOIN curation_overrides co ON co.cluster_id = sc.id
        WHERE sc.expires_at > ?
        AND (sc.source_count >= 2 OR (COALESCE(co.boost, 0) > 0 AND co.scoop_boosted_at IS NOT NULL))
        AND (co.locked IS NULL OR co.locked = 0)
        AND EXISTS (
            SELECT 1 FROM raw_items ri
            WHERE ri.cluster_id = sc.id
            AND ri.original_headline = sc.rewritten_headline
        )
        ORDER BY sc.source_count DESC, sc.last_updated DESC
        LIMIT ?
    """, (now, total_limit)).fetchall()

    if not rows:
        conn.close()
        return

    # For each cluster, gather all headlines from its raw_items
    clusters = []
    for r in rows:
        cluster_id = r["id"]
        items = conn.execute(
            "SELECT original_headline, source_name FROM raw_items WHERE cluster_id=? ORDER BY published_at ASC",
            (cluster_id,)
        ).fetchall()
        if items:
            clusters.append({
                "id": cluster_id,
                "current_headline": r["rewritten_headline"],
                "headlines": [(i["original_headline"], i["source_name"]) for i in items],
            })

    if not clusters:
        conn.close()
        return

    # Split into chunks and send in parallel
    chunks = [clusters[i:i + _BATCH_CHUNK_SIZE] for i in range(0, len(clusters), _BATCH_CHUNK_SIZE)]
    client = anthropic.Anthropic(api_key=api_key)
    model = cfg["ai"]["model"]

    tasks = [_rewrite_chunk(client, model, chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)

    total_rewrites = 0
    total_recats = 0
    for chunk_clusters, response_text in results:
        if response_text is None:
            continue
        rw, rc = _parse_rewrite_response(response_text, chunk_clusters, conn)
        total_rewrites += rw
        total_recats += rc

    conn.commit()
    if total_rewrites > 0:
        log.info(f"Rewrote {total_rewrites} headlines, recategorized {total_recats} ({len(chunks)} parallel batches)")
    conn.close()


URGENT_REWRITE_THRESHOLD = 5  # source_count at which we trigger immediate rewrite


async def urgent_rewrite():
    """Immediately rewrite any high-coverage clusters that haven't been rewritten yet.

    Called after each poll cycle to ensure breaking stories with many sources
    don't wait for the next scheduled rewrite pass.
    """
    cfg = load_config()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return

    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()

    rows = conn.execute("""
        SELECT sc.id, sc.rewritten_headline, sc.source_count
        FROM story_clusters sc
        LEFT JOIN curation_overrides co ON co.cluster_id = sc.id
        WHERE sc.expires_at > ?
        AND sc.source_count >= ?
        AND (co.locked IS NULL OR co.locked = 0)
        AND EXISTS (
            SELECT 1 FROM raw_items ri
            WHERE ri.cluster_id = sc.id
            AND ri.original_headline = sc.rewritten_headline
        )
        ORDER BY sc.source_count DESC
        LIMIT 3
    """, (now, URGENT_REWRITE_THRESHOLD)).fetchall()

    if not rows:
        conn.close()
        return

    log.info(f"Urgent rewrite: {len(rows)} high-coverage clusters need rewriting")

    clusters = []
    for r in rows:
        items = conn.execute(
            "SELECT original_headline, source_name FROM raw_items WHERE cluster_id=? ORDER BY published_at ASC",
            (r["id"],)
        ).fetchall()
        if items:
            clusters.append({
                "id": r["id"],
                "current_headline": r["rewritten_headline"],
                "headlines": [(i["original_headline"], i["source_name"]) for i in items],
            })

    if not clusters:
        conn.close()
        return

    client = anthropic.Anthropic(api_key=api_key)
    try:
        chunk_clusters, response_text = await _rewrite_chunk(client, cfg["ai"]["model"], clusters)
        if response_text:
            rw, _ = _parse_rewrite_response(response_text, chunk_clusters, conn)
            conn.commit()
            log.info(f"Urgent rewrite done: {rw} headlines")
    except Exception as e:
        ev("error", source="urgent_rewrite", message=str(e))
        log.error(f"Urgent rewrite error: {e}")
    finally:
        conn.close()
