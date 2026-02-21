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


async def rewrite_pending():
    """Batch rewrite and categorize clustered stories (source_count >= 2)."""
    cfg = load_config()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.debug("No ANTHROPIC_API_KEY set, skipping rewrites")
        return

    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()

    # Only rewrite clusters with 2+ sources whose headline still matches
    # a raw item headline (i.e. hasn't been rewritten yet)
    rows = conn.execute("""
        SELECT sc.id, sc.rewritten_headline, sc.source_count
        FROM story_clusters sc
        WHERE sc.expires_at > ?
        AND sc.source_count >= 2
        AND EXISTS (
            SELECT 1 FROM raw_items ri
            WHERE ri.cluster_id = sc.id
            AND ri.original_headline = sc.rewritten_headline
        )
        LIMIT ?
    """, (now, cfg["ai"]["batch_size"])).fetchall()

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

    # Build the prompt: each numbered cluster with all its headlines
    parts = []
    for i, c in enumerate(clusters):
        headline_lines = "\n".join(f"  - [{src}] {hl}" for hl, src in c["headlines"])
        parts.append(f"{i+1}. Cluster headlines:\n{headline_lines}")
    numbered = "\n\n".join(parts)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=cfg["ai"]["model"],
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": numbered}]
        )
        text = resp.content[0].text.strip()
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

        conn.commit()
        log.info(f"Rewrote {rewrites} headlines, recategorized {recats}")
    except Exception as e:
        ev("error", source="rewrite", message=str(e))
        log.error(f"Rewrite error: {e}")
    finally:
        conn.close()
