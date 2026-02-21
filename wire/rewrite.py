import os
import logging
import asyncio
from datetime import datetime, timezone

import anthropic

from wire.db import get_conn
from wire.config import load_config

log = logging.getLogger("wire.rewrite")

SYSTEM_PROMPT = """You are a Bloomberg terminal news wire editor. You have two jobs:

JOB 1 — REWRITE: Rewrite each headline in Bloomberg wire style:
- Concise (max 80 chars preferred, 100 max)
- Factual/neutral, present tense, no articles where possible, active voice
- No clickbait, no questions

JOB 2 — CATEGORIZE: Assign each headline to exactly ONE category:
- TECH: Technology, software, hardware, AI, cybersecurity, startups, gadgets, social media platforms, gaming, space/rockets, science discoveries
- MARKETS: Stock markets, earnings, IPOs, interest rates, crypto/bitcoin, financial instruments, economic indicators, company valuations
- POLITICS: Government, legislation, elections, political parties, policy, courts/legal rulings, regulation
- WORLD: International affairs, wars/conflicts, diplomacy, disasters, non-US domestic news
- GENERAL: Everything else (sports, entertainment, lifestyle, obituaries, weather, travel, food, opinion, product deals/reviews, human interest)

IMPORTANT: Stories about tariffs, trade policy, or government regulation are POLITICS unless they specifically focus on stock price impact (then MARKETS). Stories about science, space, or medicine without a tech angle are GENERAL. Product deal roundups and "best of" lists are GENERAL. Blog meta-posts ("Adding TILs to my blog") are GENERAL.

For each numbered headline, respond with EXACTLY this format:
1. CATEGORY | Rewritten headline
2. CATEGORY | Rewritten headline
...

Example:
1. TECH | Apple Plans Low-Cost MacBook in Multiple Colors for 2026
2. POLITICS | Trump Vows New Tariffs After Supreme Court Strikes Down Emergency Powers
3. MARKETS | OpenAI Revenue Forecast Tops $280B by 2030
4. GENERAL | Winter Storm Threatens Northeast With More Snow This Weekend"""

_pending = []
_lock = asyncio.Lock()
_last_flush = None

VALID_CATEGORIES = {"tech", "markets", "politics", "world", "general"}


async def rewrite_pending():
    """Batch rewrite and categorize headlines."""
    cfg = load_config()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.debug("No ANTHROPIC_API_KEY set, skipping rewrites")
        return

    conn = get_conn()
    rows = conn.execute("""
        SELECT sc.id, sc.rewritten_headline
        FROM story_clusters sc
        LEFT JOIN raw_items ri ON ri.cluster_id = sc.id
        WHERE sc.expires_at > ?
        AND sc.rewritten_headline = ri.original_headline
        LIMIT ?
    """, (datetime.now(timezone.utc).isoformat(), cfg["ai"]["batch_size"])).fetchall()

    if not rows:
        conn.close()
        return

    headlines = {r["id"]: r["rewritten_headline"] for r in rows}
    numbered = "\n".join(f"{i+1}. {h}" for i, (cid, h) in enumerate(headlines.items()))

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=cfg["ai"]["model"],
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": numbered}]
        )
        text = resp.content[0].text.strip()
        lines = text.split("\n")

        cluster_ids = list(headlines.keys())
        rewrites = 0
        recats = 0

        for i, line in enumerate(lines):
            if i >= len(cluster_ids):
                break

            line = line.strip()
            if not line:
                continue

            # Strip numbering prefix
            for prefix in [f"{i+1}.", f"{i+1})"]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()

            # Parse "CATEGORY | Headline"
            category = None
            rewritten = line
            if "|" in line:
                parts = line.split("|", 1)
                cat_candidate = parts[0].strip().lower()
                if cat_candidate in VALID_CATEGORIES:
                    category = cat_candidate
                    rewritten = parts[1].strip()

            # Update headline
            if rewritten and len(rewritten) <= 100:
                conn.execute("UPDATE story_clusters SET rewritten_headline=? WHERE id=?",
                           (rewritten, cluster_ids[i]))
                rewrites += 1

            # Update category if AI classified it
            if category:
                conn.execute("UPDATE story_clusters SET category=? WHERE id=?",
                           (category, cluster_ids[i]))
                recats += 1

        conn.commit()
        log.info(f"Rewrote {rewrites} headlines, recategorized {recats}")
    except Exception as e:
        log.error(f"Rewrite error: {e}")
    finally:
        conn.close()
