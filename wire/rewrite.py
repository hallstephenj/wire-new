import os
import logging
import asyncio
from datetime import datetime, timezone

import anthropic

from wire.db import get_conn
from wire.config import load_config

log = logging.getLogger("wire.rewrite")

SYSTEM_PROMPT = """You are a Bloomberg terminal news wire editor. Rewrite the following headlines to be:
- Concise (max 80 chars preferred, 100 max)
- Factual/neutral
- Bloomberg wire style (present tense, no articles where possible, active voice)
- No clickbait/questions

For each headline, return ONLY the rewritten headline, one per line, in the same order as input.
Number each line to match the input numbering."""

_pending = []
_lock = asyncio.Lock()
_last_flush = None

async def rewrite_pending():
    """Batch rewrite headlines that haven't been rewritten yet."""
    cfg = load_config()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.debug("No ANTHROPIC_API_KEY set, skipping rewrites")
        return

    conn = get_conn()
    # Find clusters whose headline is still the raw original (no rewrite yet)
    # We use a simple heuristic: headlines > 100 chars or all of them periodically
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
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": numbered}]
        )
        text = resp.content[0].text.strip()
        lines = text.split("\n")

        cluster_ids = list(headlines.keys())
        for i, line in enumerate(lines):
            if i >= len(cluster_ids):
                break
            # Strip numbering
            rewritten = line.strip()
            for prefix in [f"{i+1}.", f"{i+1})"]:
                if rewritten.startswith(prefix):
                    rewritten = rewritten[len(prefix):].strip()
            if rewritten and len(rewritten) <= 100:
                conn.execute("UPDATE story_clusters SET rewritten_headline=? WHERE id=?",
                           (rewritten, cluster_ids[i]))

        conn.commit()
        log.info(f"Rewrote {len(lines)} headlines")
    except Exception as e:
        log.error(f"Rewrite error: {e}")
    finally:
        conn.close()
