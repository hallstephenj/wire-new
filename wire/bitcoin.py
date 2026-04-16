"""
BITCOIN WIRE — /bitcoin

A focused, exhaustive Bitcoin-only news aggregator built on the Edrowire engine.
Ingests Bitcoin-specific feeds and search queries, filters the stream to Bitcoin
stories only, categorizes via a dedicated Bitcoin AI prompt, and serves the result
via its own route at /bitcoin.

Categories (stored as-is in story_clusters.category):
  bitcoin-markets   Price, ETF flows, institutional, treasury, exchanges
  bitcoin-policy    Regulation, legislation, government, CBDCs, reserves
  bitcoin-tech      Protocol, Lightning, Taproot, Ordinals, development
  bitcoin-mining    Mining companies, hashrate, difficulty, energy, ASICs
  bitcoin-macro     Inflation, gold, fiat debasement, monetary policy
  bitcoin-culture   Adoption, conferences, education, circular economy
"""

import os
import json
import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
from fastapi import Query, Request
from fastapi.responses import JSONResponse

from wire.db import get_conn
from wire.events import push as ev
from wire.config import load_config
from wire.algorithm import get_active_version, build_hot_score_sql, build_market_mover_case_sql

log = logging.getLogger("wire.bitcoin")

# ── Bitcoin category taxonomy ─────────────────────────────────────────────

BITCOIN_CATEGORIES = {
    "bitcoin-markets",
    "bitcoin-policy",
    "bitcoin-tech",
    "bitcoin-mining",
    "bitcoin-macro",
    "bitcoin-culture",
}

# Display labels for tabs (key = slug used in API, value = tab label)
BTC_CATEGORY_LABELS = {
    "markets": "Markets",
    "policy":  "Policy",
    "tech":    "Tech",
    "mining":  "Mining",
    "macro":   "Macro",
    "culture": "Culture",
}

# ── Bitcoin-native publication sources ────────────────────────────────────
# Articles from these sources bypass the Bitcoin signal check —
# every story they publish is Bitcoin-relevant by definition.

BITCOIN_SOURCES = {
    "Bitcoin Magazine",
    "Bitcoin Optech",
    "Bitcoin Optech Newsletter",
    "Unchained",
    "Unchained Capital",
    "River",
    "River Financial",
    "River Blog",
    "Swan Bitcoin",
    "Swan",
    "Blockstream",
    "Blockstream Blog",
    "Lightning Labs",
    "Casa",
    "Keys.casa",
    "Voltage",
    "Voltage Blog",
    "Strike",
    "Strike Blog",
    "Spiral",
    "Spiral Blog",
    "Bitcoin Policy Institute",
    "NYDIG",
    "Lyn Alden",
    "Saifedean Ammous",
    "10x Research",
    "Glassnode",
    "Glassnode Insights",
    "Braiins",
    "Mempool Space",
    "WhatBitcoinDid",
    "Stephan Livera",
    "Human Rights Foundation",
}

# ── Bitcoin signal detection ──────────────────────────────────────────────
# A story must contain at least one of these substrings to qualify as
# a Bitcoin story (unless the source is in BITCOIN_SOURCES).

_BITCOIN_SIGNALS = frozenset({
    "bitcoin", "#bitcoin", "btc ", " btc", "btc,", "btc.",
    "lightning network", "lightning payment",
    "taproot", "segwit", "ordinals", "bitcoin inscription",
    "bitcoin runes", "bitcoin rune",
    "bitcoin miner", "bitcoin mining", "bitcoin hashrate",
    "bitcoin halving", "bitcoin etf", "spot bitcoin",
    "bitcoin treasury", "bitcoin reserve", "bitcoin wallet",
    "bitcoin price", "bitcoin rally", "bitcoin drop", "bitcoin crash",
    "bitcoin bull", "bitcoin bear", "bitcoin all-time",
    "proof of work", "bitcoin core", "bitcoin protocol",
    "bitcoin node", "bitcoin block", "bitcoin transaction",
    "bitcoin adoption", "bitcoin holder", "bitcoin hodl",
    "satoshi nakamoto", "satoshi", "sats",
    "michael saylor", "microstrategy bitcoin",
})

# Coins whose dominant presence (without Bitcoin context) means reject.
_ALTCOIN_DOMINANT = frozenset({
    "ethereum", "solana", "cardano", " xrp ", "ripple",
    "dogecoin", "doge", "shiba inu", "avalanche avax",
    "polygon matic", "chainlink link", "polkadot dot",
    "litecoin ltc", "stellar xlm", "tron trx",
    "cosmos atom", "near protocol", "filecoin fil",
    "algorand algo", "tezos xtz", "monero xmr",
    "zcash zec", "dash coin",
})


def is_bitcoin_story(headline: str, source_name: str = "") -> bool:
    """Return True if this article is primarily about Bitcoin.

    Rules:
    1. Bitcoin-native sources always pass.
    2. Headline must contain a Bitcoin signal substring.
    3. If the headline's primary subject is an altcoin with no Bitcoin mention,
       reject it — these slip through on general crypto queries.
    """
    if source_name in BITCOIN_SOURCES:
        return True

    h = headline.lower()

    # Must have a Bitcoin signal
    has_signal = any(s in h for s in _BITCOIN_SIGNALS)
    if not has_signal:
        # Weaker check: just the word bitcoin or btc anywhere
        if "bitcoin" not in h and "btc" not in h:
            return False

    # Reject if altcoin is the dominant subject with no Bitcoin mention
    if "bitcoin" not in h and "btc" not in h:
        for coin in _ALTCOIN_DOMINANT:
            if coin in h:
                return False

    return True


# ── Bitcoin sub-category classifier ──────────────────────────────────────
# Lightweight heuristic for newly ingested items before the AI rewriter runs.

_BTC_MARKETS_KW = frozenset({
    "price", "etf", "etfs", "fund", "funds", "spot", "institutional",
    "treasury", "corporate", "strategy", "microstrategy", "blackrock",
    "fidelity", "ark invest", "vaneck", "spot etf", "holdings", "reserve",
    "exchange", "coinbase", "kraken", "binance", "trading", "market cap",
    "all-time high", "ath", "bull", "bear", "rally", "crash", "inflow",
    "outflow", "investment", "investors", "custody", "asset management",
    "billion", "accumulate", "hodl", "grayscale", "gbtc", "ibit",
    "valuation", "flows", "on-chain", "supply", "demand", "whale",
    "long-term holder", "short-term holder", "realized cap",
})

_BTC_POLICY_KW = frozenset({
    "regulation", "regulatory", "regulator", "sec", "cftc", "congress",
    "senate", "legislation", "law", "government", "national",
    "legal", "court", "cbdc", "ban", "license", "compliance",
    "tax", "irs", "treasury department", "policy", "senator",
    "representative", "white house", "executive order",
    "strategic reserve", "state bill", "el salvador", "legal tender",
    "reserve currency", "federal", "doj", "fbi", "sanctions",
    "money transmission", "aml", "kyc", "financial crime",
    "department of justice", "attorney general",
})

_BTC_TECH_KW = frozenset({
    "lightning", "taproot", "segwit", "ordinals", "inscriptions", "runes",
    "protocol", "upgrade", "soft fork", "hard fork", "bitcoin core",
    "development", "bip", "wallet", "node", "mempool", "transaction",
    "signature", "multisig", "script", "layer 2", "channel", "routing",
    "voltage", "breez", "phoenix", "lnd", "cln", "eclair", "open source",
    "github", "developer", "whitepaper", "cryptography", "schnorr",
    "musig", "eltoo", "ark protocol", "fedimint", "cashu",
})

_BTC_MINING_KW = frozenset({
    "mining", "miner", "miners", "hashrate", "hash rate", "difficulty",
    "asic", "rig", "energy", "power", "electricity", "kilowatt", "megawatt",
    "marathon", "riot", "cleanspark", "core scientific", "hut 8",
    "bitdeer", "cipher mining", "antminer", "bitmain", "foundry",
    "pool", "block reward", "subsidy", "fee", "nonce", "stratum",
    "farm", "data center", "nuclear", "natural gas", "hydro",
    "renewable", "carbon", "wattage",
})

_BTC_MACRO_KW = frozenset({
    "inflation", "dollar", "usd", "gold", "hedge", "macro", "fiat",
    "monetary", "central bank", "fed", "federal reserve", "interest rate",
    "store of value", "hard money", "sound money", "debasement",
    "purchasing power", "devaluation", "hyperinflation", "fiscal",
    "debt", "deficit", "quantitative easing", "money printing",
    "dxy", "m2", "treasury yield", "cpi", "ppi", "gdp",
})


def classify_bitcoin_headline(headline: str) -> str:
    """Heuristic category assignment for a Bitcoin story (pre-AI rewrite)."""
    h = headline.lower()

    def score(kw_set):
        return sum(1 for kw in kw_set if kw in h)

    scores = {
        "bitcoin-markets": score(_BTC_MARKETS_KW),
        "bitcoin-policy":  score(_BTC_POLICY_KW),
        "bitcoin-tech":    score(_BTC_TECH_KW),
        "bitcoin-mining":  score(_BTC_MINING_KW),
        "bitcoin-macro":   score(_BTC_MACRO_KW),
    }

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    return "bitcoin-markets"  # default: most common Bitcoin story type


# ── Bitcoin-specific reputable sources ────────────────────────────────────
# Used in hot-score filter. Bitcoin-native pubs count as reputable here
# even though they score lower in the main wire's quality tiers.
# Min threshold: 1 (vs 3 in main wire) — Bitcoin breaks in native pubs first.

BTC_REPUTABLE_SOURCES = {
    "Reuters", "AP",
    "Bloomberg", "Wall Street Journal", "Financial Times",
    "New York Times", "Washington Post", "BBC",
    "CNBC", "Forbes", "Fortune", "Business Insider", "Barron's",
    "Axios", "Wired", "TechCrunch", "The Verge",
    "MarketWatch", "Yahoo Finance", "Fox Business", "Seeking Alpha",
    # Bitcoin-native — elevated for Bitcoin wire
    "Bitcoin Magazine", "Unchained", "Unchained Capital",
    "CoinDesk", "The Block", "Decrypt", "Cointelegraph",
    "River", "River Financial", "Swan Bitcoin", "Swan",
    "Bitcoin Optech", "Blockstream", "Lightning Labs",
    "Casa", "Voltage", "Strike",
    "10x Research", "Glassnode", "Glassnode Insights",
    "NYDIG", "Lyn Alden",
}


# ── BTC price cache ───────────────────────────────────────────────────────

_price_cache: dict = {"data": None, "fetched_at": 0.0}
_PRICE_CACHE_TTL = 60  # seconds


async def get_btc_price() -> dict:
    """Fetch BTC price from CoinGecko public API, cached 60s."""
    now = time.time()
    if _price_cache["data"] and (now - _price_cache["fetched_at"]) < _PRICE_CACHE_TTL:
        return _price_cache["data"]

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={
                    "ids": "bitcoin",
                    "vs_currencies": "usd",
                    "include_24hr_change": "true",
                    "include_market_cap": "true",
                },
                headers={"Accept": "application/json", "User-Agent": "EDROWIRE/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
            btc = data.get("bitcoin", {})
            result = {
                "price": btc.get("usd", 0),
                "change_24h": round(btc.get("usd_24h_change", 0), 2),
                "market_cap": btc.get("usd_market_cap", 0),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            _price_cache["data"] = result
            _price_cache["fetched_at"] = now
            return result
    except Exception as e:
        log.warning(f"BTC price fetch failed: {e}")
        return _price_cache["data"] or {"price": 0, "change_24h": 0, "market_cap": 0, "fetched_at": None}


# ── Halving calculator ────────────────────────────────────────────────────

def get_halving_info() -> dict:
    """Estimate current block height and next halving timing.

    The 4th halving occurred at block 840,000 on April 20, 2024.
    The 5th halving will occur at block 1,050,000.
    Uses 10-minute average block time as an approximation.
    """
    FOURTH_HALVING_BLOCK = 840_000
    FOURTH_HALVING_DATE = datetime(2024, 4, 20, 0, 0, 0, tzinfo=timezone.utc)
    BLOCKS_PER_HALVING = 210_000
    AVG_BLOCK_SECONDS = 600

    now = datetime.now(timezone.utc)
    elapsed_s = (now - FOURTH_HALVING_DATE).total_seconds()
    blocks_since = int(elapsed_s / AVG_BLOCK_SECONDS)
    current_block = FOURTH_HALVING_BLOCK + blocks_since

    next_halving_block = FOURTH_HALVING_BLOCK + BLOCKS_PER_HALVING
    blocks_remaining = max(0, next_halving_block - current_block)
    seconds_remaining = blocks_remaining * AVG_BLOCK_SECONDS
    next_date = now + timedelta(seconds=seconds_remaining)

    return {
        "current_block": f"{current_block:,}",
        "next_halving_block": f"{next_halving_block:,}",
        "blocks_remaining": f"{blocks_remaining:,}",
        "days_remaining": max(0, int(seconds_remaining / 86400)),
        "estimated_date": next_date.strftime("%b %Y"),
    }


# ── Bitcoin search queries ────────────────────────────────────────────────
# Run by bitcoin_search_sweep() independently of the main search_sweep.
# Queries are paired with their target bitcoin-* category.

BITCOIN_SEARCH_QUERIES = [
    # Markets & price
    ("bitcoin price today",                  "bitcoin-markets"),
    ("bitcoin etf flows today",              "bitcoin-markets"),
    ("bitcoin institutional adoption",       "bitcoin-markets"),
    ("bitcoin corporate treasury",           "bitcoin-markets"),
    ("microstrategy bitcoin",                "bitcoin-markets"),
    ("strategy bitcoin holdings",            "bitcoin-markets"),
    ("blackrock bitcoin etf ibit",           "bitcoin-markets"),
    ("fidelity bitcoin fund",                "bitcoin-markets"),
    ("bitcoin all time high",                "bitcoin-markets"),
    ("bitcoin exchange reserves",            "bitcoin-markets"),
    ("bitcoin spot etf",                     "bitcoin-markets"),
    ("bitcoin whale accumulation",           "bitcoin-markets"),
    ("bitcoin on-chain data",                "bitcoin-markets"),
    ("bitcoin hodlers long term",            "bitcoin-markets"),
    # Policy & regulation
    ("bitcoin regulation news",              "bitcoin-policy"),
    ("bitcoin legislation congress senate",  "bitcoin-policy"),
    ("bitcoin strategic reserve",            "bitcoin-policy"),
    ("bitcoin government policy",            "bitcoin-policy"),
    ("bitcoin cbdc central bank",            "bitcoin-policy"),
    ("bitcoin legal tender country",         "bitcoin-policy"),
    ("bitcoin sec cftc ruling",              "bitcoin-policy"),
    ("bitcoin money laundering law",         "bitcoin-policy"),
    # Technology
    ("bitcoin lightning network",            "bitcoin-tech"),
    ("bitcoin taproot ordinals",             "bitcoin-tech"),
    ("bitcoin protocol upgrade development", "bitcoin-tech"),
    ("bitcoin layer 2 scaling",              "bitcoin-tech"),
    ("bitcoin core developers",              "bitcoin-tech"),
    ("bitcoin inscriptions runes",           "bitcoin-tech"),
    # Mining
    ("bitcoin mining news today",            "bitcoin-mining"),
    ("bitcoin hashrate difficulty",          "bitcoin-mining"),
    ("bitcoin mining company earnings",      "bitcoin-mining"),
    ("bitcoin mining energy renewable",      "bitcoin-mining"),
    ("bitcoin halving 2028 miners",          "bitcoin-mining"),
    ("marathon riot cleanspark bitcoin",     "bitcoin-mining"),
    # Macro
    ("bitcoin inflation hedge",              "bitcoin-macro"),
    ("bitcoin gold comparison store value",  "bitcoin-macro"),
    ("bitcoin macro economics",              "bitcoin-macro"),
    ("bitcoin federal reserve interest",     "bitcoin-macro"),
    ("bitcoin fiat debasement hard money",   "bitcoin-macro"),
    # General / breaking
    ("bitcoin news today",                   "bitcoin-markets"),
    ("bitcoin breaking news",                "bitcoin-markets"),
    ("bitcoin adoption latest",              "bitcoin-culture"),
    ("bitcoin conference summit",            "bitcoin-culture"),
    ("bitcoin circular economy",             "bitcoin-culture"),
]


async def bitcoin_search_sweep(on_progress=None):
    """Run Bitcoin-focused Google News searches, ingest results with bitcoin-* categories."""
    from wire.ingest import _poll_single_feed, _gnews_blocked

    if _gnews_blocked:
        log.debug("Bitcoin search sweep skipped (Google News consent wall active)")
        return

    log.info("Bitcoin search sweep starting...")
    ev("ingest_start", job="bitcoin_search")

    count = 0
    total = len(BITCOIN_SEARCH_QUERIES)
    completed = 0
    sem = asyncio.Semaphore(3)

    async def _run_query(query: str, category: str):
        nonlocal completed, count
        async with sem:
            try:
                encoded = query.replace(" ", "+")
                url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
                n = await _poll_single_feed(url, "Google News", category)
                count += n
            except Exception as e:
                log.debug(f"Bitcoin search error ({query!r}): {e}")
            completed += 1
            if on_progress:
                on_progress(completed, total)

    await asyncio.gather(*[_run_query(q, cat) for q, cat in BITCOIN_SEARCH_QUERIES])
    log.info(f"Bitcoin search sweep done: {count} new items")
    ev("ingest_done", job="bitcoin_search", items=count)


# ── Bitcoin AI rewriter ───────────────────────────────────────────────────

BITCOIN_SYSTEM_PROMPT = """You are the Bitcoin Wire editor — the authoritative voice for Bitcoin news. \
You have deep knowledge of Bitcoin: the protocol, Lightning Network, Taproot, Ordinals, mining economics, \
ETF mechanics, regulatory landscape, macro thesis, and key players. \
You are NOT a crypto editor. You cover Bitcoin only.

Key people: Satoshi Nakamoto, Michael Saylor, Jack Dorsey, Cathie Wood, Larry Fink, Adam Back, \
Pieter Wuille, Greg Maxwell, Luke Dashjr, Peter Todd, Jimmy Song, Lyn Alden, Saifedean Ammous.

Key companies: Strategy (MicroStrategy), BlackRock, Fidelity, ARK Invest, VanEck, Grayscale, \
Coinbase, Kraken, Binance, River Financial, Swan Bitcoin, Unchained Capital, Casa, Strike, \
Voltage, Blockstream, Lightning Labs, Spiral, NYDIG, Marathon Digital, Riot Platforms, \
CleanSpark, Core Scientific, Hut 8, Bitdeer, Foundry, Bitmain.

You have four jobs for each cluster of Bitcoin headlines:

JOB 1 — REWRITE: Synthesize into a single sharp Bitcoin wire headline.
- ALL CAPS, concise, factual, present tense, active voice
- No clickbait, no questions, no articles where possible
- Max 120 characters (shorter is better)
- If one publication clearly broke the story: append ": SOURCE" (e.g. ": COINDESK")
- Use precise Bitcoin terminology: HASHRATE not COMPUTING POWER, LN not LIGHTNING, \
HALVING not REWARD REDUCTION, SAT not SATOSHI (in quantity contexts), ASIC not MINING HARDWARE

JOB 2 — CATEGORIZE: Assign exactly ONE category:
- BTC-MARKETS: Price, ETF flows, institutional buying/selling, corporate treasury, exchange reserves, \
on-chain balances, custody, trading, valuations, derivatives, long-term holder data
- BTC-POLICY: Regulation, legislation, SEC/CFTC actions, legal rulings, national reserves, \
El Salvador/country adoption policy, AML/KYC, sanctions, government statements about Bitcoin
- BTC-TECH: Lightning Network, Taproot, Ordinals, Runes, Inscriptions, Bitcoin Core development, \
BIPs, soft forks, wallets, nodes, mempool, cryptography, Schnorr, MuSig, Ark, Fedimint, Cashu
- BTC-MINING: Mining companies (Marathon, Riot, CleanSpark, Hut 8), hashrate, difficulty, \
ASICs, mining pools, energy (nuclear, hydro, stranded gas), block rewards, fee market
- BTC-MACRO: Bitcoin as inflation hedge, gold comparison, fiat debasement, monetary policy impact, \
central bank decisions affecting Bitcoin, DXY, M2, CPI impact on BTC narrative
- BTC-CULTURE: Grassroots adoption, circular economies, Bitcoin conferences (BTC Prague, \
Bitcoin Nashville, Lugano, etc.), education, podcasts, books, community, art on Bitcoin

JOB 3 — MARKET IMPACT: Rate price impact on Bitcoin (0-3):
- 3 (HIGH): Major ETF approval/rejection, significant government ban/adoption, \
major institutional announcement >$1B, systemic exchange failure, landmark regulatory action
- 2 (MEDIUM): ETF flow data, institutional buying disclosures, mining difficulty changes, \
important legislative progress, corporate treasury decisions, major technical upgrades shipping
- 1 (LOW): Analysis, minor adoption news, conference coverage, startup funding, \
minor regulatory clarifications, developer updates
- 0 (NONE): History, education, opinion, conference schedules, book/podcast announcements

JOB 4 — TICKER: Use BTC for market impact 2 or 3. Use NONE for 0 or 1.

For each numbered cluster, respond with EXACTLY this format:
1. BTC-CATEGORY | REWRITTEN HEADLINE | MARKET_SCORE | TICKER

Examples:
1. BTC-MARKETS | BLACKROCK IBIT SEES RECORD $2.1B SINGLE-DAY INFLOW: BLOOMBERG | 3 | BTC
2. BTC-POLICY | SENATE BILL WOULD MAKE BITCOIN US STRATEGIC RESERVE ASSET | 3 | BTC
3. BTC-TECH | LIGHTNING NETWORK CAPACITY HITS ALL-TIME HIGH OF 5,200 BTC | 1 | NONE
4. BTC-MINING | MARATHON DIGITAL MINES 706 BITCOIN IN JANUARY, RECORD MONTH | 1 | NONE
5. BTC-MACRO | BITCOIN SURGES 4% AS FED SIGNALS RATE PAUSE, DOLLAR INDEX FALLS | 2 | BTC
6. BTC-CULTURE | BITCOIN 2025 NASHVILLE DRAWS RECORD 35,000 ATTENDEES | 0 | NONE"""

VALID_BTC_REWRITE_CATEGORIES = {
    "btc-markets": "bitcoin-markets",
    "btc-policy":  "bitcoin-policy",
    "btc-tech":    "bitcoin-tech",
    "btc-mining":  "bitcoin-mining",
    "btc-macro":   "bitcoin-macro",
    "btc-culture": "bitcoin-culture",
}


def _parse_bitcoin_rewrite_response(text: str, clusters: list, conn) -> tuple[int, int]:
    """Parse Bitcoin rewriter response. Returns (rewrites, recats)."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    rewrites = 0
    recats = 0
    line_idx = 0

    for i, cluster in enumerate(clusters):
        if line_idx >= len(lines):
            break

        line = lines[line_idx]
        line_idx += 1

        for prefix in [f"{i+1}.", f"{i+1})"]:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()

        category = None
        rewritten = line
        market_mover = 0
        market_ticker = None

        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            cat_raw = parts[0].lower()
            if cat_raw in VALID_BTC_REWRITE_CATEGORIES:
                category = VALID_BTC_REWRITE_CATEGORIES[cat_raw]
                rewritten = parts[1].strip() if len(parts) > 1 else rewritten
                if len(parts) >= 3:
                    try:
                        score = int(parts[2].strip())
                        if 0 <= score <= 3:
                            market_mover = score
                    except (ValueError, IndexError):
                        pass
                if len(parts) >= 4:
                    ticker = parts[3].strip().upper()
                    if ticker and ticker != "NONE" and ticker.isalpha() and len(ticker) <= 5:
                        market_ticker = ticker

        if rewritten and len(rewritten) <= 120:
            ev("rewrite", before=cluster["current_headline"], after=rewritten,
               category=category or "unchanged", market_mover=market_mover)
            conn.execute(
                "UPDATE story_clusters SET rewritten_headline=? WHERE id=?",
                (rewritten, cluster["id"])
            )
            rewrites += 1

        if category:
            conn.execute(
                "UPDATE story_clusters SET category=? WHERE id=?",
                (category, cluster["id"])
            )
            recats += 1

        try:
            conn.execute("""
                INSERT INTO curation_overrides (cluster_id, market_mover, market_ticker, updated_at)
                VALUES (?, ?, ?, datetime('now'))
                ON CONFLICT(cluster_id) DO UPDATE SET
                    market_mover=?, market_ticker=?, updated_at=datetime('now')
            """, (cluster["id"], market_mover, market_ticker, market_mover, market_ticker))
        except Exception as e:
            log.warning(f"Could not write market_mover for cluster {cluster['id']}: {e}")

    return rewrites, recats


async def rewrite_bitcoin_pending():
    """Rewrite pending Bitcoin clusters using the Bitcoin-specific AI prompt."""
    import anthropic
    from wire.rewrite import _build_rewrite_prompt, _BATCH_CHUNK_SIZE, _PARALLEL_REQUESTS, _rewrite_chunk

    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

    # Fetch Bitcoin clusters pending rewrite (source_count >= 1 to surface early)
    rows = conn.execute("""
        SELECT sc.id, sc.rewritten_headline, sc.source_count
        FROM story_clusters sc
        LEFT JOIN curation_overrides co ON co.cluster_id = sc.id
        WHERE sc.expires_at > ?
          AND sc.last_updated > ?
          AND sc.category LIKE 'bitcoin-%'
          AND sc.source_count >= 1
          AND (co.locked IS NULL OR co.locked = 0)
          AND EXISTS (
              SELECT 1 FROM raw_items ri
              WHERE ri.cluster_id = sc.id
                AND ri.original_headline = sc.rewritten_headline
          )
        ORDER BY sc.source_count DESC, sc.last_updated DESC
        LIMIT 60
    """, (now, cutoff_24h)).fetchall()

    if not rows:
        conn.close()
        return

    cluster_ids = [r["id"] for r in rows]
    ph = ",".join("?" for _ in cluster_ids)
    all_items = conn.execute(
        f"""SELECT cluster_id, original_headline, source_name
            FROM raw_items WHERE cluster_id IN ({ph})
            ORDER BY published_at DESC""",
        cluster_ids
    ).fetchall()
    conn.close()

    items_by_cluster: dict[str, list] = {}
    for item in all_items:
        items_by_cluster.setdefault(item["cluster_id"], []).append(item)

    clusters = []
    for r in rows:
        items = items_by_cluster.get(r["id"], [])
        if items:
            clusters.append({
                "id": r["id"],
                "current_headline": r["rewritten_headline"],
                "headlines": [(i["original_headline"], i["source_name"]) for i in items],
            })

    if not clusters:
        return

    cfg = load_config()
    model = cfg.get("ai", {}).get("model", "claude-haiku-4-5-20251001")
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    chunks = [clusters[i:i + _BATCH_CHUNK_SIZE] for i in range(0, len(clusters), _BATCH_CHUNK_SIZE)]
    total_rewrites = 0
    total_recats = 0

    for i in range(0, len(chunks), _PARALLEL_REQUESTS):
        batch = chunks[i:i + _PARALLEL_REQUESTS]
        tasks = [_rewrite_bitcoin_chunk(client, model, chunk) for chunk in batch]
        results = await asyncio.gather(*tasks)

        conn = get_conn()
        for chunk_clusters, response_text in results:
            if response_text:
                r, rc = _parse_bitcoin_rewrite_response(response_text, chunk_clusters, conn)
                total_rewrites += r
                total_recats += rc
        conn.commit()
        conn.close()

    if total_rewrites:
        log.info(f"Bitcoin rewrite: {total_rewrites} rewrites, {total_recats} recats")


async def _rewrite_bitcoin_chunk(client, model: str, clusters: list) -> tuple:
    """Send one chunk to the Bitcoin-specific rewriter prompt."""
    import anthropic as _anthropic
    from wire.rewrite import _build_rewrite_prompt, _MAX_RETRIES, _RETRY_BACKOFF, _API_TIMEOUT

    prompt = _build_rewrite_prompt(clusters)
    last_err = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.messages.create(
                        model=model,
                        max_tokens=4000,
                        system=[{
                            "type": "text",
                            "text": BITCOIN_SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }],
                        messages=[{"role": "user", "content": prompt}],
                        timeout=_API_TIMEOUT,
                    )
                ),
                timeout=_API_TIMEOUT + 10,
            )
            return clusters, resp.content[0].text.strip()
        except _anthropic.RateLimitError as e:
            last_err = e
            wait = _RETRY_BACKOFF * (2 ** attempt)
            log.warning(f"Bitcoin rewrite rate limited, waiting {wait}s (attempt {attempt + 1})")
            await asyncio.sleep(wait)
        except Exception as e:
            last_err = e
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_BACKOFF * (2 ** attempt))

    log.error(f"Bitcoin rewrite chunk failed: {last_err}")
    return clusters, None


# ── Story API ─────────────────────────────────────────────────────────────

def _btc_reputable_sql() -> str:
    names = ", ".join(f"'{n}'" for n in BTC_REPUTABLE_SOURCES)
    return f"({names})"


async def get_bitcoin_stories(
    request: Request,
    sort: str = "hot",
    category: str = "all",
    since: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
):
    """Serve Bitcoin wire stories. Mirrors get_stories() but scoped to bitcoin-* categories."""
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    algo = get_active_version()

    where = ["sc.expires_at > ?"]
    params: list = [now]

    where.append("(co.hidden IS NULL OR co.hidden = 0)")

    if category == "all":
        where.append("sc.category LIKE 'bitcoin-%'")
    else:
        btc_cat = f"bitcoin-{category}"
        where.append("COALESCE(co.category_override, sc.category) = ?")
        params.append(btc_cat)

    if since:
        where.append("sc.last_updated > ?")
        params.append(since)

    boost_hours = algo.get("scoop_boost_hours", 4)
    sort_prefix = (
        f"CASE WHEN COALESCE(co.breaking, 0) = 1 THEN 0"
        f"     WHEN COALESCE(co.pinned, 0) = 1 THEN 1"
        f"     WHEN COALESCE(co.boost, 0) > 0 AND (co.scoop_boosted_at IS NULL"
        f"          OR co.scoop_boosted_at > datetime('now', '-{boost_hours} hours')) THEN 2"
        f"     WHEN COALESCE(co.boost, 0) < 0 THEN 4"
        f"     ELSE 3 END"
    )
    hot_score = build_hot_score_sql(algo)
    market_mover_case = build_market_mover_case_sql(algo)
    boosted_hot_score = f"({hot_score} * ({market_mover_case}))"
    reputable_sql = _btc_reputable_sql()

    if sort == "hot":
        # Min 1 reputable source for Bitcoin (vs 3 for main wire — Bitcoin breaks in native pubs)
        where.append(f"""(
            (SELECT COUNT(DISTINCT cs2.source_name)
             FROM cluster_sources cs2
             WHERE cs2.cluster_id = sc.id
               AND cs2.source_name IN {reputable_sql}) >= 1
            OR (COALESCE(co.boost, 0) > 0 AND co.scoop_boosted_at IS NOT NULL)
        )""")
        order = f"{sort_prefix}, {boosted_hot_score} DESC, sc.published_at DESC"
    else:
        order = "sc.published_at DESC"

    where_clause = " AND ".join(where)

    rows = conn.execute(f"""
        SELECT sc.id,
               COALESCE(co.headline_override, sc.rewritten_headline) as rewritten_headline,
               sc.primary_url, sc.primary_source,
               COALESCE(co.category_override, sc.category) as category,
               sc.source_count, sc.first_seen, sc.last_updated, sc.published_at,
               COALESCE(co.breaking, 0) as breaking,
               co.scoop_boosted_at,
               COALESCE(co.market_mover, 0) as market_mover,
               co.market_ticker,
               sc.group_id,
               cg.label as group_label,
               (SELECT ri2.original_headline FROM raw_items ri2
                WHERE ri2.cluster_id = sc.id LIMIT 1) as original_headline
        FROM story_clusters sc
        LEFT JOIN curation_overrides co ON co.cluster_id = sc.id
        LEFT JOIN cluster_groups cg ON cg.id = sc.group_id
        WHERE {where_clause}
        ORDER BY {order}
        LIMIT ? OFFSET ?
    """, (*params, limit, offset)).fetchall()

    cluster_ids = [r["id"] for r in rows]
    sources_by_cluster: dict = {}
    items_by_cluster: dict = {}

    if cluster_ids:
        ph = ",".join("?" for _ in cluster_ids)
        for s in conn.execute(
            f"SELECT cluster_id, source_name, MIN(source_url) as source_url"
            f" FROM cluster_sources WHERE cluster_id IN ({ph})"
            f" GROUP BY cluster_id, source_name",
            cluster_ids
        ).fetchall():
            sources_by_cluster.setdefault(s["cluster_id"], []).append(s)

        for i in conn.execute(
            f"SELECT cluster_id, original_headline, source_url, source_name, published_at"
            f" FROM raw_items WHERE cluster_id IN ({ph})"
            f" ORDER BY published_at DESC",
            cluster_ids
        ).fetchall():
            items_by_cluster.setdefault(i["cluster_id"], []).append(i)

    conn.close()

    stories = []
    for r in rows:
        cid = r["id"]
        # Strip "bitcoin-" prefix for front-end category display
        display_cat = (r["category"] or "").replace("bitcoin-", "")
        other_sources = [
            {"name": s["source_name"], "url": s["source_url"]}
            for s in sources_by_cluster.get(cid, [])
            if s["source_url"] != r["primary_url"]
        ]
        items = [
            {"headline": i["original_headline"], "url": i["source_url"],
             "source": i["source_name"], "published_at": i["published_at"]}
            for i in items_by_cluster.get(cid, [])
        ]
        stories.append({
            "id":               cid,
            "headline":         r["rewritten_headline"],
            "original_headline": r["original_headline"],
            "url":              r["primary_url"],
            "source":           r["primary_source"],
            "category":         display_cat,
            "source_count":     r["source_count"],
            "first_seen":       r["first_seen"],
            "published_at":     r["published_at"],
            "last_updated":     r["last_updated"],
            "breaking":         bool(r["breaking"]),
            "scoop":            bool(r["scoop_boosted_at"]),
            "market_mover":     r["market_mover"],
            "market_ticker":    r["market_ticker"],
            "group_id":         r["group_id"],
            "group_label":      r["group_label"],
            "other_sources":    other_sources,
            "items":            items,
        })

    return JSONResponse({
        "stories":    stories,
        "total":      len(stories),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
