"""
Publication quality scores for selecting primary cluster links.

Sources are grouped into tiers for reliability and reputation,
with individual access (paywall) scores. Composite score (0-100,
higher = better) determines which source becomes primary.

Composite = reliability * 0.4 + access * 0.3 + reputation * 0.3
"""

# ── Tier definitions: (reliability, reputation) ──────────────────────────

TIER_1 = (95, 95)  # Wire services
TIER_2 = (90, 90)  # Major broadsheets / broadcasters
TIER_3 = (85, 80)  # Strong specialist / investigative
TIER_4 = (75, 70)  # Solid trade / beat press
TIER_5 = (65, 55)  # Niche / enthusiast / partisan
TIER_6 = (50, 40)  # Aggregators / misc
TIER_DEFAULT = (45, 35)

_TIER_MAP = {
    # Tier 1
    "Reuters": TIER_1,
    "Reuters Business": TIER_1,
    "Reuters Politics": TIER_1,
    "Reuters World": TIER_1,
    "AP": TIER_1,
    "AP Politics": TIER_1,
    "AP Top News": TIER_1,

    # Tier 2
    "New York Times": TIER_2,
    "Wall Street Journal": TIER_2,
    "Washington Post": TIER_2,
    "BBC": TIER_2,
    "BBC World": TIER_2,
    "Bloomberg": TIER_2,
    "Financial Times": TIER_2,
    "The Guardian": TIER_2,
    "CNN": TIER_2,
    "NBC News": TIER_2,
    "NPR": TIER_2,
    "Al Jazeera": TIER_2,

    # Tier 3
    "Ars Technica": TIER_3,
    "ProPublica": TIER_3,
    "MIT Technology Review": TIER_3,
    "Nature": TIER_3,
    "Politico": TIER_3,
    "Wired": TIER_3,
    "404 Media": TIER_3,
    "Krebs on Security": TIER_3,
    "BleepingComputer": TIER_3,
    "Pew Research": TIER_3,
    "The Atlantic": TIER_3,
    "The Record": TIER_3,

    # Tier 4
    "TechCrunch": TIER_4,
    "The Verge": TIER_4,
    "Axios": TIER_4,
    "CNBC": TIER_4,
    "Fortune": TIER_4,
    "Forbes": TIER_4,
    "Fox Business": TIER_4,
    "Business Insider": TIER_4,
    "The Hill": TIER_4,
    "Semafor": TIER_4,
    "South China Morning Post": TIER_4,
    "Nikkei Asia": TIER_4,
    "Tom's Hardware": TIER_4,
    "MacRumors": TIER_4,
    "CoinDesk": TIER_4,
    "iFixit": TIER_4,
    "The Register": TIER_4,
    "Investopedia": TIER_4,
    "Simon Willison": TIER_4,

    # Tier 5
    "9to5Google": TIER_5,
    "Android Authority": TIER_5,
    "Windows Central": TIER_5,
    "The Block": TIER_5,
    "Bitcoin Magazine": TIER_5,
    "Fox News": TIER_5,
    "USA Today": TIER_5,
    "MarketWatch": TIER_5,
    "Yahoo Finance": TIER_5,
    "Nextgov": TIER_5,
    "Seeking Alpha": TIER_5,
    "Motley Fool": TIER_5,
    "Sky News": TIER_5,
    "Barron's": TIER_5,
    "Crazy Stupid Tech": TIER_5,
    "Sources": TIER_5,

    # Tier 6
    "MSN": TIER_6,
    "AOL": TIER_6,
    "Social Media Today": TIER_6,
    "Google News": TIER_6,
}

# ── Access (paywall) scores ──────────────────────────────────────────────

_METERED = 60   # soft paywall / metered
_PAYWALL = 30   # hard paywall
_FREE = 100

_ACCESS_MAP = {
    # Hard paywall
    "Wall Street Journal": _PAYWALL,
    "Financial Times": _PAYWALL,
    "Bloomberg": _PAYWALL,
    "The Information": _PAYWALL,
    "Barron's": _PAYWALL,
    "Nikkei Asia": _PAYWALL,

    # Metered / soft paywall
    "New York Times": _METERED,
    "Washington Post": _METERED,
    "Wired": _METERED,
    "The Atlantic": _METERED,
    "Fortune": _METERED,
    "Forbes": _METERED,
    "Business Insider": _METERED,
    "South China Morning Post": _METERED,
    "MIT Technology Review": _METERED,
}
# Everything else defaults to _FREE (100)


def _compute(reliability: int, access: int, reputation: int) -> int:
    return round(reliability * 0.4 + access * 0.3 + reputation * 0.3)


def get_score(name: str) -> dict:
    """Return full score breakdown for a source."""
    reliability, reputation = _TIER_MAP.get(name, TIER_DEFAULT)
    access = _ACCESS_MAP.get(name, _FREE)
    total = _compute(reliability, access, reputation)
    return {
        "total": total,
        "reliability": reliability,
        "access": access,
        "reputation": reputation,
    }


def get_total_score(name: str) -> int:
    """Return composite score (0-100, higher = better)."""
    reliability, reputation = _TIER_MAP.get(name, TIER_DEFAULT)
    access = _ACCESS_MAP.get(name, _FREE)
    return _compute(reliability, access, reputation)


def all_scores() -> dict:
    """Return scores for all known sources, sorted by total descending."""
    names = set(_TIER_MAP.keys()) | set(_ACCESS_MAP.keys())
    scores = {name: get_score(name) for name in names}
    return dict(sorted(scores.items(), key=lambda x: (-x[1]["total"], x[0])))
