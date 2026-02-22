import time
import yaml
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent
_runtime_overrides = {}

def set_override(key_path, value):
    """Set a runtime config override using dot-notation key path.
    e.g. set_override('clustering.similarity_threshold', 0.25)
    """
    _runtime_overrides[key_path] = value
    # Invalidate config cache so next load picks up the override
    _config_cache["data"] = None

def get_overrides():
    """Return a copy of current runtime overrides."""
    return dict(_runtime_overrides)

def clear_overrides():
    """Clear all runtime overrides."""
    _runtime_overrides.clear()
    _config_cache["data"] = None

def _apply_overrides(cfg):
    """Merge runtime overrides into config dict."""
    for key_path, value in _runtime_overrides.items():
        keys = key_path.split(".")
        d = cfg
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    return cfg

# ── Cached config loading ─────────────────────────────────────────────
_config_cache = {"data": None, "loaded_at": 0}
_CONFIG_TTL = 30  # seconds

def load_config():
    now = time.monotonic()
    if _config_cache["data"] is not None and (now - _config_cache["loaded_at"]) < _CONFIG_TTL:
        return _config_cache["data"]
    with open(_BASE / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    result = _apply_overrides(cfg)
    _config_cache["data"] = result
    _config_cache["loaded_at"] = now
    return result

# ── Cached feeds loading (rarely changes) ─────────────────────────────
_feeds_cache = None

def load_feeds():
    global _feeds_cache
    if _feeds_cache is not None:
        return _feeds_cache
    with open(_BASE / "feeds.yaml") as f:
        _feeds_cache = yaml.safe_load(f)
    return _feeds_cache
