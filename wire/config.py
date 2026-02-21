import yaml
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent
_runtime_overrides = {}

def set_override(key_path, value):
    """Set a runtime config override using dot-notation key path.
    e.g. set_override('clustering.similarity_threshold', 0.25)
    """
    _runtime_overrides[key_path] = value

def get_overrides():
    """Return a copy of current runtime overrides."""
    return dict(_runtime_overrides)

def clear_overrides():
    """Clear all runtime overrides."""
    _runtime_overrides.clear()

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

def load_config():
    with open(_BASE / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    return _apply_overrides(cfg)

def load_feeds():
    with open(_BASE / "feeds.yaml") as f:
        return yaml.safe_load(f)
