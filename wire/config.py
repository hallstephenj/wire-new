import yaml
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent

def load_config():
    with open(_BASE / "config.yaml") as f:
        return yaml.safe_load(f)

def load_feeds():
    with open(_BASE / "feeds.yaml") as f:
        return yaml.safe_load(f)
