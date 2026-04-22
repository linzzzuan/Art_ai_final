"""Configuration loader — reads config.yaml and returns the active profile."""

import yaml
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).parent.parent  # backend/
CONFIG_FILE = ROOT / "config.yaml"


def load_config() -> dict:
    """Load config.yaml and return the active profile's settings."""
    with open(CONFIG_FILE, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    profile_name = cfg["active_profile"]
    profile = cfg["profiles"][profile_name]

    return {**profile, "_profile": profile_name, "_root": str(ROOT)}


@lru_cache()
def get_settings() -> dict:
    """Cached config singleton."""
    return load_config()
