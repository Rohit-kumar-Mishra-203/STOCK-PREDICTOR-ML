"""
Central config loader.

Why this exists:
Every module needs settings (tickers, paths, params). Instead of each file
reading config.yaml independently (duplicated, error-prone), we load it once
here and import `get_config()` everywhere. This is the single source of truth.
"""

from pathlib import Path
from functools import lru_cache
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict:
    """
    Load and cache config.yaml.

    lru_cache ensures we only read+parse the YAML file once per process,
    even if get_config() is called from 10 different modules. Cheap
    optimization, but also guarantees every module sees the same config
    object during a single run.
    """
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Turn a config-relative path (e.g. 'data/raw/prices') into an absolute Path."""
    return PROJECT_ROOT / relative_path


if __name__ == "__main__":
    # Quick manual check: run `python -m src.utils.config` to sanity-check config loads.
    cfg = get_config()
    print("Loaded config for tickers:", cfg["data"]["tickers"])