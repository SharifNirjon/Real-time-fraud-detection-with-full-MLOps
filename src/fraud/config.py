"""Configuration loading. One place for paths and YAML configs."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("FRAUD_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = Path(os.environ.get("FRAUD_CONFIG_DIR", ROOT / "configs"))


@lru_cache(maxsize=8)
def _load_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_config() -> dict[str, Any]:
    return _load_yaml(str(CONFIG_DIR / "config.yaml"))


def load_feature_config() -> dict[str, Any]:
    return _load_yaml(str(CONFIG_DIR / "features.yaml"))


def path(key: str) -> Path:
    """Resolve a configured path relative to the repo root (env override: FRAUD_<KEY>)."""
    override = os.environ.get(f"FRAUD_{key.upper()}")
    p = Path(override) if override else ROOT / load_config()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
