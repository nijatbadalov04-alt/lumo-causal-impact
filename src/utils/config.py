"""YAML config loader — keeps magic numbers/URLs out of code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import CONFIGS


def load_config(name: str | Path) -> dict[str, Any]:
    """Load a YAML config by name (looked up under configs/) or absolute path."""
    p = Path(name)
    if not p.is_absolute():
        p = CONFIGS / p
    if p.suffix not in (".yaml", ".yml"):
        p = p.with_suffix(".yaml")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
