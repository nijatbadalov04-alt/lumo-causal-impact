"""Canonical project paths — single source of truth for the directory layout."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA = PROJECT_ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

CONFIGS = PROJECT_ROOT / "configs"

RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
METRICS = RESULTS / "metrics"
CHECKPOINTS = RESULTS / "checkpoints"

LOGS = PROJECT_ROOT / "logs"


def ensure_dirs() -> None:
    """Create all standard output directories if they do not yet exist."""
    for d in (RAW, INTERIM, PROCESSED, FIGURES, TABLES, METRICS, CHECKPOINTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
