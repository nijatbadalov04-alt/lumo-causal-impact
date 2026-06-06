"""
Download supplementary ORR data: operator/sector journeys + confounders.

Keeps `python run_pipeline.py` fully reproducible — these feed the operator-level
analysis (Table 1223/1221) and the confounder controls (Table 3113 punctuality,
Table 7180 fares). Reuses the provenance-tracked downloader from download_orr.
URLs verified 2026-06-04/05 (see SOURCES.md). Idempotent (skips existing).

Run:  python -m src.ingest.download_supplementary
"""

from __future__ import annotations

import json

from src.ingest.download_orr import download_one
from src.utils.logging_setup import get_logger
from src.utils.paths import RAW, ensure_dirs

LOG = get_logger("ingest.download_supplementary", log_file="logs/ingest.log")

BASE = "https://dataportal.orr.gov.uk"
FILES = [
    (
        f"{BASE}/media/1476/table-1223-passenger-journeys-by-operator.ods",
        "operator/table-1223-journeys-by-operator.ods",
    ),
    (f"{BASE}/media/2011/table-1221-passenger-journeys-by-sector.ods", "operator/table-1221-journeys-by-sector.ods"),
    (
        f"{BASE}/media/1428/table-3113-public-performance-measure-by-operator-and-sector.ods",
        "confounders/table-3113-ppm-by-operator.ods",
    ),
    (
        f"{BASE}/media/1692/table-7180-average-change-in-fares-by-regulated-and-unregulated-tickets.ods",
        "confounders/table-7180-fares-change.ods",
    ),
]


def main() -> None:
    ensure_dirs()
    (RAW / "operator").mkdir(parents=True, exist_ok=True)
    (RAW / "confounders").mkdir(parents=True, exist_ok=True)
    manifest = []
    for url, rel in FILES:
        dest = RAW / rel
        try:
            manifest.append(download_one(url, dest))
        except Exception as exc:  # noqa: BLE001 — log & continue (don't halt the pipeline)
            LOG.error("FAILED %s: %s", url, exc)
            manifest.append({"url": url, "filename": rel, "status": "error", "error": str(exc)})
    (RAW / "_manifest_supplementary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ok = sum(1 for m in manifest if m.get("status") in ("downloaded", "skipped_exists"))
    LOG.info("supplementary downloads: %d/%d OK", ok, len(manifest))


if __name__ == "__main__":
    main()
