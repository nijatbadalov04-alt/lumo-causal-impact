"""
Download ORR 'Estimates of Station Usage' source files into data/raw/.

THINK -> RESEARCH -> CODE
  WHAT: Fetch the station-usage spine (Table 1415 time series), the latest
        annual snapshot (Table 1410, ODS + CSV), per-year historical snapshots,
        and the methodology/quality + statistical-release PDFs.
  WHY : Table 1415 is the per-station annual time series that anchors every
        counterfactual (pre/post Lumo 2021, Hull Trains 2000, Grand Central
        2007/2010). The 1410 snapshots give an independent QA cross-check. The
        methodology PDF is required reading (ticket->journey conversion, breaks).
  ALT : Could re-scrape the portal HTML for links on every run. Rejected: the
        /media/ URLs are stable and now pinned in configs/data.yaml, which is
        more reproducible and avoids silent URL drift. (If ORR rotates a URL,
        the failure is loud here and we re-discover + update the config.)

Idempotent: skips a file already present at the expected path unless --force.
Writes data/raw/_manifest.json (url, bytes, sha256, http status, access time)
for full provenance. URLs verified 2026-06-04 (see SOURCES.md).

Run from project root:
    python -m src.ingest.download_orr            # download missing files
    python -m src.ingest.download_orr --force    # re-download everything
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import RAW, ensure_dirs

LOG = get_logger("ingest.download_orr", log_file="logs/ingest.log")

_CHUNK = 1 << 16  # 64 KiB streaming chunk
_HEADERS = {"User-Agent": "uk-rail-openaccess-research/1.0 (academic; contact via repo)"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(url: str, dest: Path, force: bool = False, timeout: int = 120) -> dict:
    """Stream a single URL to dest atomically; return a manifest record."""
    rec: dict = {"url": url, "filename": dest.name}
    if dest.exists() and not force:
        size = dest.stat().st_size
        rec.update(status="skipped_exists", bytes=size, sha256=sha256_of(dest))
        LOG.info("skip (exists): %s (%s bytes)", dest.name, f"{size:,}")
        return rec

    LOG.info("GET %s", url)
    with requests.get(url, stream=True, timeout=timeout, headers=_HEADERS) as resp:
        rec["http_status"] = resp.status_code
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        nbytes = 0
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(_CHUNK):
                if chunk:
                    fh.write(chunk)
                    nbytes += len(chunk)
        tmp.replace(dest)  # atomic on same filesystem

    rec.update(
        status="downloaded",
        bytes=nbytes,
        sha256=sha256_of(dest),
        access_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    LOG.info("saved %s (%s bytes, sha256=%s...)", dest.name, f"{nbytes:,}", rec["sha256"][:12])
    return rec


def iter_items(cfg: dict):
    """Yield (absolute_url, filename) for every file declared in the config."""
    base = cfg["orr_base_url"].rstrip("/")
    for group in ("station_usage", "documents"):
        for entry in cfg.get(group, {}).values():
            yield base + entry["url"], entry["filename"]
    for entry in cfg.get("historical_snapshots", []):
        yield base + entry["url"], entry["filename"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ORR station-usage source files.")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--config", default="data.yaml", help="config name under configs/")
    args = parser.parse_args()

    ensure_dirs()
    cfg = load_config(args.config)

    manifest: list[dict] = []
    for url, fname in iter_items(cfg):
        try:
            manifest.append(download_one(url, RAW / fname, force=args.force))
        except Exception as exc:  # noqa: BLE001 — log & continue (do not halt the pipeline on one failure)
            LOG.error("FAILED %s: %s", url, exc)
            manifest.append({"url": url, "filename": fname, "status": "error", "error": str(exc)})

    manifest_path = RAW / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ok = sum(1 for m in manifest if m.get("status") in ("downloaded", "skipped_exists"))
    LOG.info("manifest -> %s  (%d/%d files OK)", manifest_path, ok, len(manifest))
    if ok < len(manifest):
        LOG.warning("%d file(s) failed — see manifest and re-run.", len(manifest) - ok)


if __name__ == "__main__":
    main()
