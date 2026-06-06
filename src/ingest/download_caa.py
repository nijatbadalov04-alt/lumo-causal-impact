"""
Download CAA Table 12.2 (Domestic Air Passenger Traffic Route Analysis) annual CSVs.

THINK -> RESEARCH -> CODE
  WHAT: the UK Civil Aviation Authority publishes, per year, a domestic route-analysis table
        giving total passengers on every UK domestic airport-pair (both directions). Each
        annual file carries this_period (the year) AND last_period (the prior year), so two
        files span four years. We grab 2019 (-> 2018, 2019, the clean pre-COVID baseline) and
        2024 (-> 2023, 2024, the recovered post-Lumo period) -- matching the ODM pre/post.
  WHY:  the single most valuable missing dataset (CRITIQUE E) -- it decomposes the ODM
        "creation" into induced demand vs air->rail modal shift, the policy/climate punchline.
  SOURCE: caa.co.uk UK airport data, annual airport statistics, Table 12.2 (Open Government
        Licence). Document-download URLs carry a per-year GUID (hard-coded below from the
        published annual pages).

Run:  python -m src.ingest.download_caa
"""

from __future__ import annotations

import urllib.request

from src.utils.logging_setup import get_logger
from src.utils.paths import RAW, ensure_dirs

LOG = get_logger("ingest.download_caa", log_file="logs/ingest.log")

BASE = "https://www.caa.co.uk"
# Table 12.2 Domestic Air Passenger Traffic Route Analysis CSV, per annual page (GUID per year)
CAA_FILES = {
    2019: "/Documents/Download/3951/e925ed1f-e4b5-4d12-ad1c-e95e0b5b3307/2298",  # -> 2018, 2019
    2024: "/Documents/Download/11911/0af1d44e-1648-4fd7-94e3-0b9697934148/17025",  # -> 2023, 2024
}


def main() -> None:
    ensure_dirs()
    out_dir = RAW / "air"
    out_dir.mkdir(parents=True, exist_ok=True)
    for year, path in CAA_FILES.items():
        dest = out_dir / f"caa_domestic_{year}.csv"
        if dest.exists() and dest.stat().st_size > 1000:
            LOG.info("already have %s (%d bytes) — skipping", dest.name, dest.stat().st_size)
            continue
        url = BASE + path
        LOG.info("downloading CAA domestic route analysis %d ...", year)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; OGL data)"})
            with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - fixed https CAA URL
                dest.write_bytes(r.read())
            LOG.info("  -> %s (%d bytes)", dest.name, dest.stat().st_size)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("  FAILED %d (%s). Get manually from caa.co.uk airport data, Table 12.2.", year, exc)


if __name__ == "__main__":
    main()
