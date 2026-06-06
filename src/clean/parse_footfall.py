"""
Parse Network Rail Daily Concourse Footfall -> tidy daily panel.

THINK -> RESEARCH -> CODE
  WHAT: Network Rail publishes daily concourse footfall (turnstile/sensor counts) for the
        ~18 managed stations (London termini, Edinburgh Waverley, Glasgow Central, Leeds,
        Birmingham, etc.). This is a REAL passenger count, unlike the modelled ORR/ODM
        station-usage (LENNON->MOIRA estimates from ticket sales).
  WHY:  Both Lumo endpoints are covered -- Edinburgh Waverley AND London Kings Cross -- plus
        Glasgow Central (Edinburgh's main non-London flow, a natural comparator). We use it
        to VALIDATE that the modelled ODM growth signal shows up in real counts (2023->2025).
  GOTCHA 1: the files ship in TWO schemas (auto-detected per file from the header):
        - BASE file only (DailyConcourseFootfall_NR.csv): ';'-delimited, CountForward/CountBackward, DD/MM/YYYY
        - every year-suffixed file (_2023.._2026): ','-delimited, IN/OUT, YYYY-MM-DD
        We normalise both to (site, date, entries, exits, total).
  GOTCHA 2: the BASE file is a partial/rolling export whose final day is TRUNCATED (e.g. King's
        Cross 2023-10-31 ~9k vs the full ~111k in the year file). The base file fully overlaps
        the year files, so on any (site,date) clash we deterministically PREFER the authoritative
        year-suffixed file (priority) rather than relying on concat order.
  SOURCE: Rail Data Marketplace, "NWR Daily Concourse Footfall" (Network Rail).

Run:  python -m src.clean.parse_footfall
"""

from __future__ import annotations

import glob
import re

import polars as pl

from src.utils.logging_setup import get_logger
from src.utils.paths import INTERIM, RAW, ensure_dirs

LOG = get_logger("clean.parse_footfall", log_file="logs/clean.log")


def _read_one(path: str) -> pl.DataFrame | None:
    """Read a single footfall CSV, auto-detecting the ';CountForward' vs ',IN' schema."""
    with open(path, encoding="utf-8-sig") as fh:
        header = fh.readline()
    semicolon = ";" in header
    sep = ";" if semicolon else ","
    df = pl.read_csv(path, separator=sep, infer_schema_length=2000, encoding="utf8-lossy")
    df = df.rename({c: c.lstrip("﻿").strip() for c in df.columns})

    if "CountForward" in df.columns:  # base / 2023 schema
        df = df.rename({"CountForward": "entries", "CountBackward": "exits"})
        date_fmt = "%d/%m/%Y"
    else:  # 2024+ schema
        df = df.rename({"IN": "entries", "OUT": "exits"})
        date_fmt = "%Y-%m-%d"

    df = df.with_columns(
        pl.col("SiteName").str.strip_chars().alias("site"),
        pl.col("DateFrom").str.strip_chars().str.to_date(date_fmt, strict=False).alias("date"),
        pl.col("entries").cast(pl.Int64, strict=False),
        pl.col("exits").cast(pl.Int64, strict=False),
    )
    return df.select("site", "date", "entries", "exits").drop_nulls("date")


def main() -> None:
    ensure_dirs()
    files = sorted(glob.glob(str(RAW / "footfall" / "*.csv")))
    if not files:
        LOG.warning("no footfall files in data/raw/footfall/ — skipping (user-provided via RDM).")
        return

    frames = []
    for f in files:
        name = f.split("/")[-1].split("\\")[-1]
        try:
            d = _read_one(f)
        except Exception as exc:  # noqa: BLE001 - log and continue on a bad file
            LOG.warning("  skip %s (%s)", name, exc)
            continue
        if d is not None and d.height:
            # year-suffixed files are authoritative; the base file is a partial/rolling export
            priority = 1 if re.search(r"_20\d{2}", name) else 0
            frames.append(d.with_columns(pl.lit(priority).alias("priority")))
            LOG.info("  %-38s %6d rows  %s..%s (prio %d)", name, d.height, d["date"].min(), d["date"].max(), priority)

    panel = (
        pl.concat(frames)
        .sort(["site", "date", "priority"])  # higher priority (year file) sorts last...
        .unique(subset=["site", "date"], keep="last")  # ...so keep="last" deterministically wins
        .drop("priority")
        .with_columns((pl.col("entries").fill_null(0) + pl.col("exits").fill_null(0)).alias("total"))
        .sort(["site", "date"])
    )
    panel.write_parquet(INTERIM / "footfall_daily.parquet")
    LOG.info(
        "wrote %d daily rows across %d sites (%s..%s) -> data/interim/footfall_daily.parquet",
        panel.height,
        panel["site"].n_unique(),
        panel["date"].min(),
        panel["date"].max(),
    )


if __name__ == "__main__":
    main()
