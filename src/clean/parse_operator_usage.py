"""
Parse ORR operator/sector passenger-journey tables (1223, 1221) -> long Parquet.

THINK -> RESEARCH -> CODE
  WHAT: ORR Table 1223 (journeys by operator) and 1221 (by sector) each stack an
        ANNUAL block (sub-table 'a') above a QUARTERLY block ('b') in one sheet.
        Parse both blocks to a tidy long frame: (series, period_start, freq, value,
        flag). 1223 has LNER (col11) and Lumo (col24) separately — the key series.
  WHY : LNER ≈ the East Coast franchise; Lumo is ECML-only. So LNER + Lumo ≈ the
        total ECML long-distance market. This is the operator-level lever for RQ1
        (substitution vs creation) that station totals cannot give (WEAKNESSES W4).
  PERIOD -> decimal year (for a continuous time axis):
        'Apr YYYY to Mar YYYY+1' -> YYYY (FY start);  quarter midpoints:
        Apr-Jun=YYYY.375, Jul-Sep=.625, Oct-Dec=.875, Jan-Mar=YYYY.125.
  SENTINELS: [z] not-applicable (e.g. Lumo pre-2021), [c]/[x] missing, [b] break,
        [p] provisional — stripped to numeric value + flag.

Run:  python -m src.clean.parse_operator_usage
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger
from src.utils.paths import INTERIM, RAW, ensure_dirs

LOG = get_logger("clean.parse_operator", log_file="logs/clean.log")

_ANNUAL = re.compile(r"Apr\s+(\d{4})\s+to\s+Mar\s+\d{4}")
_QUARTER = re.compile(r"(\w{3})\s+to\s+(\w{3})\s+(\d{4})")
_QMAP = {"Apr": 0.375, "Jul": 0.625, "Oct": 0.875, "Jan": 0.125}


def _clean_series_name(s: str) -> str:
    return re.sub(r"\(million\).*$|\[.*?\]", "", str(s)).strip()


def _period_to_decimal(label: str):
    m = _ANNUAL.search(label)
    if m:
        return float(m.group(1)), "annual"
    m = _QUARTER.search(label)
    if m:
        start_mon, _, yr = m.groups()
        frac = _QMAP.get(start_mon)
        if frac is not None:
            # Jan-Mar belongs to the FY that started the previous calendar year
            base = int(yr) - 1 if start_mon == "Jan" else int(yr)
            return base + frac, "quarterly"
    return None, None


def _parse_value(v):
    # genuine blank / NaN is NOT an observation -> 'missing' (float(np.nan) would wrongly pass)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan, "missing"
    if isinstance(v, (int, float)):
        return float(v), "observed"
    s = str(v).strip()
    sentinel = {"[z]": "not_applicable", "[x]": "not_available", "[c]": "confidential"}.get(s)
    if sentinel:
        return np.nan, sentinel
    # ORR tags provisional/break/revised numeric cells like '1234.5 [p]' / '[b]' / '[r]';
    # KEEP the number (the latest FY is routinely provisional) rather than dropping it.
    stripped = re.sub(r"\s*\[[a-z]\]", "", s).replace(",", "").strip()
    try:
        return float(stripped), "observed"
    except ValueError:
        return np.nan, "missing"


def parse_stacked(path, sheet, header_row=5):
    raw = pd.read_excel(path, sheet_name=sheet, engine="odf", header=None)
    names = [_clean_series_name(x) for x in raw.iloc[header_row].tolist()]
    out = []
    for _, row in raw.iloc[header_row + 1 :].iterrows():
        dec, freq = _period_to_decimal(str(row.iloc[0]))
        if dec is None:
            continue
        is_break = "[b]" in str(row.iloc[0])
        for j in range(1, len(names)):
            series = names[j]
            if not series or series.lower().startswith("nan"):
                continue
            val, flag = _parse_value(row.iloc[j])
            out.append(
                {
                    "series": series,
                    "period_start": dec,
                    "freq": freq,
                    "value_m": val,
                    "flag": flag,
                    "period_break": is_break,
                }
            )
    return pd.DataFrame(out)


def main() -> None:
    ensure_dirs()
    op = parse_stacked(RAW / "operator/table-1223-journeys-by-operator.ods", "1223_Journeys_by_operator")
    op.to_parquet(INTERIM / "operator_journeys.parquet", engine="pyarrow", index=False)
    LOG.info(
        "operator journeys: %d rows, %d series, freqs=%s", len(op), op["series"].nunique(), sorted(op["freq"].unique())
    )
    LOG.info("  series: %s", sorted(op["series"].unique()))

    sec = parse_stacked(RAW / "operator/table-1221-journeys-by-sector.ods", "1221_Journeys_by_sector")
    sec.to_parquet(INTERIM / "sector_journeys.parquet", engine="pyarrow", index=False)
    LOG.info("sector journeys: %d rows, series=%s", len(sec), sorted(sec["series"].unique()))

    # sanity: LNER & Lumo annual recent values
    for s in ["London North Eastern Railway", "Lumo", "Grand Central", "Hull Trains"]:
        d = op[(op.series == s) & (op.freq == "annual") & (op.flag == "observed")]
        if len(d):
            recent = d.sort_values("period_start").tail(4)
            LOG.info(
                "  %s (annual, last 4 FY): %s",
                s,
                {int(r.period_start): round(r.value_m, 2) for r in recent.itertuples()},
            )
    LOG.info("operator/sector parse complete.")


if __name__ == "__main__":
    main()
