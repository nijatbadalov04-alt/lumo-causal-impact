"""
Parse ORR Table 1415 (time series of station usage) ODS -> tidy long Parquet.

THINK -> RESEARCH -> CODE
  WHAT: Turn the wide, multi-junk-sheet ODS into one tidy long table
        (crs, year, metric, value, flag) written as typed Parquet.
  WHY : Long form is the canonical shape for panel econometrics + the
        counterfactual models; Parquet is fast/typed/compressed (§5).
  HOW : Read ONLY the two real sheets (1415a entries+exits, 1415b interchanges);
        header is on row index 3, data from row 4; melt 28 financial-year columns
        (1997-98 … 2024-25) to long. Map ORR sentinels to NaN but KEEP a `flag`
        column so we never conflate a TRUE ZERO, a MISSING value, and a
        STRUCTURALLY-ABSENT (not-applicable) station-year (§5, Sarah Chen).
  SENTINELS (from the sheet's own shorthand note, verified 2026-06-04):
        [x] = data not available   -> flag 'not_available'
        [z] = data not applicable  -> flag 'not_applicable' (e.g. station absent)
        [b] = break in time series -> annotated on the YEAR HEADER (e.g. 2022-23);
              captured as a boolean `header_break` per (metric, year).
  NOTE: values are non-negative REALS (CAPRI-era decimals exist) -> float64.

Run:  python -m src.clean.parse_station_usage
Out:  data/interim/station_usage_long.parquet
      data/interim/station_meta.parquet
      results/metrics/qa_station_usage.json
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger
from src.utils.paths import INTERIM, METRICS, RAW, ensure_dirs

LOG = get_logger("clean.parse_station_usage", log_file="logs/clean.log")

ODS_FILE = RAW / "table-1415-time-series-station-usage.ods"
SHEETS = {
    "entries_exits": "1415a_Entries_and_Exits",
    "interchanges": "1415b_Interchanges",
}
HEADER_ROW = 3  # 0-indexed row holding the real column names
ID_RENAME = {
    "Sort": "sort",
    "Station name": "station_name",
    "Three Letter Code(TLC)": "crs",
    "National Location Code (NLC)": "nlc",
    "Region": "region",
    "Local authority: district or unitary": "la_district",
}
ID_COLS = ["sort", "station_name", "crs", "nlc", "region", "la_district"]
_YEAR_RE = re.compile(r"Apr\s+(\d{4})\s+to\s+Mar\s+(\d{4})\s*(?:\[(\w)\])?")


def _parse_year_header(h: str) -> tuple[int, str, bool] | None:
    """'Apr 1997 to Mar 1998 [b]' -> (1997, '1997-98', break=True)."""
    m = _YEAR_RE.search(str(h))
    if not m:
        return None
    start = int(m.group(1))
    label = f"{start}-{str(int(m.group(2)))[2:]}"
    is_break = (m.group(3) or "").lower() == "b"
    return start, label, is_break


def parse_values(raw: pd.Series) -> tuple[pd.Series, np.ndarray]:
    """Map a raw value column to (float value, categorical flag).

    Numeric -> 'observed' (including a genuine 0 = TRUE ZERO, never 'missing').
    [x] -> 'not_available'  [z] -> 'not_applicable'  [c] -> 'confidential'
    anything else non-numeric -> 'unknown_missing'. Pure function (unit-tested).
    """
    val = pd.to_numeric(raw, errors="coerce")
    raw_str = raw.astype(str).str.strip()
    flag = np.select(
        [val.notna(), raw_str.eq("[x]"), raw_str.eq("[z]"), raw_str.eq("[c]")],
        ["observed", "not_available", "not_applicable", "confidential"],
        default="unknown_missing",
    )
    return val.astype("float64"), flag


def parse_metric(sheet: str, metric: str) -> pd.DataFrame:
    """Read one sheet and return a tidy long frame for a single metric."""
    LOG.info("reading sheet %r (metric=%s)", sheet, metric)
    raw = pd.read_excel(ODS_FILE, sheet_name=sheet, engine="odf", header=None)
    header = raw.iloc[HEADER_ROW].tolist()
    body = raw.iloc[HEADER_ROW + 1 :].copy()
    body.columns = header

    # Keep only rows that are real stations (numeric Sort).
    body = body[pd.to_numeric(body["Sort"], errors="coerce").notna()].copy()
    body = body.rename(columns=ID_RENAME)

    # Identify & parse year columns.
    year_map: dict[str, tuple[int, str, bool]] = {}
    for col in body.columns:
        parsed = _parse_year_header(col)
        if parsed is not None:
            year_map[col] = parsed
    if not year_map:
        raise ValueError(f"No year columns found in sheet {sheet!r}")

    long = body.melt(
        id_vars=ID_COLS,
        value_vars=list(year_map),
        var_name="year_header",
        value_name="value_raw",
    )
    long["year_start"] = long["year_header"].map(lambda c: year_map[c][0]).astype("int16")
    long["fin_year"] = long["year_header"].map(lambda c: year_map[c][1])
    long["header_break"] = long["year_header"].map(lambda c: year_map[c][2])

    # Parse value: numeric stays, sentinels -> NaN; preserve the distinction.
    long["value"], long["flag"] = parse_values(long["value_raw"])
    long["metric"] = metric

    # Tidy id dtypes
    for c in ("station_name", "crs", "nlc", "region", "la_district"):
        long[c] = long[c].astype("string")
    long = long.drop(columns=["year_header", "value_raw", "sort"])
    LOG.info("  -> %d long rows (%d stations x %d years)", len(long), body.shape[0], len(year_map))
    return long


def run_qa(long: pd.DataFrame) -> dict:
    """Data-quality gates. Fails loudly (assert) on hard violations (§5)."""
    qa: dict = {}
    # 1. Uniqueness of (crs, year, metric) among rows with a valid CRS.
    valid_crs = long[long["crs"].str.fullmatch(r"[A-Z]{3}", na=False)]
    dup = valid_crs.duplicated(subset=["crs", "year_start", "metric"]).sum()
    qa["duplicate_crs_year_metric"] = int(dup)
    assert dup == 0, f"Duplicate (crs, year, metric) keys: {dup}"

    # 2. Non-negativity of observed values.
    neg = int((long.loc[long["flag"] == "observed", "value"] < 0).sum())
    qa["negative_observed_values"] = neg
    assert neg == 0, f"Negative observed values: {neg}"

    # 3. Coverage / shape.
    qa["n_rows"] = int(len(long))
    qa["n_stations_total"] = int(long["station_name"].nunique())
    qa["n_stations_valid_crs"] = int(valid_crs["crs"].nunique())
    qa["year_min"] = int(long["year_start"].min())
    qa["year_max"] = int(long["year_start"].max())
    qa["metrics"] = sorted(long["metric"].unique().tolist())

    # 4. Flag distribution (the missing-value taxonomy).
    qa["flag_counts"] = {k: int(v) for k, v in long["flag"].value_counts().items()}
    qa["header_break_years"] = sorted(long.loc[long["header_break"], "year_start"].unique().tolist())

    # 5. The structural 2003-04 gap should be ~entirely missing.
    g = long[long["year_start"] == 2003]
    qa["frac_observed_2003_04"] = round(float((g["flag"] == "observed").mean()), 4)

    assert qa["n_stations_valid_crs"] > 2000, "Too few valid-CRS stations — parse error?"
    assert qa["year_min"] == 1997 and qa["year_max"] == 2024, "Unexpected year span"
    return qa


def main() -> None:
    ensure_dirs()
    frames = [parse_metric(sheet, metric) for metric, sheet in SHEETS.items()]
    long = pd.concat(frames, ignore_index=True)

    # Categorical dtypes for compact Parquet.
    long["metric"] = long["metric"].astype("category")
    long["flag"] = long["flag"].astype("category")
    long["region"] = long["region"].astype("category")

    qa = run_qa(long)

    out = INTERIM / "station_usage_long.parquet"
    long.to_parquet(out, engine="pyarrow", index=False, compression="zstd")
    LOG.info("wrote %s (%d rows, %.2f MB)", out, len(long), out.stat().st_size / 1e6)

    meta = (
        long[long["metric"] == "entries_exits"][["crs", "station_name", "nlc", "region", "la_district"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    meta_out = INTERIM / "station_meta.parquet"
    meta.to_parquet(meta_out, engine="pyarrow", index=False)
    LOG.info("wrote %s (%d stations)", meta_out, len(meta))

    (METRICS / "qa_station_usage.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    LOG.info("QA: %s", json.dumps(qa))
    LOG.info("M1 parse complete.")


if __name__ == "__main__":
    main()
