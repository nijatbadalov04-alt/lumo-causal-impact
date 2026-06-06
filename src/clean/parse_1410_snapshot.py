"""
Parse ORR Table 1410 (latest annual snapshot, 2024-25) and RECONCILE it against
the Table 1415 time series — an independent QA cross-check of the spine (§5).

THINK -> RESEARCH -> CODE
  WHAT: (1) Parse the rich 1410 CSV into a station-attributes table:
            ticket-type split (Full/Reduced/Season/All), interchanges,
            facility owner (TOC), station GROUP flag, main origin/destination,
            and ORR's own 'Data source or adjustments' + 'Quality limitations'.
        (2) Reconcile 1410 'entries & exits: All' (2024-25) against the 1415
            time-series value for 2024-25, per CRS. They derive from the same
            release and should agree almost exactly; disagreement = a parse bug.
  WHY : Independent reconciliation is the difference between "I parsed a file" and
        "I validated the spine". The attributes (owner, group, ticket split) are
        also the backbone of treated/donor selection + heterogeneity at M2.
  COLS referenced BY POSITION (header has embedded newlines; positions are stable):
        0 station_name 1 ee_full 2 ee_reduced 3 ee_season 4 ee_all 5 ee_ranger
        6 interchanges 7 main_od 8 journeys_main_od 9 data_source_adj
        10 estimates_supp 11 quality_limit 12 additional_info 13 nlc 14 crs
        15 region 16 facility_owner 17 station_group

Run:  python -m src.clean.parse_1410_snapshot
Out:  data/interim/station_attributes_2024_25.parquet
      results/metrics/qa_reconcile_1410_1415.json
"""

from __future__ import annotations

import json

import polars as pl

from src.utils.logging_setup import get_logger
from src.utils.paths import INTERIM, METRICS, RAW, ensure_dirs

LOG = get_logger("clean.parse_1410", log_file="logs/clean.log")

CSV_1410 = RAW / "table-1410-station-usage-2024-25.csv"
COLNAMES = [
    "station_name",
    "ee_full",
    "ee_reduced",
    "ee_season",
    "ee_all",
    "ee_ranger",
    "interchanges",
    "main_od",
    "journeys_main_od",
    "data_source_adj",
    "estimates_supp",
    "quality_limit",
    "additional_info",
    "nlc",
    "crs",
    "region",
    "facility_owner",
    "station_group",
]
NUMERIC = ["ee_full", "ee_reduced", "ee_season", "ee_all", "ee_ranger", "interchanges", "journeys_main_od"]
TREATED_CRS = {"EDB": "Edinburgh", "NCL": "Newcastle", "MPT": "Morpeth", "SVG": "Stevenage"}


def _clean_num(col: str) -> pl.Expr:
    """'11,873,686' -> 11873686.0 ; '[z]'/'[x]'/'' -> null."""
    return (
        pl.col(col).cast(pl.Utf8).str.replace_all(",", "").str.strip_chars().cast(pl.Float64, strict=False).alias(col)
    )


def parse_1410() -> pl.DataFrame:
    raw = pl.read_csv(CSV_1410, has_header=False, infer_schema_length=0, truncate_ragged_lines=True)
    df = raw.rename(dict(zip(raw.columns, COLNAMES))).slice(4)  # drop 3 title rows + header
    df = df.with_columns([_clean_num(c) for c in NUMERIC])
    # Keep rows that look like real stations (valid 3-letter CRS).
    df = df.filter(pl.col("crs").str.contains(r"^[A-Z]{3}$"))
    LOG.info("parsed 1410: %d stations with valid CRS", df.height)
    return df


def reconcile(attr: pl.DataFrame) -> dict:
    ts = (
        pl.read_parquet(INTERIM / "station_usage_long.parquet")
        .filter((pl.col("metric") == "entries_exits") & (pl.col("year_start") == 2024))
        .select(["crs", pl.col("value").alias("ts_ee_2024")])
    )
    m = attr.select(["crs", "ee_all"]).join(ts, on="crs", how="inner").drop_nulls()
    m = m.with_columns(
        [
            (pl.col("ee_all") - pl.col("ts_ee_2024")).abs().alias("abs_diff"),
            ((pl.col("ee_all") - pl.col("ts_ee_2024")).abs() / pl.col("ts_ee_2024").clip(lower_bound=1)).alias(
                "rel_diff"
            ),
        ]
    )
    rep = {
        "n_joined": m.height,
        "n_exact_match": int((m["abs_diff"] == 0).sum()),
        "n_within_0.1pct": int((m["rel_diff"] <= 0.001).sum()),
        "max_abs_diff": float(m["abs_diff"].max()),
        "max_rel_diff": float(m["rel_diff"].max()),
        "pearson_corr": float(pl.DataFrame({"a": m["ee_all"], "b": m["ts_ee_2024"]}).select(pl.corr("a", "b")).item()),
    }
    LOG.info("reconcile 1410 vs 1415 (2024-25 EE): %s", json.dumps(rep))
    # Hard gate: the two ORR products MUST agree to within rounding on >99% of stations.
    assert rep["n_within_0.1pct"] / rep["n_joined"] > 0.99, "1410/1415 disagree — parse bug?"
    assert rep["pearson_corr"] > 0.9999, "1410/1415 correlation too low — parse bug?"
    return rep


def check_treated(attr: pl.DataFrame) -> None:
    LOG.info("--- Treated (Lumo-served) station attributes, 2024-25 ---")
    sub = attr.filter(pl.col("crs").is_in(list(TREATED_CRS)))
    found = set(sub["crs"].to_list())
    for crs in sorted(sub.rows_by_key("crs")):
        r = sub.filter(pl.col("crs") == crs).to_dicts()[0]
        LOG.info(
            "  %s %-10s owner=%-14s group=%-4s EE_all=%s  main_OD=%s",
            crs,
            r["station_name"],
            str(r["facility_owner"])[:14],
            str(r["station_group"]),
            f"{r['ee_all']:,.0f}" if r["ee_all"] else "NA",
            r["main_od"],
        )
    missing = set(TREATED_CRS) - found
    assert not missing, f"Treated stations missing from panel: {missing}"


def main() -> None:
    ensure_dirs()
    attr = parse_1410()
    out = INTERIM / "station_attributes_2024_25.parquet"
    attr.write_parquet(out, compression="zstd")
    LOG.info("wrote %s (%d rows, %d cols)", out, attr.height, attr.width)

    rep = reconcile(attr)
    check_treated(attr)

    (METRICS / "qa_reconcile_1410_1415.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    LOG.info("1410 snapshot parse + reconciliation complete.")


if __name__ == "__main__":
    main()
