"""
Build the unified analysis panel (M2 foundation).

THINK -> RESEARCH -> CODE
  WHAT: Restrict the entries+exits series to the comparable **LENNON era** (>=2004),
        join the 2024-25 station attributes, and attach treatment / exposure flags.
  WHY : The counterfactual estimators need a clean, balanced donor matrix and an
        unambiguous treated set. We work in the LENNON era to avoid the CAPRI break
        (D-006). Exposure to Lumo is proxied data-drivenly by each station's
        dominant flow (`main_od == "London Kings Cross"`) rather than guessed.
  FLAGS added:
        lumo_served   - one of EDB/NCL/MPT/SVG (Lumo stop)
        post          - financial year >= 2021 (launch year FY)
        exposure_kgx  - dominant 2024-25 flow is London Kings Cross (KGX corridor)
        not_grouped   - station_group == [z] (no apportionment contamination)
        balanced      - observed in EVERY LENNON year (eligible for SC donor matrix)
  Out:  data/processed/panel.parquet  +  results/metrics/m2_panel_summary.json
"""

from __future__ import annotations

import json

import polars as pl

from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import INTERIM, METRICS, PROCESSED, ensure_dirs

LOG = get_logger("features.build_panel", log_file="logs/features.log")


def main() -> None:
    ensure_dirs()
    cfg = load_config("base")
    lumo = cfg["treatments"]["lumo"]
    served = list(lumo["served_crs"])
    treat_year = int(lumo["treat_year_start"])
    lennon_min = int(cfg["panel"]["lennon_era_min"])
    year_max = int(cfg["panel"]["year_max"])
    n_years = year_max - lennon_min + 1

    long = pl.read_parquet(INTERIM / "station_usage_long.parquet")
    ee = long.filter(
        (pl.col("metric") == "entries_exits")
        & (pl.col("year_start") >= lennon_min)
        & (pl.col("crs").str.contains(r"^[A-Z]{3}$"))
    ).select(["crs", "station_name", "region", "year_start", "value", "flag", "header_break"])

    attr = pl.read_parquet(INTERIM / "station_attributes_2024_25.parquet").select(
        [
            "crs",
            "facility_owner",
            "station_group",
            "main_od",
            "journeys_main_od",
            "ee_full",
            "ee_reduced",
            "ee_season",
            "ee_all",
        ]
    )

    panel = ee.join(attr, on="crs", how="left").with_columns(
        pl.col("value").log1p().alias("log_ee"),
        pl.col("crs").is_in(served).alias("lumo_served"),
        (pl.col("year_start") >= treat_year).alias("post"),
        (pl.col("main_od") == "London Kings Cross").alias("exposure_kgx"),
        (pl.col("station_group") == "[z]").alias("not_grouped"),
    )

    # Balanced units: an observed value in EVERY LENNON year (donor-matrix eligible).
    obs_counts = (
        panel.filter(pl.col("flag") == "observed")
        .group_by("crs")
        .agg(pl.col("year_start").n_unique().alias("n_obs_years"))
    )
    panel = panel.join(obs_counts, on="crs", how="left").with_columns(
        (pl.col("n_obs_years") == n_years).alias("balanced")
    )

    out = PROCESSED / "panel.parquet"
    panel.write_parquet(out, compression="zstd")
    LOG.info("wrote %s (%d rows, %d cols)", out, panel.height, panel.width)

    # ----- summary for the panel review -----
    latest = panel.filter(pl.col("year_start") == year_max)
    summary = {
        "lennon_years": [lennon_min, year_max],
        "n_years": n_years,
        "n_stations": panel["crs"].n_unique(),
        "n_balanced_stations": int(panel.filter(pl.col("balanced"))["crs"].n_unique()),
        "treated_lumo": served,
        "n_kgx_exposed_stations": int(latest.filter(pl.col("exposure_kgx"))["crs"].n_unique()),
        "kgx_exposed_examples": latest.filter(pl.col("exposure_kgx"))
        .sort("ee_all", descending=True)["station_name"]
        .head(20)
        .to_list(),
        "donor_eligible_pool": int(
            panel.filter(pl.col("balanced") & ~pl.col("lumo_served") & pl.col("not_grouped"))["crs"].n_unique()
        ),
    }
    (METRICS / "m2_panel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("panel summary: %s", json.dumps(summary))
    LOG.info("M2 panel build complete.")


if __name__ == "__main__":
    main()
