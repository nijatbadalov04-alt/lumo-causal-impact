"""
Parse the Green Travel (GTD) per-OD-pair carbon table -> compact corridor lookup.

THINK -> RESEARCH -> CODE
  WHAT: The GTD "emissions by journey (with comparisons)" file (1.2 GB) gives, for every
        GB station pair, the average kg CO2e per passenger by mode: standard rail, first
        rail, average car, petrol/diesel/PHEV/BEV car, and AIR, plus pre-computed savings
        of rail over car / rail over air.
  WHY:  Multiplied by the ODM journey growth on the Lumo corridor, this turns "the market
        grew" into "...and here is the CO2 consequence" (the climate punchline). We only
        need the London Kings Cross pairs (the ECML / Lumo terminus), so we filter the
        1.2 GB file down to ~2.5k rows and persist a tidy parquet.
  SOURCE: Rail Data Marketplace, "Green Travel Pledge Emissions Output" (OGL3),
          methodology gtd-methodology-2026-04-01.docx. Snapshot 2025-12 -> 2026-05.

Run:  python -m src.clean.parse_carbon
"""

from __future__ import annotations

import glob

import polars as pl

from src.utils.logging_setup import get_logger
from src.utils.paths import INTERIM, RAW, ensure_dirs

LOG = get_logger("clean.parse_carbon", log_file="logs/clean.log")

# the ECML / Lumo London terminus; emissions are ~distance-driven so the terminus matters
LONDON_TERMINI = ["London Kings Cross"]

# raw GTD column -> tidy name (kg CO2e per passenger, one-way)
_COLS = {
    "OriginStation": "origin",
    "OriginStationCRSCode": "origin_crs",
    "DestinationStation": "destination",
    "DestinationStationCRSCode": "destination_crs",
    "AverageEmissionsPerStandardRailPassengerKGCO2e": "rail_standard_kg",
    "AverageEmissionsPerFirstClassRailPassengerKGCO2e": "rail_first_kg",
    "AverageEmissionsPerRailPassengerKGCO2e": "rail_avg_kg",
    "AverageCarEmissionsKGCO2e": "car_avg_kg",
    "AveragePetrolDieselCarEmissionsKGCO2e": "car_petroldiesel_kg",
    "AverageBatteryElectricCarEmissionsKGCO2e": "car_bev_kg",
    "AverageAirEmissionsKGCO2e": "air_kg",
    "SavingStandardRailOverAveCarKGCO2e": "saving_rail_over_car_kg",
    "SavingStandardRailOverAirKGCO2e": "saving_rail_over_air_kg",
}


def _find_gtd_file() -> str | None:
    hits = glob.glob(str(RAW / "carbon" / "*by-journey*comparisons*.csv"))
    return sorted(hits)[0] if hits else None


def main() -> None:
    ensure_dirs()
    path = _find_gtd_file()
    if not path:
        LOG.warning("no GTD carbon file in data/raw/carbon/ — skipping (user-provided via RDM).")
        return

    LOG.info("scanning GTD carbon file (lazy): %s", path.split("/")[-1].split("\\")[-1])
    lf = pl.scan_csv(path, infer_schema_length=5000)
    keep = list(_COLS.keys())
    sub = (
        lf.select(keep)
        .filter(
            pl.col("OriginStation").is_in(LONDON_TERMINI) | pl.col("DestinationStation").is_in(LONDON_TERMINI)
        )
        .rename(_COLS)
        .collect()
    )

    # orient every row as (London terminus) <-> (other station): "city" = the non-London end
    sub = sub.with_columns(
        pl.when(pl.col("origin").is_in(LONDON_TERMINI))
        .then(pl.col("destination"))
        .otherwise(pl.col("origin"))
        .alias("city"),
        pl.when(pl.col("origin").is_in(LONDON_TERMINI))
        .then(pl.col("destination_crs"))
        .otherwise(pl.col("origin_crs"))
        .alias("city_crs"),
    )
    # both directions of a pair carry identical emissions -> keep one row per city
    out = (
        sub.sort("city")
        .unique(subset=["city_crs"], keep="first")
        .select(
            "city",
            "city_crs",
            "rail_standard_kg",
            "rail_first_kg",
            "rail_avg_kg",
            "car_avg_kg",
            "car_bev_kg",
            "air_kg",
            "saving_rail_over_car_kg",
            "saving_rail_over_air_kg",
        )
    )
    out.write_parquet(INTERIM / "carbon_kgx_pairs.parquet")
    LOG.info(
        "wrote %d London Kings Cross carbon pairs -> data/interim/carbon_kgx_pairs.parquet",
        out.height,
    )
    # spot-check the headline corridor pairs
    chk = out.filter(pl.col("city").is_in(["Edinburgh", "Newcastle", "York", "Leeds", "Glasgow Central"]))
    for r in chk.iter_rows(named=True):
        LOG.info(
            "  %-22s rail=%.1f car=%.1f air=%.1f | save vs air=%.1f kg/pax",
            r["city"],
            r["rail_standard_kg"],
            r["car_avg_kg"],
            r["air_kg"],
            r["saving_rail_over_air_kg"],
        )


if __name__ == "__main__":
    main()
