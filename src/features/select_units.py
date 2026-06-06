"""
Select & classify analysis units for the Lumo experiment (M2 → unblocks M3).

THINK -> RESEARCH -> CODE
  WHAT: Assign every station a ROLE for the synthetic-control / DiD design:
        treated_lumo          - a Lumo stop (NCL/SVG/EDB/MPT)
        ecml_corridor_control - on the East Coast Main Line / KGX corridor but NOT
                                a Lumo stop ⇒ exposed to LNER spillovers (SUTVA risk)
        donor_clean           - balanced, not station-grouped, OFF the ECML corridor
                                ⇒ the conservative donor pool
        excluded              - unbalanced series or in a station group (contaminated)
  WHY : donors must not be contaminated by the treatment (ECML spillovers)
        nor by station-group apportionment. Chen: stopping vs through must be explicit.
        We therefore separate the corridor (spillover-test group) from clean donors.
  HOW : Corridor = data-driven KGX exposure (main_od == 'London Kings Cross') UNION a
        name-curated set of NORTHERN ECML stations whose dominant flow is regional
        (Newcastle/Edinburgh), which the KGX proxy alone would miss. Station-group
        screening uses ORR's own 'station_group' field (D-006). Codes are matched by
        NAME for the northern additions to avoid asserting unverified CRS codes.
  Out: data/processed/units.parquet  +  results/tables/m2_unit_classification.csv
       +  results/metrics/m2_unit_selection.json
"""

from __future__ import annotations

import json

import polars as pl

from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import METRICS, PROCESSED, TABLES, ensure_dirs

LOG = get_logger("features.select_units", log_file="logs/features.log")

# Northern ECML stations whose main flow is regional (not KGX) — matched by NAME
# (robust; the KGX-exposure proxy already captures the southern/intercity corridor).
ECML_NORTH_NAMES = {
    "Durham",
    "Chester-le-Street",
    "Berwick-upon-Tweed",
    "Alnmouth",
    "Morpeth",
    "Northallerton",
    "Thirsk",
    "Edinburgh",
    "Dunbar",
    "Wakefield Westgate",
    "Leeds",
    "Newcastle",
    "York",
    "Darlington",
    "Doncaster",
}
BASELINE_YEAR = 2019  # pre-COVID, pre-Lumo size for donor matching
LARGE_INTERCITY_THRESHOLD = 1_000_000  # entries+exits — "comparable to a city station"


def main() -> None:
    ensure_dirs()
    cfg = load_config("base")
    served = set(cfg["treatments"]["lumo"]["served_crs"])

    panel = pl.read_parquet(PROCESSED / "panel.parquet")

    # Station-level static attributes (constant across years per crs).
    stn = panel.group_by("crs").agg(
        pl.col("station_name").first(),
        pl.col("facility_owner").first(),
        pl.col("main_od").first(),
        pl.col("exposure_kgx").first(),
        pl.col("not_grouped").first(),
        pl.col("balanced").first(),
    )
    base = panel.filter(pl.col("year_start") == BASELINE_YEAR).select("crs", pl.col("value").alias("baseline_ee_2019"))
    stn = stn.join(base, on="crs", how="left")

    ecml_north_lc = [n.lower() for n in ECML_NORTH_NAMES]
    stn = stn.with_columns(
        pl.col("crs").is_in(list(served)).alias("lumo_served"),
        (pl.col("exposure_kgx") | pl.col("station_name").str.to_lowercase().is_in(ecml_north_lc))  # case-insensitive
        .alias("ecml_corridor"),
    )
    stn = stn.with_columns(
        pl.when(pl.col("lumo_served"))
        .then(pl.lit("treated_lumo"))
        .when(pl.col("ecml_corridor"))
        .then(pl.lit("ecml_corridor_control"))
        .when(pl.col("balanced") & pl.col("not_grouped"))
        .then(pl.lit("donor_clean"))
        .otherwise(pl.lit("excluded"))
        .alias("role")
    )

    out = PROCESSED / "units.parquet"
    stn.write_parquet(out, compression="zstd")

    # --- verify the name-curated ECML additions actually matched (case-insensitive) ---
    matched_names = set(
        stn.filter(pl.col("station_name").str.to_lowercase().is_in(ecml_north_lc))["station_name"].to_list()
    )
    matched_lc = {m.lower() for m in matched_names}
    missing_names = sorted(n for n in ECML_NORTH_NAMES if n.lower() not in matched_lc)
    if missing_names:
        LOG.warning("ECML_NORTH names not found in panel (verify spelling/CRS): %s", missing_names)

    role_counts = {r: int(c) for r, c in zip(*stn["role"].value_counts().to_dict(as_series=False).values())}
    clean = stn.filter(pl.col("role") == "donor_clean")
    n_large_clean = int(clean.filter(pl.col("baseline_ee_2019") >= LARGE_INTERCITY_THRESHOLD).height)

    summary = {
        "role_counts": role_counts,
        "n_donor_clean": int(clean.height),
        "n_donor_clean_large_intercity": n_large_clean,
        "ecml_north_names_matched": sorted(matched_names),
        "ecml_north_names_missing": missing_names,
    }
    (METRICS / "m2_unit_selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Human-readable classification table for treated + corridor (the design-critical ones).
    design = (
        stn.filter(pl.col("role").is_in(["treated_lumo", "ecml_corridor_control"]))
        .sort(["role", "baseline_ee_2019"], descending=[False, True])
        .select(
            "crs", "station_name", "role", "facility_owner", "main_od", "baseline_ee_2019", "not_grouped", "balanced"
        )
    )
    design.write_csv(TABLES / "m2_unit_classification.csv")

    LOG.info("unit selection: %s", json.dumps(summary))
    LOG.info("treated + corridor units (design-critical):")
    for r in design.iter_rows(named=True):
        LOG.info(
            "  %-22s %-26s %-22s base2019=%s",
            r["role"],
            r["station_name"],
            str(r["main_od"])[:22],
            f"{r['baseline_ee_2019']:,.0f}" if r["baseline_ee_2019"] else "NA",
        )
    LOG.info("M2 unit selection complete.")


if __name__ == "__main__":
    main()
