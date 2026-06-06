"""
OD-pair substitution vs creation — THE decisive test (using the ODM).

THINK -> RESEARCH -> CODE
  WHAT: The ODM gives total journeys on every station pair. We extract the TOTAL
        London <-> city market for each year (2018-19 → 2024-25) and ask: did the
        London<->Newcastle / London<->Edinburgh market (where Lumo entered) GROW more
        than comparable London routes after Lumo (Oct 2021)?
  LOGIC: the OD flow is operator-agnostic (LNER + Lumo + anyone), so:
        - CREATION: the London<->Lumo-city market grows ABOVE comparable markets ⇒ Lumo
          generated net-new journeys on that flow.
        - SUBSTITUTION: the market is flat (recovers like comparators) while Lumo merely
          takes share from LNER ⇒ no net market growth.
  CONTROLS (London<->city recovery, post/pre):
        Lumo stops: Newcastle, Edinburgh, (Stevenage commuter, Morpeth small)
        ECML non-Lumo (corridor): York, Leeds, Doncaster, Darlington
        Off-corridor long-distance: Manchester, Liverpool, Birmingham, Bristol, Sheffield,
          Glasgow, Cardiff, Nottingham (served from London by non-open-access operators)
  Pre = 2018-19 & 2019-20 (clean pre-COVID); Post = 2023-24 & 2024-25 (recovered post-Lumo).

Run:  python -m src.models.od_substitution
"""

from __future__ import annotations

import glob
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, INTERIM, METRICS, RAW, TABLES, ensure_dirs

LOG = get_logger("models.od_substitution", log_file="logs/models.log")

CITIES = {
    "Newcastle": "Lumo",
    "Edinburgh": "Lumo",
    "Morpeth": "Lumo",
    "Stevenage": "Lumo",
    "York": "ECML non-Lumo",
    "Leeds": "ECML non-Lumo",
    "Doncaster": "ECML non-Lumo",
    "Darlington": "ECML non-Lumo",
    "Manchester Piccadilly": "Off-corridor",
    "Liverpool Lime Street": "Off-corridor",
    "Birmingham New Street": "Off-corridor",
    "Bristol Temple Meads": "Off-corridor",
    "Sheffield": "Off-corridor",
    "Glasgow Central": "Off-corridor",
    "Cardiff Central": "Off-corridor",
    "Nottingham": "Off-corridor",
}

# The ODM is the complete set of 7 financial years 2018-19 … 2024-25 (parsed by start year 2018 …
# 2024); FY2021-22 = year 2021 is the Lumo LAUNCH year. The loader is year-agnostic: every CSV in
# data/raw/odm/ is read as one timeline, so the set is reproduced just by pointing at that folder.
LAUNCH_FY_YEAR = 2021
EXPECTED_FULL_SET = [2018, 2019, 2020, 2021, 2022, 2023, 2024]


def _extract_london_flows(file: str) -> dict:
    """journeys between London and each non-London station (both directions), one pass."""
    lf = pl.scan_csv(file, infer_schema_length=3000)
    one = lf.filter((pl.col("origin_region") == "London") != (pl.col("destination_region") == "London"))
    one = one.with_columns(
        pl.when(pl.col("origin_region") == "London")
        .then(pl.col("destination_station_name"))
        .otherwise(pl.col("origin_station_name"))
        .alias("city")
    )
    agg = one.group_by("city").agg(pl.col("journeys").sum().alias("london_journeys")).collect()
    return dict(zip(agg["city"].to_list(), agg["london_journeys"].to_list()))


def main() -> None:
    ensure_dirs()
    files = sorted(glob.glob(str(RAW / "odm" / "*.csv")))
    if not files:
        LOG.warning("no ODM files in data/raw/odm/ — skipping (user-provided via Rail Data Marketplace).")
        return
    rows = []
    for f in files:
        m = re.search(r"(20\d{2})-\d{2}", f)
        if not m:
            continue
        year = int(m.group(1))
        LOG.info("extracting London flows from %s (FY %d)...", f.split("/")[-1].split("\\")[-1], year)
        flows = _extract_london_flows(f)
        for city, grp in CITIES.items():
            rows.append({"year": year, "city": city, "group": grp, "london_journeys": flows.get(city)})

    panel = pl.DataFrame(rows)
    panel.write_parquet(INTERIM / "od_london_flows.parquet")
    pdf = panel.to_pandas().pivot_table(index="city", columns="year", values="london_journeys")
    pdf.to_csv(TABLES / "od_london_flows.csv")

    # --- explicit ODM coverage (single source of truth; never vague) ---
    years_loaded = sorted(panel["year"].unique().to_list())
    has_launch = LAUNCH_FY_YEAR in years_loaded
    LOG.info(
        "ODM coverage: %d financial year(s) loaded %s | launch FY2021-22 (year 2021): %s",
        len(years_loaded),
        years_loaded,
        "PRESENT" if has_launch else "absent in this checkout — add the year's CSV to data/raw/odm/ (year-agnostic loader)",
    )

    # recovery = mean(post 2023,2024) / mean(pre 2018,2019)
    import pandas as pd

    def recov(r):
        pre = np.nanmean([r.get(2018), r.get(2019)])
        post = np.nanmean([r.get(2023), r.get(2024)])
        return post / pre if pre and pre > 0 else np.nan

    rec = pdf.apply(lambda r: recov(r), axis=1).rename("recovery").to_frame()
    rec["group"] = rec.index.map(CITIES)
    grp_means = rec.groupby("group")["recovery"].mean().round(3).to_dict()
    ncl = float(rec.loc["Newcastle", "recovery"]) if "Newcastle" in rec.index else np.nan
    edb = float(rec.loc["Edinburgh", "recovery"]) if "Edinburgh" in rec.index else np.nan

    summary = {
        "odm_coverage": {
            "financial_years_loaded": years_loaded,
            "n_years": len(years_loaded),
            "launch_year_2021_present": has_launch,
            "expected_full_set": EXPECTED_FULL_SET,
            "missing": [y for y in EXPECTED_FULL_SET if y not in years_loaded],
            "note": "year-agnostic loader; add data/raw/odm/ODM_for_rdm_2021-22.csv to include FY2021-22 (no code change).",
        },
        "interpretation": "recovery = (2023-24 & 2024-25 mean) / (2018-19 & 2019-20 mean) of the London<->city market",
        "group_mean_recovery": grp_means,
        "Newcastle_London_recovery": round(ncl, 3),
        "Edinburgh_London_recovery": round(edb, 3),
        "Newcastle_London_journeys": {int(y): (None if pd.isna(v) else int(v)) for y, v in pdf.loc["Newcastle"].items()}
        if "Newcastle" in pdf.index
        else {},
        "Edinburgh_London_journeys": {int(y): (None if pd.isna(v) else int(v)) for y, v in pdf.loc["Edinburgh"].items()}
        if "Edinburgh" in pdf.index
        else {},
    }
    (METRICS / "od_substitution_creation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("London<->city market recovery (post/pre): %s", grp_means)
    LOG.info("Newcastle<->London recovery=%.3f | Edinburgh<->London recovery=%.3f", ncl, edb)
    LOG.info("Newcastle<->London journeys by year: %s", summary["Newcastle_London_journeys"])

    # figure: indexed London-market trajectories by group
    fig, ax = plt.subplots(figsize=(11, 6))
    colour = {"Lumo": "#d62728", "ECML non-Lumo": "#ff7f0e", "Off-corridor": "#1f77b4"}
    base_year = 2019
    for grp in ["Lumo", "ECML non-Lumo", "Off-corridor"]:
        cities = [c for c, g in CITIES.items() if g == grp and c in pdf.index]
        sub = pdf.loc[cities]
        idx = sub.div(sub[base_year], axis=0).mean(axis=0) * 100
        ax.plot(
            idx.index,
            idx.values,
            "o-",
            color=colour[grp],
            lw=2.2 if grp == "Lumo" else 1.5,
            label=f"{grp} (London market)",
        )
    ax.axvline(2021, color="grey", ls=":", lw=1.5)
    ax.text(2021.05, ax.get_ylim()[0], " Lumo (Oct 2021)", color="grey", rotation=90, fontsize=8, va="bottom")
    ax.axhline(100, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("Financial year (start)")
    ax.set_ylabel("London market journeys, indexed to 2019-20 = 100")
    ax.set_title(
        "OD-pair test — did the London market GROW where Lumo entered?\n"
        "Lumo-city London markets vs ECML-non-Lumo vs off-corridor (creation if Lumo line rises above)",
        fontsize=10,
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "od_substitution_creation.png", dpi=150)
    plt.close(fig)
    LOG.info("figure -> %s | OD substitution/creation complete.", FIGURES / "od_substitution_creation.png")


if __name__ == "__main__":
    main()
