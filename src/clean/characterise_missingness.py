"""
Characterise missingness in the station-usage panel (Milestone M1).

THINK -> RESEARCH -> CODE
  WHAT: Diagnose BEFORE imputing. Produce (a) a station x year missingness matrix
        that reveals the *structure* of missingness, (b) per-year and per-station
        summaries, and (c) a written MCAR/MAR/MNAR characterisation.
  WHY : The brief insists we never conflate a TRUE ZERO, a genuinely MISSING value,
        and a STRUCTURALLY-ABSENT station-year (station not yet open). Those have
        different correct treatments (true zero = keep; missing = principled
        imputation + sensitivity; structurally absent = panel entry/exit, NOT
        imputation). Imputation strategy at M2 must be justified by this diagnosis.
  TAXONOMY (from the parsed `flag`):
        observed        -> real estimate (incl. genuine 0 = TRUE ZERO)
        not_available   -> [x] missing (incl. the universal 2003-04 admin gap)
        not_applicable  -> [z] structurally absent (station not open / N/A)

Run:  python -m src.clean.characterise_missingness
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, INTERIM, METRICS, RESULTS, TABLES, ensure_dirs

LOG = get_logger("clean.missingness", log_file="logs/clean.log")

FLAG_CODE = {
    "observed": 0,
    "not_available": 1,
    "not_applicable": 2,
    "confidential": 3,
    "unknown_missing": 4,
}
FLAG_COLOR = {
    "observed": "#1b7837",  # green
    "not_available": "#d73027",  # red  (genuinely missing)
    "not_applicable": "#bdbdbd",  # grey (structurally absent)
    "confidential": "#762a83",  # purple
    "unknown_missing": "#000000",
}


def _missingness_matrix(ee: pd.DataFrame) -> None:
    """station x year matrix, rows sorted to expose the station-entry staircase."""
    mat = ee.pivot_table(index="crs", columns="year_start", values="fc", aggfunc="first")
    years = list(mat.columns)
    arr = mat.to_numpy(dtype=float)

    observed = arr == 0
    first_obs = np.where(observed.any(1), observed.argmax(1), arr.shape[1])
    n_obs = observed.sum(1)
    order = np.lexsort((-n_obs, first_obs))  # by first-observed year, then #observed desc
    arr_sorted = arr[order]

    used = [f for f, c in FLAG_CODE.items() if (arr == c).any()]
    cmap = ListedColormap([FLAG_COLOR[f] for f in used])
    codes = [FLAG_CODE[f] for f in used]
    norm = BoundaryNorm([c - 0.5 for c in codes] + [max(codes) + 0.5], cmap.N)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(arr_sorted, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([f"{y}" for y in years], rotation=90, fontsize=7)
    ax.set_xlabel("Financial year (start)")
    ax.set_ylabel(f"Stations (n={arr.shape[0]}, sorted by first observed year)")
    ax.set_title(
        "Missingness structure — ORR station entries & exits, 1997-98 to 2024-25\n"
        "Grey staircase = stations not yet open (structurally absent); red vertical "
        "stripe at 2003-04 = ORR's universal 'no estimates produced' gap.",
        fontsize=10,
    )
    legend = [Patch(facecolor=FLAG_COLOR[f], label=f) for f in used]
    ax.legend(handles=legend, loc="lower right", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    out = FIGURES / "m1_missingness_matrix_entries_exits.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("figure -> %s", out)


def _observed_fraction_by_year(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby(["metric", "year_start", "flag"], observed=True).size().unstack("flag", fill_value=0)
    g["total"] = g.sum(axis=1)
    g["frac_observed"] = g.get("observed", 0) / g["total"]
    g = g.reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    for metric, sub in g.groupby("metric", observed=True):
        ax.plot(sub["year_start"], sub["frac_observed"], marker="o", ms=3, label=metric)
    ax.axvline(2003, color="#d73027", ls="--", lw=1, alpha=0.7)
    ax.text(2003.2, 0.05, "2003-04 gap", color="#d73027", fontsize=8, rotation=90, va="bottom")
    ax.axvline(2021, color="#2166ac", ls=":", lw=1, alpha=0.8)
    ax.text(2021.2, 0.05, "Lumo (Oct 2021)", color="#2166ac", fontsize=8, rotation=90, va="bottom")
    ax.set_xlabel("Financial year (start)")
    ax.set_ylabel("Fraction of stations 'observed'")
    ax.set_ylim(0, 1.02)
    ax.set_title("Share of stations with an observed estimate, by year and metric")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "m1_observed_fraction_by_year.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("figure -> %s", out)
    return g


def main() -> None:
    ensure_dirs()
    long = pd.read_parquet(INTERIM / "station_usage_long.parquet")
    long["flag"] = long["flag"].astype(str)

    ee = long[(long["metric"] == "entries_exits") & long["crs"].str.fullmatch(r"[A-Z]{3}", na=False)].copy()
    ee["fc"] = ee["flag"].map(FLAG_CODE)

    _missingness_matrix(ee)
    by_year = _observed_fraction_by_year(long)
    by_year.to_csv(TABLES / "m1_missingness_by_year.csv", index=False)

    # Per-station profile (entries & exits): #observed years, first/last observed,
    # internal gaps (missing years *between* first and last observed = imputable).
    prof_rows = []
    for crs, sub in ee.sort_values("year_start").groupby("crs", observed=True):
        obs_years = sub.loc[sub["flag"] == "observed", "year_start"].tolist()
        if obs_years:
            lo, hi = min(obs_years), max(obs_years)
            span = [y for y in range(lo, hi + 1) if y != 2003]  # 2003-04 excluded by design
            internal_gaps = sorted(set(span) - set(obs_years))
        else:
            lo = hi = None
            internal_gaps = []
        prof_rows.append(
            {
                "crs": crs,
                "station_name": sub["station_name"].iloc[0],
                "n_observed": len(obs_years),
                "first_observed": lo,
                "last_observed": hi,
                "n_internal_gaps": len(internal_gaps),
                "internal_gap_years": ";".join(map(str, internal_gaps)),
            }
        )
    profile = pd.DataFrame(prof_rows)
    profile.to_csv(TABLES / "m1_station_missingness_profile.csv", index=False)

    # True zeros are a distinct, important category — count them explicitly.
    n_true_zero = int(((long["flag"] == "observed") & (long["value"] == 0)).sum())

    summary = {
        "n_station_year_metric_rows": int(len(long)),
        "flag_counts": {k: int(v) for k, v in long["flag"].value_counts().items()},
        "true_zero_observed_cells": n_true_zero,
        "stations_with_any_observed": int((profile["n_observed"] > 0).sum()),
        "stations_full_28y_minus_gap": int((profile["n_observed"] >= 27).sum()),
        "stations_with_internal_gaps": int((profile["n_internal_gaps"] > 0).sum()),
        "total_internal_gap_cells_ee": int(profile["n_internal_gaps"].sum()),
        "median_first_observed_year": float(profile["first_observed"].median()),
    }
    (METRICS / "missingness_characterisation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("missingness summary: %s", json.dumps(summary))

    _write_prose(summary, by_year)
    LOG.info("missingness characterisation complete.")


def _write_prose(summary: dict, by_year: pd.DataFrame) -> None:
    """Written MCAR/MAR/MNAR characterisation (research narrative, not box-ticking)."""
    md = f"""# Missingness Characterisation — ORR Station Usage panel (M1)

Generated by `src/clean/characterise_missingness.py`. Diagnose **before** imputing.

## The three faces of "no number" (kept distinct — never conflated)
- **TRUE ZERO** — `flag=observed, value=0`: a real estimate of zero usage
  (e.g. interchanges at a small station). **{summary["true_zero_observed_cells"]:,}** such cells.
  *Keep as 0; never impute.*
- **STRUCTURALLY ABSENT** — `flag=not_applicable` (`[z]`): the station did not exist
  / the metric is N/A that year. **{summary["flag_counts"].get("not_applicable", 0):,}** cells.
  These form the grey "staircase" in the missingness matrix as stations open over
  time. *Handle by panel ENTRY/EXIT, not imputation* (a closed/unborn station has no
  counterfactual demand to fill in).
- **GENUINELY MISSING** — `flag=not_available` (`[x]`): **{summary["flag_counts"].get("not_available", 0):,}** cells.
  Dominated by the **universal 2003-04 administrative gap** ("no estimates produced"),
  which is missing for *every* station irrespective of its value.

## MCAR / MAR / MNAR
- **2003-04 gap → missing by DESIGN (effectively MCAR within that column).** It does
  not depend on the unobserved usage value; it is an administrative non-production.
  Treatment: exclude 2003-04 from all within-station interpolation spans (we already
  drop it from gap accounting); for models needing a continuous index, treat as a
  known structural break between the CAPRI (≤2002-03) and LENNON (≥2004-05) regimes.
- **Pre-opening `[z]` runs → structural, not random.** Strongly related to *observed*
  metadata (opening date / station age) ⇒ closest to **MAR**, but the right action is
  panel entry, so imputation is moot.
- **Scattered internal `[x]` gaps** ({summary["stations_with_internal_gaps"]:,} stations,
  {summary["total_internal_gap_cells_ee"]:,} cells *between* first & last observed year):
  candidates for principled imputation. Plausibly **MAR** (relate to line/region/PTE
  methodology changes) rather than MNAR. Treatment at M2: interpolation for short gaps;
  **sensitivity analysis** under ≥2 imputation strategies (per §5); flag any station-year
  carrying a `[b]` break or a "Quality limitations" note as non-comparable.

## Implication for the causal design
The natural experiments live in the LENNON era: **Lumo (2021)** and **Grand Central
(2007/2010)** are entirely post-2004 (no regime break in window). **Hull Trains (2000)**
straddles the CAPRI→LENNON break and the 2003-04 gap — its pre-period is CAPRI; we will
treat it as a robustness/replication case with the break modelled explicitly, not as the
primary estimate.

See `results/figures/m1_missingness_matrix_entries_exits.png` and
`results/figures/m1_observed_fraction_by_year.png`.
"""
    out = RESULTS / "missingness_characterisation.md"
    out.write_text(md, encoding="utf-8")
    LOG.info("prose -> %s", out)


if __name__ == "__main__":
    main()
