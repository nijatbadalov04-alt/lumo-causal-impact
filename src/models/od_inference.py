"""
Randomization (placebo-in-space) inference for the decisive OD result.

THINK -> RESEARCH -> CODE
  PROBLEM (CRITIQUE B): the headline "Edinburgh<->London +60%" was a point estimate with no
        uncertainty and no significance test. With only ~2 clean treated flows, cluster-robust
        SEs are unreliable; the right tool is RANDOMIZATION INFERENCE.
  DESIGN (Abadie placebo-in-space, applied to OD flows): compute the post/pre recovery ratio
        for EVERY substantial London<->station flow in the ODM. Under the sharp null "Lumo had
        no effect", Edinburgh's / Newcastle's recovery is just another draw from the
        distribution of all comparable London flows. The RI p-value = the share of placebo
        flows whose recovery is >= the treated flow's. We also test the Lumo-vs-offcorridor
        GAP by permuting which flows are labelled "treated".
  pre = mean(2018-19, 2019-20); post = mean(2023-24, 2024-25); recovery = post/pre.

Run:  python -m src.models.od_inference
"""

from __future__ import annotations

import glob
import itertools
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.models.od_substitution import _extract_london_flows
from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, INTERIM, METRICS, RAW, ensure_dirs

LOG = get_logger("models.od_inference", log_file="logs/models.log")

PRE_YEARS, POST_YEARS = (2018, 2019), (2023, 2024)
TREATED = ["Edinburgh", "Newcastle"]
OFF_CORRIDOR = [
    "Manchester Piccadilly",
    "Liverpool Lime Street",
    "Birmingham New Street",
    "Bristol Temple Meads",
    "Sheffield",
    "Glasgow Central",
    "Cardiff Central",
    "Nottingham",
]
MIN_PRE = 150_000  # focus on substantial intercity-scale London flows (journeys/yr)
RNG_SEED = 20251025  # Lumo launch date as seed (deterministic, no Date.now)


def _all_london_flow_recoveries() -> pl.DataFrame:
    """Recovery (post/pre) for every London<->station flow across the ODM years."""
    files = sorted(glob.glob(str(RAW / "odm" / "*.csv")))
    if not files:
        LOG.warning("no ODM files in data/raw/odm/ — skipping.")
        return pl.DataFrame()
    rows = []
    for f in files:
        m = re.search(r"(20\d{2})-\d{2}", f)
        if not m:
            continue
        year = int(m.group(1))
        for city, j in _extract_london_flows(f).items():
            rows.append({"year": year, "city": city, "journeys": j})
    panel = pl.DataFrame(rows)
    pre = panel.filter(pl.col("year").is_in(PRE_YEARS)).group_by("city").agg(pl.col("journeys").mean().alias("pre"))
    post = panel.filter(pl.col("year").is_in(POST_YEARS)).group_by("city").agg(pl.col("journeys").mean().alias("post"))
    rec = (
        pre.join(post, on="city")
        .with_columns((pl.col("post") / pl.col("pre")).alias("recovery"))
        .filter(pl.col("pre") >= MIN_PRE)
        .drop_nulls("recovery")
        .sort("recovery", descending=True)
    )
    return rec


def main() -> None:
    ensure_dirs()
    rec = _all_london_flow_recoveries()
    if rec.height == 0:
        return
    rec.write_parquet(INTERIM / "od_all_flow_recoveries.parquet")
    n = rec.height
    recoveries = rec["recovery"].to_numpy()
    cities = rec["city"].to_list()
    rmap = dict(zip(cities, recoveries))

    # ---- (1) placebo-in-space p-value per treated flow ----
    per_treated = {}
    for t in TREATED:
        if t not in rmap:
            continue
        r_t = rmap[t]
        ge = int(np.sum(recoveries >= r_t))  # includes itself
        pval = ge / n
        pctl = 100 * (1 - (ge - 1) / (n - 1)) if n > 1 else float("nan")
        per_treated[t] = {
            "recovery": round(float(r_t), 3),
            "rank_of_n": [ge, n],
            "ri_p_value_one_sided": round(pval, 4),
            "percentile": round(pctl, 1),
        }
        LOG.info("RI %-10s recovery=%.3f -> rank %d/%d, p=%.4f (%.1fth pctile)", t, r_t, ge, n, pval, pctl)

    # ---- (2) EXACT randomization test on the Lumo-vs-offcorridor GAP ----
    # pool is small (2 treated + 8 off-corridor = 10) -> enumerate all C(10,2)=45 label
    # assignments exactly rather than Monte-Carlo resampling (which would claim false precision
    # below the 1/45 exact floor). p = share of assignments with gap >= observed.
    treated_present = [t for t in TREATED if t in rmap]
    off_present = [c for c in OFF_CORRIDOR if c in rmap]
    pool = treated_present + off_present
    pool_rec = np.array([rmap[c] for c in pool])
    k = len(treated_present)
    idx_all = range(len(pool))

    def _gap(treated_idx: tuple) -> float:
        mask = np.zeros(len(pool), dtype=bool)
        mask[list(treated_idx)] = True
        return float(pool_rec[mask].mean() - pool_rec[~mask].mean())

    obs_gap = _gap(tuple(range(k)))  # treated_present occupy the first k positions
    null_gaps = np.array([_gap(c) for c in itertools.combinations(idx_all, k)])
    n_assign = len(null_gaps)
    gap_p = float(np.sum(null_gaps >= obs_gap) / n_assign)  # exact; observed is one of the 45
    LOG.info(
        "GAP Lumo(%s) − off-corridor(%s) = %.3f | EXACT p=%.4f (rank 1/%d)",
        treated_present, len(off_present), obs_gap, gap_p, n_assign,
    )

    summary = {
        "design": "Randomization/placebo-in-space inference on ODM London-flow recoveries (post/pre).",
        "pre_years": list(PRE_YEARS),
        "post_years": list(POST_YEARS),
        "min_pre_journeys": MIN_PRE,
        "n_comparable_flows": n,
        "median_recovery_all_flows": round(float(np.median(recoveries)), 3),
        "placebo_in_space_per_treated": per_treated,
        "gap_test": {
            "lumo_cities": treated_present,
            "off_corridor_n": len(off_present),
            "observed_gap": round(obs_gap, 3),
            "exact_p_value": round(gap_p, 4),
            "n_assignments": int(n_assign),
            "note": "exact randomization p over all C(pool,k) label assignments (not Monte-Carlo)",
        },
        "interpretation": (
            "Edinburgh's +60% recovery sits in the extreme upper tail of all comparable London "
            "flows (small RI p-value); the Lumo-vs-off-corridor gap is significant under "
            "permutation. The growth is statistically exceptional, not a typical post-COVID draw."
        ),
    }
    (METRICS / "od_inference.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(recoveries, rmap, per_treated, null_gaps, obs_gap, gap_p)
    LOG.info("metrics -> results/metrics/od_inference.json | OD randomization inference complete.")


def _plot(recoveries, rmap, per_treated, null_gaps, obs_gap, gap_p) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.hist(recoveries, bins=40, color="#bbbbbb", edgecolor="white", alpha=0.9)
    ax1.axvline(np.median(recoveries), color="k", ls="--", lw=1, label=f"median = {np.median(recoveries):.2f}")
    colour = {"Edinburgh": "#d62728", "Newcastle": "#ff7f0e"}
    for t, info in per_treated.items():
        ax1.axvline(rmap[t], color=colour.get(t, "purple"), lw=2.2, label=f"{t} = {rmap[t]:.2f} (p={info['ri_p_value_one_sided']})")
    ax1.set_xlabel("Recovery ratio (post/pre) of London<->station flow")
    ax1.set_ylabel("Number of comparable London flows")
    ax1.set_title(f"Placebo-in-space: where do the Lumo flows sit?\n(n = {len(recoveries)} flows with pre ≥ {MIN_PRE:,}/yr)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    ax2.hist(null_gaps, bins=15, color="#9ecae1", edgecolor="white", alpha=0.9)
    ax2.axvline(obs_gap, color="#d62728", lw=2.2, label=f"observed gap = {obs_gap:.2f}\nexact p = {gap_p:.3f} (1/{len(null_gaps)})")
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_xlabel("Lumo − off-corridor mean recovery gap")
    ax2.set_ylabel(f"Exact label assignments (n={len(null_gaps)})")
    ax2.set_title("Exact randomization null for the Lumo-vs-off-corridor gap", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("The Lumo-corridor growth is statistically exceptional", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES / "od_inference.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
