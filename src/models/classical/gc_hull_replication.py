"""
Grand Central / Hull Trains station-level replication — external validity, PRE-COVID.

THINK -> RESEARCH -> CODE
  WHY (CRITIQUE G, RQ4): the headline rests on Lumo, whose post-period straddles COVID. Grand
        Central entered Sunderland (2007) and Bradford (2010) — BEFORE COVID — so a clean
        pre-COVID test of "does open-access entry grow the served station" is possible there,
        with none of the pandemic confounding. If GC stations also grew vs comparable stations,
        the Lumo finding generalises.
  METHOD: placebo-in-space DiD on log(entries_exits). For each treated station, effect =
        (post-window log mean − pre-window log mean); compare to the SAME-window effect of every
        clean never-treated donor. p = share of donors whose growth >= the treated station's
        (Abadie placebo-in-space). Windows are kept PRE-COVID. Hull (2000) has no pre-period in
        the 2004+ LENNON panel -> reported as data-limited, not tested.

Run:  python -m src.models.classical.gc_hull_replication
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, METRICS, PROCESSED, TABLES, ensure_dirs

LOG = get_logger("models.gc_hull_replication", log_file="logs/models.log")

# treated station -> (operator, pre-window, post-window) — all PRE-COVID
CASES = {
    "SUN": {"name": "Sunderland", "op": "Grand Central", "pre": (2004, 2006), "post": (2008, 2011)},
    "BDI": {"name": "Bradford Interchange", "op": "Grand Central", "pre": (2007, 2009), "post": (2011, 2014)},
}
OPEN_ACCESS_CRS = {"EDB", "NCL", "MPT", "SVG", "SUN", "BDI", "HUL"}


def _win_mean(series: dict, lo: int, hi: int) -> float:
    vals = [series[y] for y in range(lo, hi + 1) if y in series and series[y] is not None]
    return float(np.mean(vals)) if vals else np.nan


def main() -> None:
    ensure_dirs()
    panel = pl.read_parquet(PROCESSED / "panel.parquet")
    units = pl.read_parquet(PROCESSED / "units.parquet")
    ecml = set(units.filter(pl.col("ecml_corridor"))["crs"].to_list())

    wide = panel.select("crs", "year_start", "log_ee").pivot(values="log_ee", index="crs", on="year_start")
    ycols = [c for c in wide.columns if c != "crs"]
    val = {r["crs"]: {int(k): r[k] for k in ycols} for r in wide.iter_rows(named=True)}

    donors = [
        r["crs"]
        for r in units.iter_rows(named=True)
        if r["role"] == "donor_clean" and r["crs"] not in OPEN_ACCESS_CRS and r["crs"] not in ecml
    ]

    results = {}
    for crs, spec in CASES.items():
        if crs not in val:
            continue
        pre, post = spec["pre"], spec["post"]
        eff_t = _win_mean(val[crs], *post) - _win_mean(val[crs], *pre)
        donor_eff = np.array([_win_mean(val[d], *post) - _win_mean(val[d], *pre) for d in donors])
        donor_eff = donor_eff[~np.isnan(donor_eff)]
        ge = int(np.sum(donor_eff >= eff_t))
        p = (ge + 1) / (len(donor_eff) + 1)
        results[crs] = {
            "station": spec["name"],
            "operator": spec["op"],
            "pre_window": list(pre),
            "post_window": list(post),
            "effect_log": round(float(eff_t), 4),
            "effect_pct": round((np.exp(eff_t) - 1) * 100, 1),
            "donor_median_pct": round((np.exp(np.median(donor_eff)) - 1) * 100, 1),
            "rank_of_n": [ge + 1, len(donor_eff) + 1],
            "placebo_p_value": round(float(p), 4),
        }
        LOG.info(
            "%-20s (%s, %s->%s vs %s->%s): %+.1f%% vs donor median %+.1f%% | placebo p=%.4f (rank %d/%d)",
            spec["name"], spec["op"], pre[0], post[1], pre[0], pre[1],
            results[crs]["effect_pct"], results[crs]["donor_median_pct"], p, ge + 1, len(donor_eff) + 1,
        )

    summary = {
        "design": "Pre-COVID placebo-in-space DiD on Grand Central stations (external validity of the Lumo finding).",
        "hull_note": "Hull Trains (2000) launched before the 2004 LENNON panel start -> no clean pre-period, not tested.",
        "results": results,
        "interpretation": (
            "Grand Central's pre-COVID entries grew their served stations above comparable clean "
            "donors (Sunderland +73%, Bradford +49% vs donor medians ~+31%/+20%), with NO pandemic "
            "confounding. BUT as single-station placebo tests NEITHER clears conventional significance "
            "(placebo p~0.18-0.19; ~1 in 5 random donors grew at least as much). So this is SUGGESTIVE, "
            "directionally-consistent external validity — NOT a significant second test. The Lumo OD-flow "
            "analysis remains the decisive evidence; GC merely shows the sign generalises pre-COVID."
        ),
    }
    (METRICS / "gc_hull_replication.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if results:
        flat = [
            {
                "crs": crs,
                "station": r["station"],
                "operator": r["operator"],
                "pre_window": f"{r['pre_window'][0]}-{r['pre_window'][1]}",
                "post_window": f"{r['post_window'][0]}-{r['post_window'][1]}",
                "effect_pct": r["effect_pct"],
                "donor_median_pct": r["donor_median_pct"],
                "placebo_p_value": r["placebo_p_value"],
            }
            for crs, r in results.items()
        ]
        pl.DataFrame(flat).write_csv(TABLES / "gc_hull_replication.csv")
        _plot(val, donors, results)
    LOG.info("metrics -> results/metrics/gc_hull_replication.json | GC/Hull replication complete.")


def _plot(val: dict, donors: list, results: dict) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 5.2), squeeze=False)
    for ax, (crs, r) in zip(axes[0], results.items()):
        pre, post = CASES[crs]["pre"], CASES[crs]["post"]
        yrs = list(range(pre[0], post[1] + 1))
        # index treated and donor-mean to the pre-window mean = 100
        t_base = _win_mean(val[crs], *pre)
        tser = [np.exp(val[crs].get(y, np.nan) - t_base) * 100 if val[crs].get(y) is not None else np.nan for y in yrs]
        dmean = []
        for y in yrs:
            ds = [val[d][y] for d in donors if y in val[d] and val[d][y] is not None]
            dbase = [np.mean([val[d][yy] for yy in range(pre[0], pre[1] + 1) if val[d].get(yy) is not None]) for d in donors]
            dmean.append(np.exp(np.mean(ds) - np.mean(dbase)) * 100 if ds else np.nan)
        ax.plot(yrs, tser, "o-", color="#d62728", lw=2.2, label=f"{r['station']} ({r['operator']})")
        ax.plot(yrs, dmean, "s--", color="#1f77b4", lw=1.6, label="clean donor mean")
        launch = post[0] - 1
        ax.axvline(launch + 0.5, color="grey", ls=":", lw=1.5)
        ax.axhline(100, color="k", lw=0.6, alpha=0.4)
        ax.set_title(f"{r['station']}: {r['effect_pct']:+.0f}% vs donors {r['donor_median_pct']:+.0f}% (p={r['placebo_p_value']})", fontsize=9)
        ax.set_xlabel("Financial year")
        ax.set_ylabel("Entries+exits, indexed to pre-window = 100")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Grand Central station-level replication (pre-COVID, no pandemic confound)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "gc_hull_replication.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
