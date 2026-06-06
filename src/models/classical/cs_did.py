"""
Callaway & Sant'Anna (2021) staggered-adoption DiD — group-time ATTs.

THINK -> RESEARCH -> CODE
  WHY (brief Tier 1, named): open-access entry is STAGGERED — Grand Central Sunderland (2007),
        GC Bradford (2010), Lumo (2021). A single TWFE event-study with heterogeneous effects
        across cohorts suffers the negative-weighting problem (Goodman-Bacon 2021): already-
        treated units act as controls. Callaway-Sant'Anna fixes this with clean 2x2 group-time
        ATT(g,t): each treated COHORT is compared only to NEVER-TREATED units, from a clean base
        period, then aggregated.
  COVID-ROBUST BASELINE: CS uses g-1 as the base period, but for the Lumo cohort g-1 = 2020 is the
        COVID trough -> a spuriously low base. We anchor post-2020 cohorts at base = 2019 (the last
        clean pre-COVID year); GC cohorts (2007/2010) use the standard g-1. Documented honestly.
  OUTCOME: log(entries_exits). COMPARISON: never-treated clean donors, with ALL open-access-served
        CRS removed from controls (Sunderland/Hull were mislabeled donor_clean) + ECML corridor
        excluded. INFERENCE: cluster bootstrap over control units.

Run:  python -m src.models.classical.cs_did
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

LOG = get_logger("models.cs_did", log_file="logs/models.log")

# treated CRS -> cohort (treat_year_start). Hull (2000) has no pre-period in the 2004+ panel -> excluded.
COHORTS = {"SUN": 2007, "BDI": 2010, "EDB": 2021, "NCL": 2021, "MPT": 2021, "SVG": 2021}
OPEN_ACCESS_CRS = set(COHORTS) | {"HUL"}  # never allow these in the control pool
LAST_CLEAN_PRE_COVID = 2019
N_BOOT = 2000
RNG_SEED = 20211025


def _base_year(g: int) -> int:
    """CS base period g-1, but anchored at 2019 for post-COVID cohorts (avoid the 2020 trough)."""
    return min(g - 1, LAST_CLEAN_PRE_COVID)


def main() -> None:
    ensure_dirs()
    panel = pl.read_parquet(PROCESSED / "panel.parquet")
    units = pl.read_parquet(PROCESSED / "units.parquet")
    ecml = set(units.filter(pl.col("ecml_corridor"))["crs"].to_list())

    # outcome matrix: log_ee by crs x year
    wide = panel.select("crs", "year_start", "log_ee").pivot(values="log_ee", index="crs", on="year_start")
    ycols = [c for c in wide.columns if c != "crs"]
    yrs = sorted(int(c) for c in ycols)
    val = {r["crs"]: {int(k): r[k] for k in ycols} for r in wide.iter_rows(named=True)}

    # clean never-treated control pool
    control_crs = [
        r["crs"]
        for r in units.iter_rows(named=True)
        if r["role"] == "donor_clean" and r["crs"] not in OPEN_ACCESS_CRS and r["crs"] not in ecml
    ]
    LOG.info("CS-DiD: %d controls (clean, never-treated, open-access+ECML removed)", len(control_crs))

    def ctrl_mean(year: int, crs_subset: list[str]) -> float:
        vals = [val[c][year] for c in crs_subset if c in val and year in val[c] and val[c][year] is not None]
        return float(np.mean(vals)) if vals else np.nan

    # ---- group-time ATT(g,t) ----
    cohort_years = sorted(set(COHORTS.values()))
    rows = []
    rng = np.random.default_rng(RNG_SEED)
    boot_idx = [rng.choice(len(control_crs), size=len(control_crs), replace=True) for _ in range(N_BOOT)]

    att_gt = {}
    for g in cohort_years:
        base = _base_year(g)
        treated = [c for c in COHORTS if COHORTS[c] == g and c in val]
        for t in yrs:
            if t <= base:
                continue
            # treated change from base to t (avg over cohort units)
            d_tr = np.mean([val[c][t] - val[c][base] for c in treated if val[c].get(t) is not None and val[c].get(base) is not None])
            d_co = ctrl_mean(t, control_crs) - ctrl_mean(base, control_crs)
            att = float(d_tr - d_co)
            # bootstrap SE over control units
            boot = []
            cm_base = np.array([val[c][base] for c in control_crs])
            cm_t = np.array([val[c][t] for c in control_crs])
            for bi in boot_idx:
                d_co_b = cm_t[bi].mean() - cm_base[bi].mean()
                boot.append(d_tr - d_co_b)
            se = float(np.std(boot))
            att_gt[(g, t)] = (att, se)
            rows.append({"cohort": g, "year": t, "event_time": t - g, "att_log": round(att, 4), "att_pct": round((np.exp(att) - 1) * 100, 1), "se": round(se, 4), "n_treated": len(treated)})

    pl.DataFrame(rows).write_csv(TABLES / "cs_did_group_time.csv")

    # ---- aggregations ----
    # overall ATT = average of SHORT-RUN post ATT(g,t), event time e in [0,3]. We cap the horizon
    # so the early (2007/2010) cohorts' windows stay pre-COVID-clean rather than spanning 17 years
    # across the pandemic; this is the meaningful staggered-adoption summary.
    post = [(g, t) for (g, t) in att_gt if 0 <= t - g <= 3]
    weights = np.array([sum(1 for c in COHORTS if COHORTS[c] == g) for (g, t) in post], dtype=float)
    atts = np.array([att_gt[(g, t)][0] for (g, t) in post])
    overall = float(np.average(atts, weights=weights))
    # bootstrap overall
    ov_boot = []
    for j in range(N_BOOT):
        vals = []
        for g, t in post:
            base = _base_year(g)
            treated = [c for c in COHORTS if COHORTS[c] == g and c in val]
            d_tr = np.mean([val[c][t] - val[c][base] for c in treated])
            cm_base = np.array([val[c][base] for c in control_crs])
            cm_t = np.array([val[c][t] for c in control_crs])
            bi = boot_idx[j]
            vals.append(d_tr - (cm_t[bi].mean() - cm_base[bi].mean()))
        ov_boot.append(np.average(vals, weights=weights))
    overall_se = float(np.std(ov_boot))

    # HONEST few-treated-units inference: randomization inference. Assign the SAME cohort
    # structure (1 unit @2007, 1 @2010, 4 @2021) to random never-treated control stations and
    # recompute the overall ATT. The control-only bootstrap SE above understates uncertainty
    # because it ignores treated-side sampling; this placebo test does not.
    cohort_sizes = [(g, sum(1 for c in COHORTS if COHORTS[c] == g)) for g in cohort_years]
    ctrl_year_mean = {y: ctrl_mean(y, control_crs) for y in yrs}  # full-control mean (≈ rest; 6/2338 negligible)
    rng_ri = np.random.default_rng(RNG_SEED + 1)
    ri_null = []
    for _ in range(N_BOOT):
        picked = rng_ri.choice(len(control_crs), size=len(COHORTS), replace=False)
        fake, i = {g: [] for g in cohort_years}, 0
        for g, sz in cohort_sizes:
            for _s in range(sz):
                fake[g].append(control_crs[picked[i]])
                i += 1
        vals, ws = [], []
        for g, t in post:
            base = _base_year(g)
            d_tr = np.mean([val[c][t] - val[c][base] for c in fake[g]])
            vals.append(d_tr - (ctrl_year_mean[t] - ctrl_year_mean[base]))
            ws.append(len(fake[g]))
        ri_null.append(np.average(vals, weights=ws))
    ri_null = np.array(ri_null)
    ri_p = float((np.sum(np.abs(ri_null) >= abs(overall)) + 1) / (len(ri_null) + 1))

    # dynamic (event-study) ATT(e): average ATT(g, g+e) across cohorts sharing event-time e
    ev = {}
    for e in range(-4, 4):
        keys = [(g, g + e) for g in cohort_years if (g, g + e) in att_gt]
        if keys:
            a = np.mean([att_gt[k][0] for k in keys])
            s = np.sqrt(np.mean([att_gt[k][1] ** 2 for k in keys]))  # approx
            ev[e] = (float(a), float(s))

    # Lumo-only event study (the main 4-unit cohort, COVID-robust base 2019)
    lumo_ev = {t - 2021: att_gt[(2021, t)] for t in yrs if (2021, t) in att_gt}

    summary = {
        "design": "Callaway-Sant'Anna group-time ATT; never-treated clean controls; cluster bootstrap.",
        "cohorts": {str(g): [c for c in COHORTS if COHORTS[c] == g] for g in cohort_years},
        "covid_robust_base_year": {str(g): _base_year(g) for g in cohort_years},
        "n_controls": len(control_crs),
        "overall_att_log": round(overall, 4),
        "overall_att_pct": round((np.exp(overall) - 1) * 100, 1),
        "overall_att_control_bootstrap_se": round(overall_se, 4),
        "overall_att_randomization_p": round(ri_p, 4),
        "se_caveat": "control-bootstrap SE ignores few-treated-units uncertainty; trust the randomization p.",
        "dynamic_att_pct": {str(e): round((np.exp(v[0]) - 1) * 100, 1) for e, v in ev.items()},
        "dynamic_att_caveats": (
            "e=-1 is ONLY the Lumo cohort's 2020 (COVID trough), not a parallel-trends test; there are "
            "no pooled pre-trend points -> the design CANNOT test parallel trends. Cohorts are not on a "
            "common calendar clock (Lumo's e=0 spans 2019->2021 incl. COVID; GC's e=0 is a 1-yr change), "
            "so read per-cohort series, not the pooled dynamic, as the clean object."
        ),
        "lumo_event_study_pct": {str(e): round((np.exp(v[0]) - 1) * 100, 1) for e, v in lumo_ev.items()},
        "cohort_att_pct_at_e_plus1": {
            str(g): round((np.exp(att_gt[(g, g + 1)][0]) - 1) * 100, 1) for g in cohort_years if (g, g + 1) in att_gt
        },
        "interpretation": (
            "Staggered-adoption ATT free of TWFE negative weighting (Goodman-Bacon). The open-access "
            "cohorts show a POSITIVE station-total ATT (Lumo cohort +14.5% first-post; short-run "
            "overall +16%), but it is NOT significant under honest few-treated-units randomization "
            "inference (p~0.15) — consistent with the rest of the project: the station-total signal "
            "is positive but not robustly identified (4-6 treated units + multi-market dilution), and "
            "the DECISIVE significant evidence is the OD-flow analysis. CS rules out the TWFE "
            "negative-weighting artefact as the reason the station-total effect looks small."
        ),
    }
    (METRICS / "cs_did.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("CS overall ATT = %.1f%% (randomization p=%.4f; control-boot SE %.3f understates)", summary["overall_att_pct"], ri_p, overall_se)
    LOG.info("CS dynamic ATT(e) %%: %s", summary["dynamic_att_pct"])
    LOG.info("CS cohort ATT at e=+1 %%: %s", summary["cohort_att_pct_at_e_plus1"])

    _plot(ev, lumo_ev)
    LOG.info("metrics -> results/metrics/cs_did.json | Callaway-Sant'Anna complete.")


def _plot(ev: dict, lumo_ev: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for series, colour, label in [(ev, "#1f77b4", "all cohorts (pooled)"), (lumo_ev, "#d62728", "Lumo cohort (2021)")]:
        es = sorted(series.keys())
        pct = [(np.exp(series[e][0]) - 1) * 100 for e in es]
        lo = [(np.exp(series[e][0] - 1.96 * series[e][1]) - 1) * 100 for e in es]
        hi = [(np.exp(series[e][0] + 1.96 * series[e][1]) - 1) * 100 for e in es]
        ax.plot(es, pct, "o-", color=colour, lw=2, label=label)
        ax.fill_between(es, lo, hi, color=colour, alpha=0.15)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(-0.5, color="grey", ls=":", lw=1.5, label="entry")
    # the only pre-entry point (e=-1) is the Lumo cohort's 2020 = COVID trough, NOT a clean
    # parallel-trends test (GC cohorts have no pooled pre-points here) -> flag it honestly
    if -1 in ev:
        ax.annotate("e=-1 is Lumo's 2020\n(COVID, not a pre-trend test)", (-1, (np.exp(ev[-1][0]) - 1) * 100),
                    fontsize=7, color="grey", xytext=(-0.9, -38), textcoords="data",
                    arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))
    ax.set_xlabel("Event time (years since entry; cohorts NOT on a common calendar clock)")
    ax.set_ylabel("ATT vs never-treated controls (%)")
    ax.set_title(
        "Callaway-Sant'Anna staggered-adoption DiD\n(group-time ATT; design cannot test pre-trends — see note)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "cs_did_event_study.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
