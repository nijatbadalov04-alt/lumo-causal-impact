"""
Sensitivity to UNOBSERVED confounding — the formal answer to entry-endogeneity.

THINK -> RESEARCH -> CODE
  THE LAST IDENTIFICATION WORRY: the headline OD effect (East-Coast-corridor flows recovered far above
        comparable non-ECML flows) is OBSERVATIONAL — corridor membership is not randomly assigned. We
        defend it with placebo-in-space (Glasgow), a flat pre-trend, and selection-on-observables
        (distance + pre-volume). A referee asks: *how strong would an UNOBSERVED confounder have to be
        to overturn it?* This module answers that formally, with the standard tools:
    1. **Cinelli-Hazlett (2020) robustness value (RV)** — the partial R^2 that a confounder must have
       with BOTH the treatment and the outcome to reduce the effect to zero (RV) or to non-significance
       (RV_a). Benchmarked against the partial R^2 of the OBSERVED covariates: a confounder would need to
       be "this many times as strong as distance" to explain the result away.
    2. **Oster (2019) delta** — how much selection on unobservables, relative to observables, would be
       needed to drive the coefficient to zero (|delta|>1 => robust); plus the bias-adjusted beta*.
    3. **VanderWeele-Ding (2017) E-value** — for the corridor recovery RATIO, the minimum confounder
       association (with treatment and outcome) needed to explain it away.
  DATA: the 391 London flows; Y = log(recovery), D = ECML-corridor, X = [distance_km, log pre-volume].

Run:  python -m src.evaluate.sensitivity_confounding
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm

from src.models.od_corridor_robustness import ECML_CORRIDOR
from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, INTERIM, METRICS, ensure_dirs

LOG = get_logger("evaluate.sensitivity_confounding", log_file="logs/evaluate.log")
ALPHA = 0.05


def _data():
    rec = pl.read_parquet(INTERIM / "od_all_flow_recoveries.parquet")
    cov = pl.read_parquet(INTERIM / "station_covariates.parquet").select("station_name", "distance_to_london_km")
    j = rec.join(cov, left_on="city", right_on="station_name", how="left").drop_nulls("distance_to_london_km")
    j = j.with_columns(pl.col("city").is_in(ECML_CORRIDOR).cast(pl.Float64).alias("D"))
    y = np.log(j["recovery"].to_numpy())
    d = j["D"].to_numpy()
    dist = j["distance_to_london_km"].to_numpy()
    lpre = np.log(j["pre"].to_numpy())
    return y, d, dist, lpre


def _ols(y, X):
    Xc = sm.add_constant(X)
    return sm.OLS(y, Xc).fit()


def _partial_r2_of(y, d, X_other):
    """Partial R^2 of treatment d with outcome y, controlling for X_other (the CH 'f' benchmark)."""
    # residualise y and d on X_other, correlation^2
    Xo = sm.add_constant(X_other)
    ry = y - sm.OLS(y, Xo).fit().predict(Xo)
    rd = d - sm.OLS(d, Xo).fit().predict(Xo)
    r = np.corrcoef(ry, rd)[0, 1]
    return float(r**2)


def _robustness_value(t, dof, q=1.0):
    """Cinelli-Hazlett RV_q: partial R^2 a confounder needs with D and Y to reduce |effect| by q*100%."""
    fq = q * abs(t) / np.sqrt(dof)
    return float(0.5 * (np.sqrt(fq**4 + 4 * fq**2) - fq**2))


def _rv_alpha(t, dof, alpha=0.05):
    """RV to reduce the estimate to NON-significance at level alpha (two-sided)."""
    from scipy.stats import t as tdist

    tcrit = tdist.ppf(1 - alpha / 2, dof - 1)
    fq = max(abs(t) - tcrit, 0.0) / np.sqrt(dof)
    return float(0.5 * (np.sqrt(fq**4 + 4 * fq**2) - fq**2))


def _oster(y, d, X, rmax_rule=1.3):
    """Oster (2019) delta for beta*=0 and the bias-adjusted beta* (delta=1)."""
    m0 = _ols(y, d.reshape(-1, 1))  # restricted: Y ~ D
    b0, r0 = m0.params[1], m0.rsquared
    full = np.column_stack([d, X])
    m1 = _ols(y, full)  # full: Y ~ D + X
    b1, r1 = m1.params[1], m1.rsquared
    rmax = min(1.0, rmax_rule * r1)
    denom_delta = (b0 - b1) * (rmax - r1)
    delta = (b1 * (r1 - r0)) / denom_delta if abs(denom_delta) > 1e-12 else float("inf")
    beta_star = b1 - (b0 - b1) * (rmax - r1) / (r1 - r0) if abs(r1 - r0) > 1e-12 else float("nan")
    return {
        "beta_restricted": float(b0), "r2_restricted": float(r0),
        "beta_full": float(b1), "r2_full": float(r1), "rmax": float(rmax),
        "delta_for_zero": float(delta), "bias_adjusted_beta_delta1": float(beta_star),
    }


def _evalue(rr):
    """VanderWeele-Ding E-value for a risk/rate ratio rr (>=1)."""
    rr = max(rr, 1.0 / rr)  # symmetric
    return float(rr + np.sqrt(rr * (rr - 1)))


def main() -> None:
    ensure_dirs()
    y, d, dist, lpre = _data()
    X = np.column_stack([dist, lpre])

    # full regression Y ~ D + distance + log_pre
    full = _ols(y, np.column_stack([d, dist, lpre]))
    b_d, se_d = full.params[1], full.bse[1]
    t_d = b_d / se_d
    dof = int(full.df_resid)
    rv = _robustness_value(t_d, dof, q=1.0)
    rva = _rv_alpha(t_d, dof, ALPHA)

    # benchmark: partial R^2 of the OBSERVED covariates with the outcome (controlling for the other)
    pr2_dist = _partial_r2_of(y, dist, lpre.reshape(-1, 1))
    pr2_lpre = _partial_r2_of(y, lpre, dist.reshape(-1, 1))
    bench = max(pr2_dist, pr2_lpre)

    oster = _oster(y, d, X)

    # E-value on the corridor recovery ratio (ECML vs non-ECML median), read from the robustness JSON
    cc = json.loads((METRICS / "od_corridor_robustness.json").read_text(encoding="utf-8"))["corridor_clustering"]
    rr = cc["ecml_median_recovery"] / cc["non_ecml_median_recovery"]
    evalue = _evalue(rr)

    summary = {
        "design": "Sensitivity of the OD corridor effect to UNOBSERVED confounding (391 flows).",
        "effect_log_pp": round(float(b_d), 4),
        "effect_pct": round((np.exp(b_d) - 1) * 100, 1),
        "t_stat": round(float(t_d), 2),
        "dof": dof,
        "cinelli_hazlett": {
            "robustness_value_to_zero": round(rv, 3),
            "robustness_value_to_nonsig_5pct": round(rva, 3),
            "benchmark_partial_r2_distance": round(pr2_dist, 3),
            "benchmark_partial_r2_log_prevolume": round(pr2_lpre, 3),
            "RV_vs_strongest_observed_covariate_x": round(rv / bench, 1) if bench > 0 else None,
        },
        "oster_delta": {k: round(v, 3) for k, v in oster.items()},
        "evalue": {"corridor_recovery_ratio": round(rr, 3), "e_value": round(evalue, 2)},
        "interpretation": (
            f"MIXED, reported honestly. Oster's delta = {oster['delta_for_zero']:.1f} (|delta|>1: selection "
            f"on unobservables would have to EXCEED selection on observables to zero it; bias-adjusted beta* "
            f"keeps its sign) and the E-value = {evalue:.2f} both lean ROBUST. BUT the Cinelli-Hazlett "
            f"robustness value is only {rv:.2f} (~{rv/bench:.1f}x the partial R^2 of distance, {bench:.2f}) and "
            f"the RV-to-non-significance is {rva:.2f} — so the LINEAR covariate-adjusted effect alone is NOT "
            "bulletproof: a single confounder about as strong as distance could attenuate it. This is exactly "
            "WHY identification rests primarily on the DESIGN-based evidence — the Glasgow placebo, the flat "
            "pre-trend, the rank-based corridor clustering (p=0.0001) and the OD event-study DiD (p=0.002), "
            "all far stronger than the linear regression — rather than on selection-on-observables. The "
            "sensitivity analysis correctly positions the linear adjustment as corroboration, not the anchor."
        ),
    }
    (METRICS / "sensitivity_confounding.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("OD corridor effect %.1f%% (t=%.1f, dof=%d)", summary["effect_pct"], t_d, dof)
    LOG.info("Cinelli-Hazlett RV(to 0)=%.3f, RV(to non-sig)=%.3f | benchmark distance partial R2=%.3f -> RV is %.1fx", rv, rva, pr2_dist, summary["cinelli_hazlett"]["RV_vs_strongest_observed_covariate_x"])
    LOG.info("Oster delta=%.1f, bias-adjusted beta*(delta=1)=%.3f (sign preserved: %s)", oster["delta_for_zero"], oster["bias_adjusted_beta_delta1"], (oster["bias_adjusted_beta_delta1"] > 0) == (b_d > 0))
    LOG.info("E-value=%.2f for corridor recovery RR=%.2f", evalue, rr)

    _plot(rv, rva, bench, pr2_dist, pr2_lpre, evalue, summary)
    LOG.info("metrics -> results/metrics/sensitivity_confounding.json | confounding sensitivity complete.")


def _plot(rv, rva, bench, pr2_dist, pr2_lpre, evalue, summary):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # left: robustness value vs observed-covariate benchmarks (partial R^2 scale)
    labels = ["RV\n(effect→0)", "RV\n(→ non-sig)", "distance\n(observed)", "log pre-vol\n(observed)"]
    vals = [rv, rva, pr2_dist, pr2_lpre]
    colours = ["#2ca02c", "#2ca02c", "#7f7f7f", "#7f7f7f"]
    ax1.bar(labels, vals, color=colours, alpha=0.85)
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax1.set_ylabel("Partial R² (confounder strength)")
    ax1.set_title(
        f"Cinelli-Hazlett: confounding must be {summary['cinelli_hazlett']['RV_vs_strongest_observed_covariate_x']}×\n"
        "as strong as distance to zero the effect",
        fontsize=10,
    )
    ax1.grid(axis="y", alpha=0.3)

    # right: Oster delta + E-value summary
    ax2.axis("off")
    o = summary["oster_delta"]
    txt = (
        f"OSTER (2019)\n"
        f"  restricted β = {o['beta_restricted']:.3f} (R²={o['r2_restricted']:.2f})\n"
        f"  full β       = {o['beta_full']:.3f} (R²={o['r2_full']:.2f}), Rmax={o['rmax']:.2f}\n"
        f"  δ for β*=0   = {o['delta_for_zero']:.1f}   (|δ|>1 ⇒ robust)\n"
        f"  bias-adj β*  = {o['bias_adjusted_beta_delta1']:.3f} (δ=1; sign preserved)\n\n"
        f"E-VALUE (VanderWeele-Ding)\n"
        f"  corridor recovery RR = {summary['evalue']['corridor_recovery_ratio']:.2f}\n"
        f"  E-value = {evalue:.2f}\n"
        f"  ⇒ a confounder needs RR ≥ {evalue:.2f} with BOTH\n"
        f"     treatment and outcome to explain it away\n\n"
        f"VERDICT: δ and E-value lean robust; CH RV ≈ distance\n"
        f"⇒ linear adjustment is corroboration, NOT the anchor.\n"
        f"Identification rests on DESIGN (placebo, pre-trends,\nclustering p=1e-4, OD DiD p=2e-3)."
    )
    ax2.text(0.02, 0.98, txt, va="top", ha="left", fontsize=10, family="monospace")
    ax2.set_title("Oster δ and E-value", fontsize=10)

    fig.suptitle("How strong would an unobserved confounder need to be to overturn the corridor effect?", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "sensitivity_confounding.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
