"""
CausalImpact-style Bayesian structural time-series counterfactual (Tier 2, modern SC).

THINK -> RESEARCH -> CODE
  WHAT: Put a principled credible interval on the strongest result — the operator-level
        ECML market growth. Model quarterly LNER journeys as a local-level structural
        time series with a regression on PEER long-distance operators (CrossCountry, GWR,
        East Midlands, TransPennine — not facing open-access entry). Fit on the
        pre-Lumo period; forecast the no-Lumo counterfactual from the peers' actual
        post values; the gap (with state-space prediction intervals) is the impact.
        This is Brodersen et al. (2015) CausalImpact in spirit (statsmodels state space).
  WHY : §6 Tier 2 (CausalImpact / BSTS) + formal UQ. Quarterly data (~43 pre quarters)
        gives real power, unlike the 14-point annual series. The peers absorb the common
        COVID shock so the post gap isolates the ECML-market deviation.
  CAVEAT: like all our market-level evidence, the gap reflects the ECML corridor's
        outperformance (LNER's own factors + Lumo), not Lumo alone — OD data needed to
        split. But it formalises "LNER+Lumo grew well beyond the peer-implied counterfactual".

Run:  python -m src.models.classical.causal_impact_operator
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, INTERIM, METRICS, TABLES, ensure_dirs

LOG = get_logger("models.causal_impact", log_file="logs/models.log")

CONTROLS = ["CrossCountry", "Great Western Railway", "East Midlands Railway", "TransPennine Express"]
TREAT = 2021.875  # Oct-Dec 2021 quarter (Lumo launched 25 Oct 2021)


def main() -> None:
    ensure_dirs()
    op = pd.read_parquet(INTERIM / "operator_journeys.parquet")
    q = op[(op.freq == "quarterly") & (op.flag == "observed")]
    wide = q.pivot_table(index="period_start", columns="series", values="value_m").sort_index()

    cols = ["London North Eastern Railway"] + CONTROLS
    d = wide[cols].dropna()
    lumo_q = wide["Lumo"].reindex(d.index).fillna(0.0)
    pre = d.index < TREAT
    n_pre, n_post = int(pre.sum()), int((~pre).sum())
    LOG.info("CausalImpact quarterly: %d pre, %d post quarters; controls=%s", n_pre, n_post, CONTROLS)

    y = np.log(d["London North Eastern Railway"].to_numpy())
    X = np.log(d[CONTROLS].to_numpy())
    mod = sm.tsa.UnobservedComponents(y[pre], level="local level", exog=X[pre])
    res = mod.fit(disp=False, maxiter=200)

    fc = res.get_forecast(steps=n_post, exog=X[~pre])
    cf_log = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    cf, cf_lo, cf_hi = np.exp(cf_log), np.exp(ci[:, 0]), np.exp(ci[:, 1])

    actual = d["London North Eastern Railway"].to_numpy()[~pre]
    lumo_post = lumo_q.to_numpy()[~pre]
    rel_lner = float(np.mean(actual / cf) - 1)
    rel_mkt = float(np.mean((actual + lumo_post) / cf) - 1)
    # prediction-interval-based bounds on the LNER relative effect
    rel_lner_lo = float(np.mean(actual / cf_hi) - 1)
    rel_lner_hi = float(np.mean(actual / cf_lo) - 1)
    cum_effect = float(np.sum(actual - cf))  # millions of journeys above counterfactual
    sig = bool(rel_lner_lo > 0)  # 95% interval excludes 0?

    summary = {
        "n_pre_quarters": n_pre,
        "n_post_quarters": n_post,
        "controls": CONTROLS,
        "LNER_rel_effect_pct": round(100 * rel_lner, 1),
        "LNER_rel_effect_ci95_pct": [round(100 * rel_lner_lo, 1), round(100 * rel_lner_hi, 1)],
        "LNER+Lumo_rel_effect_pct": round(100 * rel_mkt, 1),
        "cumulative_excess_journeys_m": round(cum_effect, 1),
        "lner_effect_95ci_excludes_zero": sig,
    }
    (METRICS / "m3_causal_impact_operator.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).T.to_csv(TABLES / "m3_causal_impact_operator.csv", header=["value"])
    LOG.info(
        "CausalImpact: LNER %+.1f%% [%.1f, %.1f] (95%% CI %s 0); LNER+Lumo %+.1f%%; cumulative +%.1fm journeys",
        100 * rel_lner,
        100 * rel_lner_lo,
        100 * rel_lner_hi,
        "excludes" if sig else "includes",
        100 * rel_mkt,
        cum_effect,
    )

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(11, 6))
    idx = d.index
    ax.plot(idx, d["London North Eastern Railway"], "o-", color="#d62728", ms=3, lw=1.8, label="LNER (observed)")
    post_idx = idx[~pre]
    ax.plot(post_idx, cf, "s--", color="#1f77b4", lw=1.8, label="BSTS counterfactual (no Lumo)")
    ax.fill_between(post_idx, cf_lo, cf_hi, color="#1f77b4", alpha=0.18, label="95% prediction interval")
    ax.plot(
        post_idx,
        d["London North Eastern Railway"].to_numpy()[~pre] + lumo_post,
        ":",
        color="#7b3294",
        lw=1.8,
        label="LNER + Lumo",
    )
    ax.axvline(TREAT, color="grey", ls=":", lw=1.4)
    ax.text(TREAT + 0.1, ax.get_ylim()[0], " Lumo (Oct 2021)", color="grey", fontsize=8, rotation=90, va="bottom")
    ax.set_xlabel("Year (quarterly)")
    ax.set_ylabel("Journeys (millions/quarter)")
    ax.set_title(
        f"CausalImpact (BSTS) — LNER vs peer-implied counterfactual\n"
        f"LNER {100 * rel_lner:+.1f}% [{100 * rel_lner_lo:.0f}, {100 * rel_lner_hi:.0f}] "
        f"(95% CI {'excludes' if sig else 'includes'} 0); cumulative +{cum_effect:.0f}m journeys",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "m3_causal_impact_operator.png", dpi=150)
    plt.close(fig)
    LOG.info("figure -> %s | CausalImpact complete.", FIGURES / "m3_causal_impact_operator.png")


if __name__ == "__main__":
    main()
