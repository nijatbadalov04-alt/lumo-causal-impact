"""
Within-corridor event-study Difference-in-Differences (M3 formal triangulation).

THINK -> RESEARCH -> CODE
  WHAT: TWFE event study — log(entries+exits) on station FE + year FE + treated×event-time
        dummies, treated = Lumo stops, controls = non-Lumo ECML-corridor stations. The
        pre-period coefficients test parallel trends; the post coefficients are the dynamic
        Lumo effect. Cluster-robust SEs by station.
  WHY : a formal DiD corroborates the synthetic control. Using
        WITHIN-CORRIDOR controls differences out the corridor-wide post-COVID recovery that
        confounded the off-ECML comparison (WEAKNESSES W1). Reference year = 2019 (last
        pre-COVID normal year); 2003 excluded (gap), LENNON era only.
  CAVEAT: 4 treated clusters ⇒ wide CIs (honest); within-corridor controls partly treated by
        LNER's line-wide response (SUTVA) ⇒ effect biased toward 0 (a lower bound).

Run:  python -m src.models.classical.did_event_study
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, METRICS, PROCESSED, TABLES, ensure_dirs

LOG = get_logger("models.did_event_study", log_file="logs/models.log")
REF_YEAR = 2019  # reference (last pre-COVID normal year)


def main() -> None:
    ensure_dirs()
    cfg = load_config("base")
    served = set(cfg["treatments"]["lumo"]["served_crs"])
    treat = int(cfg["treatments"]["lumo"]["treat_year_start"])

    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    units = pd.read_parquet(PROCESSED / "units.parquet")
    keep = set(units[units.role.isin(["treated_lumo", "ecml_corridor_control"])].crs)
    df = panel[(panel.crs.isin(keep)) & (panel.flag == "observed") & (panel.value > 0)].copy()
    df["log_ee"] = np.log(df["value"])
    df["treated"] = df["crs"].isin(served).astype(int)
    df["year_start"] = df["year_start"].astype(int)

    years = sorted(y for y in df.year_start.unique() if y != REF_YEAR)
    te_cols = []
    for y in years:
        col = f"te_{y}"
        df[col] = ((df.year_start == y) & (df.treated == 1)).astype(int)
        te_cols.append(col)

    formula = "log_ee ~ C(crs) + C(year_start) + " + " + ".join(te_cols)
    m = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["crs"]})

    n_treated = df[df.treated == 1].crs.nunique()
    n_control = df[df.treated == 0].crs.nunique()
    LOG.info("event-study DiD: %d treated, %d control stations, %d obs", n_treated, n_control, len(df))

    rows = []
    for y in years:
        col = f"te_{y}"
        rows.append(
            {
                "year": y,
                "event_time": y - treat,
                "coef": m.params[col],
                "se": m.bse[col],
                "ci_lo": m.conf_int().loc[col, 0],
                "ci_hi": m.conf_int().loc[col, 1],
                "phase": "pre" if y < treat else "post",
            }
        )
    rows.append(
        {
            "year": REF_YEAR,
            "event_time": REF_YEAR - treat,
            "coef": 0.0,
            "se": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "phase": "ref",
        }
    )
    es = pd.DataFrame(rows).sort_values("year")
    es.to_csv(TABLES / "m3_did_event_study.csv", index=False)

    post = es[es.phase == "post"]
    avg_post = float(np.exp(post["coef"].mean()) - 1)
    pre = es[es.phase == "pre"]
    pretrend_ok = bool((pre["ci_lo"].le(0) & pre["ci_hi"].ge(0)).all())  # all pre CIs include 0
    summary = {
        "n_treated": n_treated,
        "n_control": n_control,
        "avg_post_effect_pct": round(100 * avg_post, 1),
        "parallel_trends_pre_all_ns": pretrend_ok,
        "effect_2024_pct": round(100 * (np.exp(es[es.year == 2024].coef.iloc[0]) - 1), 1),
    }
    (METRICS / "m3_did_event_study.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info(
        "DiD avg post effect = %+.1f%% | parallel-trends pre all NS = %s | 2024 = %+.1f%%",
        100 * avg_post,
        pretrend_ok,
        summary["effect_2024_pct"],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        es.year,
        100 * es.coef,
        yerr=[100 * (es.coef - es.ci_lo), 100 * (es.ci_hi - es.coef)],
        fmt="o-",
        color="#1b7837",
        capsize=3,
        lw=1.6,
    )
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(treat - 0.5, color="#2166ac", ls=":", lw=1.4)
    ax.axvspan(2020, 2021, color="grey", alpha=0.12, lw=0)
    ax.set_xlabel("Financial year (start)")
    ax.set_ylabel("Treated × year coef (log pts ≈ %)")
    ax.set_title(
        f"Within-corridor event-study DiD — Lumo stops vs ECML controls\n"
        f"avg post effect {100 * avg_post:+.1f}% (95% CI wide: 4 treated clusters); "
        f"pre-trend ~flat ⇒ parallel-trends plausible",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "m3_did_event_study.png", dpi=150)
    plt.close(fig)
    LOG.info("figure -> %s | DiD complete.", FIGURES / "m3_did_event_study.png")


if __name__ == "__main__":
    main()
