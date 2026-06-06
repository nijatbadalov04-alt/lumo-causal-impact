"""
Method-agreement triangulation (the method-comparison deliverable).

THINK -> RESEARCH -> CODE
  WHAT: Consolidate every Lumo estimate produced so far into ONE table + forest plot:
        naive off-ECML SC, within-corridor SC, deep counterfactual (with conformal
        interval), and the operator-level market result. Triangulation is the point —
        agreement = robust; disagreement = a finding we explain (it is: the naive SC is
        confounded; the others converge on ~no station-total effect; the market signal
        is operator-level).
  WHY : §8 'method triangulation' + §10 'method-comparison table'. Reads existing
        result JSONs only — no re-fitting.

Run:  python -m src.evaluate.triangulation
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, METRICS, TABLES

LOG = get_logger("evaluate.triangulation", log_file="logs/models.log")
NAMES = {"NCL": "Newcastle", "EDB": "Edinburgh", "MPT": "Morpeth", "SVG": "Stevenage"}


def _load(p):
    return json.loads((METRICS / p).read_text(encoding="utf-8")) if (METRICS / p).exists() else None


def main() -> None:
    sc = _load("m3_synthetic_control.json") or {}
    wc = {r["crs"]: r for r in (_load("m3_within_corridor.json") or [])}
    aug = {r["crs"]: r for r in (_load("m3_augmented_sc.json") or [])}
    deep = (_load("m4_deep_counterfactual.json") or {}).get("effects", {})
    op = _load("m3_operator_analysis.json") or {}
    did = _load("m3_did_event_study.json") or {}
    ci = _load("m3_causal_impact_operator.json") or {}

    rows = []
    for crs in ["NCL", "SVG", "MPT", "EDB"]:
        rows.append(
            {
                "crs": crs,
                "station": NAMES[crs],
                "naive_sc_pct": sc.get(crs, {}).get("avg_post_effect_pct"),
                "naive_sc_p": sc.get(crs, {}).get("placebo_p_value"),
                "within_corridor_pct": wc.get(crs, {}).get("within_corridor_effect_pct"),
                "within_corridor_p": wc.get(crs, {}).get("placebo_p"),
                "augmented_sc_pct": aug.get(crs, {}).get("augmented_sc_effect_pct"),
                "deep_pct": deep.get(crs, {}).get("deep_effect_pct"),
                "deep_conf_lo": deep.get(crs, {}).get("conformal_lo_pct"),
                "deep_conf_hi": deep.get(crs, {}).get("conformal_hi_pct"),
            }
        )
    tbl = pl.DataFrame(rows)
    tbl.write_csv(TABLES / "triangulation_method_comparison.csv")
    LOG.info("method comparison (station-total effect %%):")
    for r in rows:
        LOG.info(
            "  %-10s naive_SC=%s  within_corridor=%s  deep=%s [%s,%s]",
            r["station"],
            r["naive_sc_pct"],
            r["within_corridor_pct"],
            r["deep_pct"],
            r["deep_conf_lo"],
            r["deep_conf_hi"],
        )
    if op.get("lner_synthetic_control"):
        o = op["lner_synthetic_control"]
        LOG.info(
            "OPERATOR-LEVEL (market): LNER vs peers %s%%, LNER+Lumo %s%% (the real signal)",
            o.get("LNER_vs_peers_pct"),
            o.get("LNER+Lumo_vs_peers_pct"),
        )

    # overall / market-level estimators (not per-station) for the full comparison record
    o = op.get("lner_synthetic_control") or {}
    overall = {
        "did_within_corridor_avg_post_pct": did.get("avg_post_effect_pct"),
        "did_parallel_trends_ok": did.get("parallel_trends_pre_all_ns"),
        "operator_DiD_in_ratios_LNER_pct": o.get("LNER_vs_peers_pct"),
        "operator_DiD_in_ratios_LNER_Lumo_pct": o.get("LNER+Lumo_vs_peers_pct"),
        "causal_impact_BSTS_LNER_pct": ci.get("LNER_rel_effect_pct"),
        "causal_impact_BSTS_ci95": ci.get("LNER_rel_effect_ci95_pct"),
    }
    (METRICS / "triangulation_summary.json").write_text(
        json.dumps({"per_station": rows, "overall_and_market": overall}, indent=2), encoding="utf-8"
    )
    LOG.info(
        "OVERALL within-corridor DiD %s%% (parallel-trends ok=%s); MARKET DiD-in-ratios LNER %s%% / "
        "LNER+Lumo %s%%; BSTS LNER %s%% CI%s",
        overall["did_within_corridor_avg_post_pct"],
        overall["did_parallel_trends_ok"],
        overall["operator_DiD_in_ratios_LNER_pct"],
        overall["operator_DiD_in_ratios_LNER_Lumo_pct"],
        overall["causal_impact_BSTS_LNER_pct"],
        overall["causal_impact_BSTS_ci95"],
    )

    # ---- forest plot ----
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = [
        ("naive_sc_pct", "Naive SC (off-ECML) — confounded", "#999999", None, None),
        ("within_corridor_pct", "Within-corridor SC (controls corridor)", "#1b7837", None, None),
        ("deep_pct", "Deep counterfactual (conformal-90)", "#2166ac", "deep_conf_lo", "deep_conf_hi"),
    ]
    stations = [r["station"] for r in rows]
    y0 = np.arange(len(rows))[::-1]
    off = {0: 0.25, 1: 0.0, 2: -0.25}
    for mi, (key, label, color, lo, hi) in enumerate(methods):
        ys = y0 + off[mi]
        xs = [r[key] for r in rows]
        ax.scatter(xs, ys, color=color, s=55, label=label, zorder=3)
        if lo:
            for r, y in zip(rows, ys):
                if r[lo] is not None:
                    ax.plot([r[lo], r[hi]], [y, y], color=color, lw=2, alpha=0.6, zorder=2)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y0)
    ax.set_yticklabels(stations)
    ax.set_xlabel("Estimated effect on STATION-TOTAL entries+exits (%)")
    ax.set_title(
        "Triangulation — Lumo station-total effect shrinks once the corridor is controlled\n"
        "Naive SC overstates (confounded); within-corridor & deep converge near 0. "
        "Market signal is operator-level (LNER+Lumo grew).",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIGURES / "triangulation_forest.png", dpi=150)
    plt.close(fig)
    LOG.info("figure -> %s", FIGURES / "triangulation_forest.png")
    LOG.info("triangulation complete.")


if __name__ == "__main__":
    main()
