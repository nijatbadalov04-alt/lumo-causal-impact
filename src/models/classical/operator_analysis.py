"""
Operator-level substitution-vs-creation analysis (RQ1, the clean lever).

THINK -> RESEARCH -> CODE
  WHAT: Using ORR Table 1223 annual journeys, test whether Lumo's entry CANNIBALISED
        LNER (substitution) or grew the ECML market (creation), by (1) recovery-ratio
        comparison vs comparable long-distance operators, and (2) a synthetic control
        for LNER built from peer long-distance operators (donors absorb COVID).
  LOGIC: LNER ≈ East Coast franchise; Lumo is ECML-only ⇒ LNER+Lumo ≈ ECML market.
        - Substitution ⇒ LNER falls ~by Lumo's volume; LNER+Lumo flat vs counterfactual.
        - Creation     ⇒ LNER holds (≈ peers' recovery) AND Lumo is additive ⇒
          LNER+Lumo ABOVE the no-Lumo counterfactual.
  DONORS (peer long-distance, NOT facing open-access entry on their core): CrossCountry,
        Great Western, East Midlands, TransPennine. **Avanti excluded** (its own 2022-23
        collapse would bias the counterfactual) — used only as a sensitivity row.
  CAVEAT: only ~14 annual points; operator markets differ. Treated as corroborating
        evidence triangulated with the (forthcoming) ODM, not a sole proof.

Run:  python -m src.models.classical.operator_analysis
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, INTERIM, METRICS, TABLES, ensure_dirs

LOG = get_logger("models.operator_analysis", log_file="logs/models.log")

PEERS = ["CrossCountry", "Great Western Railway", "East Midlands Railway", "TransPennine Express"]
TREAT_YEAR = 2021
BASE = 2019  # pre-COVID reference


def main() -> None:
    ensure_dirs()
    op = pd.read_parquet(INTERIM / "operator_journeys.parquet")
    a = op[(op.freq == "annual") & (op.flag == "observed")]
    wide = a.pivot_table(index="period_start", columns="series", values="value_m").sort_index()
    wide["LNER+Lumo"] = wide["London North Eastern Railway"].add(wide["Lumo"].fillna(0))

    LOG.info(
        "LNER annual series: %s",
        {int(y): round(v, 1) for y, v in wide["London North Eastern Railway"].dropna().items()},
    )

    # ---- (1) recovery ratios vs 2019 ----
    def recov(series):
        s = wide[series]
        if BASE in s.index and 2024 in s.index and pd.notna(s.get(BASE)) and pd.notna(s.get(2024)):
            return float(s[2024] / s[BASE])
        return np.nan

    recov_tbl = {
        s: round(recov(s), 3)
        for s in ["London North Eastern Railway", "LNER+Lumo", *PEERS, "Avanti West Coast", "Total"]
    }
    LOG.info("recovery ratio (2024/2019): %s", recov_tbl)

    # ---- (2) DiD-in-ratios counterfactual (level-correct) ----
    # Convex SC fails here: operators have different absolute journey LEVELS, so a
    # convex combination of donor levels is not level-matched to LNER. Instead we
    # build the no-Lumo counterfactual as LNER's own 2019 level grown by the PEER
    # AVERAGE recovery path: cf(t) = LNER(2019) * mean_p[ peer_p(t)/peer_p(2019) ].
    # This is a parallel-(growth-)trends DiD and is transparent + level-correct.
    peers_present = [p for p in PEERS if p in wide.columns]

    def cf_gap(peer_set):
        idx = wide[peer_set].div(wide[peer_set].loc[BASE]).mean(axis=1)  # avg peer recovery path
        cf = wide.loc[BASE, "London North Eastern Railway"] * idx
        post = [y for y in wide.index if y >= TREAT_YEAR and pd.notna(cf.get(y))]
        lner_g = float((wide["London North Eastern Railway"].reindex(post) / cf.reindex(post)).mean() - 1)
        mkt_g = float((wide["LNER+Lumo"].reindex(post) / cf.reindex(post)).mean() - 1)
        return cf, lner_g, mkt_g

    cf, lner_gap, mkt_gap = cf_gap(peers_present)
    _, lner_xemr, mkt_xemr = cf_gap([p for p in peers_present if p != "East Midlands Railway"])
    sc_out = {
        "peers": peers_present,
        "LNER_vs_peers_pct": round(100 * lner_gap, 1),
        "LNER+Lumo_vs_peers_pct": round(100 * mkt_gap, 1),
        "LNER_vs_peers_excl_EMR_pct": round(100 * lner_xemr, 1),
        "LNER+Lumo_vs_peers_excl_EMR_pct": round(100 * mkt_xemr, 1),
    }
    LOG.info("DiD-in-ratios vs peer LD operators (post-2021):")
    LOG.info(
        "  LNER %+.1f%% (excl EMR %+.1f%%);  LNER+Lumo %+.1f%% (excl EMR %+.1f%%)",
        100 * lner_gap,
        100 * lner_xemr,
        100 * mkt_gap,
        100 * mkt_xemr,
    )
    _plot(wide, cf, lner_gap, mkt_gap)

    out = {
        "recovery_ratio_2024_over_2019": recov_tbl,
        "lner_synthetic_control": sc_out,
        "lner_2024_m": float(wide.loc[2024, "London North Eastern Railway"]),
        "lumo_2024_m": float(wide.loc[2024, "Lumo"]),
    }
    (METRICS / "m3_operator_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame([{"metric": k, "value": v} for k, v in recov_tbl.items()]).to_csv(
        TABLES / "m3_operator_recovery.csv", index=False
    )
    LOG.info("operator analysis complete.")


def _plot(wide, cf, lner_gap, mkt_gap):
    base_lner = wide.loc[BASE, "London North Eastern Railway"]
    idx_years = [y for y in wide.index if y >= 2015]
    fig, ax = plt.subplots(figsize=(10, 6))
    for s, c, lw in [("London North Eastern Railway", "#d62728", 2.6), ("LNER+Lumo", "#7b3294", 2.2)]:
        ax.plot(idx_years, (wide[s] / base_lner * 100).reindex(idx_years), "o-", color=c, lw=lw, label=s)
    for i, p in enumerate(PEERS):
        if p in wide.columns:
            ax.plot(
                idx_years,
                (wide[p] / wide.loc[BASE, p] * 100).reindex(idx_years),
                color="grey",
                lw=1.0,
                alpha=0.55,
                label="Peer long-distance operators" if i == 0 else None,
            )
    ax.plot(
        idx_years,
        (cf / base_lner * 100).reindex(idx_years),
        "s--",
        color="#1f77b4",
        lw=1.8,
        label="LNER no-Lumo counterfactual (peer recovery)",
    )
    ax.axvline(TREAT_YEAR, color="#2166ac", ls=":", lw=1.5)
    ax.text(TREAT_YEAR + 0.1, 30, " Lumo enters (Oct 2021)", color="#2166ac", rotation=90, va="bottom", fontsize=8)
    ax.axhline(100, color="k", lw=0.6, alpha=0.4)
    ax.set_ylabel("Journeys indexed to 2019-20 = 100")
    ax.set_xlabel("Financial year (start)")
    ax.set_title(
        "Operator-level: LNER did NOT fall when Lumo entered — it surged past peers\n"
        f"vs comparable long-distance operators: LNER {lner_gap * 100:+.0f}%  ·  "
        f"LNER+Lumo {mkt_gap * 100:+.0f}%  (⇒ market growth, not cannibalisation)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "m3_operator_lner_lumo.png", dpi=150)
    plt.close(fig)
    LOG.info("figure -> %s", FIGURES / "m3_operator_lner_lumo.png")


if __name__ == "__main__":
    main()
