"""
CORRECTED design: within-ECML-corridor synthetic control (M3, post-confound).

THINK -> RESEARCH -> CODE
  WHY THIS EXISTS: robustness test (C) in evaluate/robustness_sc.py showed ECML
  *through* stations that Lumo does NOT serve (York +18%, Doncaster +25%,
  Darlington +15%, Grantham +16%) rise ~as much as Newcastle (+17%) against
  OFF-ECML donors. ⇒ that +17% is mostly a CORRIDOR-WIDE recovery (long-distance
  leisure rebounding faster than commuter/regional/Avanti-hit lines), NOT a
  Lumo-specific effect. The off-ECML donor pool is an invalid counterfactual.

  THE FIX: build each Lumo stop's synthetic from OTHER ECML-CORRIDOR stations
  (role=ecml_corridor_control). These share the corridor-wide shock but receive no
  Lumo service, so the gap now isolates the *Lumo-specific* station-total effect.

  CAVEAT (logged, honest): this is a conservative within-corridor contrast. If LNER
  responded to Lumo line-wide (cheaper Advances, more capacity) the controls are
  partially treated (SUTVA), which biases the within-corridor effect toward zero —
  i.e. it may UNDER-state the true market-creation effect. The clean resolution is
  OD-pair / operator-level data (ODM, TOC) — pursued separately.

Run:  python -m src.models.classical.within_corridor
Out:  results/tables/m3_within_corridor_effects.csv, m3_within_corridor.json,
      results/figures/m3_within_corridor_<crs>.png
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.models.classical.synthetic_control import SIZE_BAND, build_outcome_matrix, run_sc
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, METRICS, PROCESSED, TABLES, ensure_dirs

LOG = get_logger("models.within_corridor", log_file="logs/models.log")

K_WITHIN = 12  # smaller pool: only ~25 corridor controls exist


def corridor_donors(panel, units, tcrs, years, treat_year, lumo_stops):
    """K nearest ECML-corridor controls (NOT Lumo stops) by pre-treatment trajectory."""
    base_map = dict(units.select("crs", "baseline_ee_2019").drop_nulls().iter_rows())
    tbase = base_map[tcrs]
    pool = (
        units.filter(
            (pl.col("role") == "ecml_corridor_control") & ~pl.col("crs").is_in(lumo_stops) & pl.col("balanced")
        )
        .select("crs", "baseline_ee_2019")
        .drop_nulls()
        .filter(
            (pl.col("baseline_ee_2019") >= SIZE_BAND[0] * tbase) & (pl.col("baseline_ee_2019") <= SIZE_BAND[1] * tbase)
        )["crs"]
        .to_list()
    )
    yarr = np.array(years)
    pre = yarr < treat_year
    M = build_outcome_matrix(panel, [tcrs] + pool, years)
    dist = np.sqrt(((M[1:][:, pre] - M[0][pre]) ** 2).sum(axis=1))
    return [pool[i] for i in np.argsort(dist)[:K_WITHIN]]


def main() -> None:
    ensure_dirs()
    cfg = load_config("base")
    treat = int(cfg["treatments"]["lumo"]["treat_year_start"])
    served = cfg["treatments"]["lumo"]["served_crs"]
    years = list(range(int(cfg["panel"]["lennon_era_min"]), int(cfg["panel"]["year_max"]) + 1))
    yarr = np.array(years)
    post = yarr >= treat

    panel = pl.read_parquet(PROCESSED / "panel.parquet")
    units = pl.read_parquet(PROCESSED / "units.parquet")
    name = dict(units.select("crs", "station_name").iter_rows())

    rows = []
    for tcrs in served:
        donors = corridor_donors(panel, units, tcrs, years, treat, served)
        if len(donors) < 4:
            LOG.warning("%s: only %d corridor donors — skipping", tcrs, len(donors))
            continue
        M = build_outcome_matrix(panel, [tcrs] + donors, years)
        res = run_sc(M[0], M[1:], yarr, treat)
        avg = float(np.exp(res["gap"][post].mean()) - 1)

        # placebo-in-space within the corridor-donor pool
        ratios = []
        for j in range(M[1:].shape[0]):
            rj = run_sc(M[1:][j], np.delete(M[1:], j, axis=0), yarr, treat)
            if rj["pre_rmspe"] > 1e-6 and np.isfinite(rj["ratio"]):
                ratios.append(rj["ratio"])
        ratios = np.array(ratios)
        p = (int((ratios >= res["ratio"]).sum()) + 1) / (len(ratios) + 1)

        LOG.info(
            "%s (%s): WITHIN-corridor effect = %+.1f%%  pre_RMSPE=%.4f  p=%.3f  (n_donors=%d)",
            tcrs,
            name[tcrs],
            100 * avg,
            res["pre_rmspe"],
            p,
            len(donors),
        )
        rows.append(
            {
                "crs": tcrs,
                "station_name": name[tcrs],
                "n_corridor_donors": len(donors),
                "within_corridor_effect_pct": round(100 * avg, 1),
                "pre_rmspe": round(res["pre_rmspe"], 4),
                "placebo_p": round(p, 3),
            }
        )
        _plot(tcrs, name[tcrs], yarr, M[0], res, treat, avg, p)

    pl.DataFrame(rows).write_csv(TABLES / "m3_within_corridor_effects.csv")
    (METRICS / "m3_within_corridor.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    LOG.info("within-corridor SC complete -> %s", TABLES / "m3_within_corridor_effects.csv")


def _plot(tcrs, tname, years, y1, res, treat, avg, p):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8), height_ratios=[2, 1], sharex=True)
    a1.plot(years, np.exp(y1) / 1e6, "o-", color="#d62728", lw=2.2, label=f"{tname} (Lumo stop)")
    a1.plot(
        years, np.exp(res["synth"]) / 1e6, "s--", color="#6a51a3", lw=1.8, label="Synthetic (ECML-corridor controls)"
    )
    a1.axvline(treat, color="grey", ls=":", lw=1.4)
    a1.axvspan(2020, 2021, color="grey", alpha=0.12, lw=0)
    a1.set_ylabel("Entries + exits (millions)")
    a1.legend()
    a1.grid(alpha=0.3)
    a1.set_title(
        f"WITHIN-CORRIDOR synthetic control — {tname} ({tcrs})\n"
        f"Lumo-specific effect = {avg * 100:+.1f}%  ·  placebo p = {p:.3f}  "
        f"(controls = non-Lumo ECML stations)",
        fontsize=10,
    )
    a2.axhline(0, color="k", lw=0.7)
    a2.plot(years, (np.exp(res["gap"]) - 1) * 100, "o-", color="#2ca02c", lw=1.8)
    a2.axvline(treat, color="grey", ls=":", lw=1.4)
    a2.set_ylabel("Gap (%)")
    a2.set_xlabel("Financial year (start)")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / f"m3_within_corridor_{tcrs}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
