"""
Synthetic Control (Abadie, Diamond & Hainmueller 2010) for the Lumo entry — M3.

THINK -> RESEARCH -> CODE
  WHAT: For each Lumo-served station, build a convex "synthetic" twin from a pool of
        the K most pre-trajectory-similar, OFF-corridor, station-group-clean donors,
        fit on the pre-treatment period, and read the post-2021 gap as the effect.
  WHY : Outcome = log(entries+exits). At the STATION-TOTAL level, pure *substitution*
        (Lumo abstracts LNER passengers) leaves the total ~unchanged; *creation*
        (new rail journeys / modal shift) raises it. A positive, significant SC gap =
        net market creation; ~0 = substitution / no net effect (RQ1).
  DONOR POOL: K=40 nearest donors by Euclidean distance on the *pre-treatment* log
        series (uses only pre data -> no leakage). A MODERATE pool is essential:
        an oversized pool (>> #pre-periods) lets convex weights perfectly interpolate
        the pre-period (pre-RMSPE -> 0), which makes the fit and placebo inference
        meaningless. K=40 ~ #pre-periods is the canonical Abadie regime.
  COVID: the pre-fit window 2004-2020 INCLUDES the 2020-21 COVID crater, so the twin
        must reproduce the crash; the post comparison then isolates Lumo from the
        common recovery (the central confound).
  INFERENCE: Abadie placebo-in-space — re-estimate treating every donor as fake-
        treated; rank the real unit's post/pre RMSPE ratio (p = rank/(N+1)).
  GUARD: pre-RMSPE < 1e-6 is flagged `overfit` and its inference treated as unreliable.

Run:  python -m src.models.classical.synthetic_control
Out:  results/tables/m3_synthetic_control_effects.csv
      results/figures/m3_sc_<crs>.png, m3_placebo_<crs>.png
      results/metrics/m3_synthetic_control.json
"""

from __future__ import annotations

import json
import warnings

import cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, METRICS, PROCESSED, TABLES, ensure_dirs

warnings.filterwarnings("ignore")
LOG = get_logger("models.synthetic_control", log_file="logs/models.log")

K_DONORS = 40  # canonical moderate donor pool (~ #pre-treatment periods)
SIZE_BAND = (0.1, 10.0)  # loose comparability pre-filter before K-nearest selection
OVERFIT_EPS = 1e-6  # pre-RMSPE below this = degenerate/overfit


def fit_weights(Y0_pre: np.ndarray, y1_pre: np.ndarray) -> np.ndarray:
    """Convex SC weights minimising pre-treatment fit (w>=0, sum w = 1)."""
    n = Y0_pre.shape[0]
    w = cp.Variable(n, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(y1_pre - Y0_pre.T @ w)), [cp.sum(w) == 1])
    prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=60000)
    wv = np.clip(np.asarray(w.value, dtype=float), 0, None)
    s = wv.sum()
    return wv / s if s > 0 else np.full(n, 1.0 / n)


def _rmspe(gap: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(gap[mask] ** 2)))


def run_sc(y1: np.ndarray, Y0: np.ndarray, years: np.ndarray, treat_year: int) -> dict:
    pre = years < treat_year
    w = fit_weights(Y0[:, pre], y1[pre])
    synth = Y0.T @ w
    gap = y1 - synth
    pr, po = _rmspe(gap, pre), _rmspe(gap, ~pre)
    return {
        "w": w,
        "synth": synth,
        "gap": gap,
        "pre": pre,
        "pre_rmspe": pr,
        "post_rmspe": po,
        "ratio": po / pr if pr > 0 else np.inf,
    }


def build_outcome_matrix(panel: pl.DataFrame, crs_list: list[str], years: list[int]) -> np.ndarray:
    sub = panel.filter(pl.col("crs").is_in(crs_list) & pl.col("year_start").is_in(years)).select(
        "crs", "year_start", "value"
    )
    wide = sub.pivot(values="value", index="crs", on="year_start")
    wide = wide.with_columns(pl.col("crs").cast(pl.Enum(crs_list)).alias("_o")).sort("_o")
    # clip to >=1 before log: the panel has a handful of true-zero donor-years (e.g. request
    # stops in 2020); without this, log(0)=-inf silently poisons distances/weights. (Parity with
    # generalised_sc / deep_counterfactual, which already guard this.)
    return np.log(np.clip(wide.select([str(y) for y in years]).to_numpy(), 1.0, None))


def main() -> None:
    ensure_dirs()
    cfg = load_config("base")
    treat_year = int(cfg["treatments"]["lumo"]["treat_year_start"])
    served = cfg["treatments"]["lumo"]["served_crs"]
    years = list(range(int(cfg["panel"]["lennon_era_min"]), int(cfg["panel"]["year_max"]) + 1))
    yarr = np.array(years)
    pre = yarr < treat_year

    panel = pl.read_parquet(PROCESSED / "panel.parquet")
    units = pl.read_parquet(PROCESSED / "units.parquet")
    donors_all = units.filter(pl.col("role") == "donor_clean").select("crs", "baseline_ee_2019").drop_nulls()
    base_map = dict(units.select("crs", "baseline_ee_2019").drop_nulls().iter_rows())

    rows, metrics = [], {}
    for tcrs in served:
        tname = units.filter(pl.col("crs") == tcrs)["station_name"][0]
        tbase = base_map.get(tcrs)
        if tbase is None:
            continue
        # (1) loose size band -> (2) K nearest by pre-treatment log-trajectory distance
        pool = donors_all.filter(
            (pl.col("baseline_ee_2019") >= SIZE_BAND[0] * tbase) & (pl.col("baseline_ee_2019") <= SIZE_BAND[1] * tbase)
        )
        pool_crs = pool["crs"].to_list()
        Mfull = build_outcome_matrix(panel, [tcrs] + pool_crs, years)
        y1, Y0full = Mfull[0], Mfull[1:]
        dist = np.sqrt(((Y0full[:, pre] - y1[pre]) ** 2).sum(axis=1))
        order = np.argsort(dist)[:K_DONORS]
        Y0 = Y0full[order]
        donor_crs = [pool_crs[i] for i in order]

        res = run_sc(y1, Y0, yarr, treat_year)
        post = ~pre
        per_year_eff = {int(y): float(np.exp(g) - 1) for y, g in zip(yarr[post], res["gap"][post])}
        avg_eff = float(np.exp(res["gap"][post].mean()) - 1)
        overfit = res["pre_rmspe"] < OVERFIT_EPS

        # placebo-in-space (cached gaps, reused for plotting)
        placebo_gaps, placebo_ratios = [], []
        for j in range(Y0.shape[0]):
            rj = run_sc(Y0[j], np.delete(Y0, j, axis=0), yarr, treat_year)
            placebo_gaps.append(rj["gap"])
            if rj["pre_rmspe"] > OVERFIT_EPS and np.isfinite(rj["ratio"]):
                placebo_ratios.append(rj["ratio"])
        placebo_ratios = np.array(placebo_ratios)
        rank = int((placebo_ratios >= res["ratio"]).sum()) + 1
        p_value = rank / (len(placebo_ratios) + 1)

        top = "; ".join(
            f"{c}:{w:.2f}" for c, w in sorted(zip(donor_crs, res["w"]), key=lambda t: -t[1])[:5] if w > 0.01
        )
        LOG.info(
            "%s (%s): K=%d  pre_RMSPE=%.4f%s  avg_effect=%+.1f%%  ratio=%.1f  p=%.3f  top=[%s]",
            tcrs,
            tname,
            len(donor_crs),
            res["pre_rmspe"],
            " [OVERFIT]" if overfit else "",
            100 * avg_eff,
            res["ratio"],
            p_value,
            top,
        )

        rows.append(
            {
                "crs": tcrs,
                "station_name": tname,
                "n_donors": len(donor_crs),
                "pre_rmspe": round(res["pre_rmspe"], 5),
                "overfit_flag": overfit,
                "avg_post_effect_pct": round(100 * avg_eff, 2),
                "rmspe_ratio": round(res["ratio"], 2),
                "placebo_p_value": round(p_value, 4),
                **{f"effect_{y}_pct": round(100 * e, 2) for y, e in per_year_eff.items()},
                "top_donors": top,
            }
        )
        metrics[tcrs] = {
            "station_name": tname,
            "avg_post_effect_pct": 100 * avg_eff,
            "placebo_p_value": p_value,
            "pre_rmspe": res["pre_rmspe"],
            "ratio": res["ratio"],
            "overfit": overfit,
            "per_year_effect_pct": {k: 100 * v for k, v in per_year_eff.items()},
        }
        _plot_sc(tcrs, tname, yarr, y1, res, treat_year, placebo_gaps, p_value, avg_eff, overfit)

    pl.DataFrame(rows).write_csv(TABLES / "m3_synthetic_control_effects.csv")
    (METRICS / "m3_synthetic_control.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOG.info("M3 synthetic control complete -> %s", TABLES / "m3_synthetic_control_effects.csv")


def _plot_sc(tcrs, tname, years, y1, res, treat_year, placebo_gaps, p_value, avg_eff, overfit):
    synth = res["synth"]
    tag = "  [pre-fit degenerate — unreliable]" if overfit else ""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), height_ratios=[2, 1], sharex=True)
    ax1.plot(years, np.exp(y1) / 1e6, "o-", color="#d62728", lw=2.2, label=f"{tname} (observed)")
    ax1.plot(years, np.exp(synth) / 1e6, "s--", color="#1f77b4", lw=1.8, label="Synthetic control")
    ax1.axvline(treat_year, color="grey", ls=":", lw=1.4)
    ax1.axvspan(2020, 2021, color="grey", alpha=0.12, lw=0)
    ax1.set_ylabel("Entries + exits (millions)")
    ax1.set_title(
        f"Synthetic control — {tname} ({tcrs})\n"
        f"avg post-2021 effect = {avg_eff * 100:+.1f}%  ·  placebo p = {p_value:.3f}{tag}",
        fontsize=11,
    )
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.axhline(0, color="k", lw=0.7)
    ax2.plot(years, (np.exp(res["gap"]) - 1) * 100, "o-", color="#2ca02c", lw=1.8)
    ax2.axvline(treat_year, color="grey", ls=":", lw=1.4)
    ax2.set_ylabel("Gap (obs − synth, %)")
    ax2.set_xlabel("Financial year (start)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / f"m3_sc_{tcrs}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for g in placebo_gaps:
        ax.plot(years, (np.exp(g) - 1) * 100, color="grey", lw=0.6, alpha=0.35)
    ax.plot(years, (np.exp(res["gap"]) - 1) * 100, color="#d62728", lw=2.6, label=tname)
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(treat_year, color="#2166ac", ls=":", lw=1.4)
    ax.set_ylabel("Gap (obs − synth, %)")
    ax.set_xlabel("Financial year (start)")
    ax.set_ylim(-60, 80)
    ax.set_title(f"Placebo-in-space — {tname} vs donor placebos (p = {p_value:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / f"m3_placebo_{tcrs}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
