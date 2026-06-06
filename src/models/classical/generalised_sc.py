"""
Generalised Synthetic Control (Xu 2017) — interactive fixed effects / latent factors.

THINK -> RESEARCH -> CODE
  WHAT: Model log usage as Y_it = alpha_i + xi_t + lambda_i . f_t + effect, where f_t are
        LATENT TIME FACTORS and lambda_i their unit loadings (estimated by SVD on the
        never-treated control units). For each treated unit we estimate its loadings from
        its PRE-treatment period, then impute the no-Lumo counterfactual from the factors.
  WHY : This is the principled answer to our central problem (WEAKNESSES W1). The
        corridor-wide post-COVID recovery is exactly a latent common factor; GSC lets the
        factor structure ABSORB it (loading on long-distance/corridor stations) so the
        treated effect is net of it — without hand-picking donors. A brief Tier-2 method.
  CONTROLS: all balanced NON-Lumo stations (clean donors + corridor controls), so the
        factors can represent both national and corridor patterns from the data.
  CAVEAT: if corridor controls are themselves lifted by Lumo's line-wide response (SUTVA),
        that lift is partly absorbed into a factor ⇒ effect biased toward 0 (a lower bound).
        We report sensitivity over the number of factors r.

Run:  python -m src.models.classical.generalised_sc
Out:  results/tables/m3_generalised_sc.csv, results/metrics/m3_generalised_sc.json,
      results/figures/m3_generalised_sc.png
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, METRICS, PROCESSED, TABLES, ensure_dirs

LOG = get_logger("models.generalised_sc", log_file="logs/models.log")


def _matrix(panel, crs_list, years):
    sub = panel.filter(pl.col("crs").is_in(crs_list) & pl.col("year_start").is_in(years)).select(
        "crs", "year_start", "value"
    )
    wide = sub.pivot(values="value", index="crs", on="year_start")
    wide = wide.with_columns(pl.col("crs").cast(pl.Enum(crs_list)).alias("_o")).sort("_o")
    return np.log(np.clip(wide.select([str(y) for y in years]).to_numpy(), 1.0, None))


def gsc(Yc: np.ndarray, Yt: np.ndarray, pre: np.ndarray, r: int):
    """Return per-treated-unit (counterfactual[T], effect_pct) under r latent factors."""
    Yc_u = Yc - Yc.mean(1, keepdims=True)  # remove control unit FE
    time_eff = Yc_u.mean(0)  # common time effect [T]
    resid = Yc_u - time_eff[None, :]  # for factor extraction
    _, _, Vt = np.linalg.svd(resid, full_matrices=False)
    F = Vt[:r].T  # [T, r] orthonormal latent factors

    Xpre = np.column_stack([np.ones(int(pre.sum())), F[pre]])
    out = []
    for i in range(Yt.shape[0]):
        y = Yt[i]
        coef, *_ = np.linalg.lstsq(Xpre, y[pre] - time_eff[pre], rcond=None)
        alpha, lam = coef[0], coef[1:]
        cf = alpha + time_eff + F @ lam  # counterfactual [T]
        eff = float(np.exp((y[~pre] - cf[~pre]).mean()) - 1)
        out.append((cf, eff))
    return out


def main() -> None:
    ensure_dirs()
    cfg = load_config("base")
    treat = int(cfg["treatments"]["lumo"]["treat_year_start"])
    served = cfg["treatments"]["lumo"]["served_crs"]
    years = list(range(int(cfg["panel"]["lennon_era_min"]), int(cfg["panel"]["year_max"]) + 1))
    yarr = np.array(years)
    pre = yarr < treat

    panel = pl.read_parquet(PROCESSED / "panel.parquet")
    units = pl.read_parquet(PROCESSED / "units.parquet")
    name = dict(units.select("crs", "station_name").iter_rows())

    # controls = all balanced non-Lumo stations (clean donors + corridor)
    ctrl = units.filter(
        pl.col("balanced")
        & ~pl.col("crs").is_in(served)
        & pl.col("role").is_in(["donor_clean", "ecml_corridor_control"])
    )["crs"].to_list()
    Yc = _matrix(panel, ctrl, years)
    Yt = _matrix(panel, served, years)
    LOG.info(
        "GSC: %d control units, %d treated, %d years (pre=%d)", Yc.shape[0], len(served), len(years), int(pre.sum())
    )

    rows = []
    res_main = None
    for r in (2, 3, 4):
        res = gsc(Yc, Yt, pre, r)
        if r == 3:
            res_main = res
        for i, crs in enumerate(served):
            rows.append(
                {"crs": crs, "station_name": name[crs], "n_factors": r, "gsc_effect_pct": round(100 * res[i][1], 1)}
            )
        LOG.info("  r=%d factors: %s", r, {crs: round(100 * res[i][1], 1) for i, crs in enumerate(served)})

    pl.DataFrame(rows).write_csv(TABLES / "m3_generalised_sc.csv")
    summary = {
        crs: {
            f"r{r}_pct": next(x["gsc_effect_pct"] for x in rows if x["crs"] == crs and x["n_factors"] == r)
            for r in (2, 3, 4)
        }
        for crs in served
    }
    (METRICS / "m3_generalised_sc.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("GSC summary (effect %% by #factors): %s", json.dumps(summary))

    # figure (r=3): observed vs GSC counterfactual
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for i, (crs, ax) in enumerate(zip(served, axes.ravel())):
        cf, eff = res_main[i]
        ax.plot(years, np.exp(Yt[i]) / 1e6, "o-", color="#d62728", lw=2, label="observed")
        ax.plot(years, np.exp(cf) / 1e6, "s--", color="#0b6e4f", lw=1.8, label="GSC counterfactual (latent factors)")
        ax.axvline(treat, color="grey", ls=":", lw=1.3)
        ax.axvspan(2020, 2021, color="grey", alpha=0.1, lw=0)
        ax.set_title(f"{name[crs]} ({crs}): GSC effect {eff * 100:+.1f}% (r=3)", fontsize=10)
        ax.set_ylabel("entries+exits (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Generalised Synthetic Control — latent factors absorb the corridor recovery", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "m3_generalised_sc.png", dpi=150)
    plt.close(fig)
    LOG.info("figure -> %s | GSC complete.", FIGURES / "m3_generalised_sc.png")


if __name__ == "__main__":
    main()
