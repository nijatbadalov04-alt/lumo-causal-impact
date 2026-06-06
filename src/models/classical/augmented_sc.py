"""
Augmented / Ridge Synthetic Control (Ben-Michael, Feller & Rothstein 2021) — Tier 2.

THINK -> RESEARCH -> CODE
  WHAT: Convex SC constrains weights to the simplex (w>=0, sum=1), which can't fit a
        treated unit outside the donors' convex hull and leaves bias when the pre-fit is
        imperfect. Augmented SC relaxes this with an L2-regularised (ridge) outcome model
        that ALLOWS extrapolation (negative weights, intercept), shrinking toward a
        well-conditioned solution. We fit ridge on the WITHIN-CORRIDOR donors (the valid,
        confound-controlled design) per Lumo stop and read the post gap.
  WHY : §6 Tier 2 (Augmented SC) + a third independent SC estimator for the triangulation.
        Ridge tends to be less biased than convex SC when no convex combination fits well.
  Out: results/tables/m3_augmented_sc.csv, results/metrics/m3_augmented_sc.json

Run:  python -m src.models.classical.augmented_sc
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from sklearn.linear_model import RidgeCV

from src.models.classical.synthetic_control import build_outcome_matrix
from src.models.classical.within_corridor import corridor_donors
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import METRICS, PROCESSED, TABLES, ensure_dirs

LOG = get_logger("models.augmented_sc", log_file="logs/models.log")


def main() -> None:
    ensure_dirs()
    cfg = load_config("base")
    treat = int(cfg["treatments"]["lumo"]["treat_year_start"])
    served = cfg["treatments"]["lumo"]["served_crs"]
    years = list(range(int(cfg["panel"]["lennon_era_min"]), int(cfg["panel"]["year_max"]) + 1))
    yarr = np.array(years)
    pre, post = yarr < treat, yarr >= treat

    panel = pl.read_parquet(PROCESSED / "panel.parquet")
    units = pl.read_parquet(PROCESSED / "units.parquet")
    name = dict(units.select("crs", "station_name").iter_rows())

    rows = []
    for tcrs in served:
        donors = corridor_donors(panel, units, tcrs, years, treat, served)
        if len(donors) < 4:
            continue
        M = build_outcome_matrix(panel, [tcrs] + donors, years)
        y, Y0 = M[0], M[1:]
        Xpre, Xpost = Y0[:, pre].T, Y0[:, post].T  # (n_pre, n_donors), (n_post, n_donors)
        ridge = RidgeCV(alphas=np.logspace(-4, 4, 40)).fit(Xpre, y[pre])
        cf_pre, cf_post = ridge.predict(Xpre), ridge.predict(Xpost)
        pre_rmspe = float(np.sqrt(np.mean((cf_pre - y[pre]) ** 2)))
        eff = float(np.exp((y[post] - cf_post).mean()) - 1)
        rows.append(
            {
                "crs": tcrs,
                "station_name": name[tcrs],
                "n_donors": len(donors),
                "ridge_alpha": round(float(ridge.alpha_), 3),
                "pre_rmspe": round(pre_rmspe, 4),
                "augmented_sc_effect_pct": round(100 * eff, 1),
            }
        )
        LOG.info(
            "%s (%s): Augmented(ridge) SC effect = %+.1f%%  (alpha=%.2f, pre_RMSPE=%.4f, n_donors=%d)",
            tcrs,
            name[tcrs],
            100 * eff,
            ridge.alpha_,
            pre_rmspe,
            len(donors),
        )

    pl.DataFrame(rows).write_csv(TABLES / "m3_augmented_sc.csv")
    (METRICS / "m3_augmented_sc.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    LOG.info("Augmented SC complete -> %s", TABLES / "m3_augmented_sc.csv")


if __name__ == "__main__":
    main()
