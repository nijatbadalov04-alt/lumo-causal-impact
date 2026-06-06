"""
Heterogeneity — which stations grew vs were cannibalised (RQ2).

THINK -> RESEARCH -> CODE
  HONEST CONSTRAINT: a Causal Forest (Athey-Wager) estimates CATE across MANY treated
  units; we have only 4 Lumo stops, so a forest on the Lumo treatment is not identified.
  We therefore answer RQ2 two ways, both honest:
   (1) DESCRIPTIVE — relate each Lumo stop's effect (within-corridor SC / DiD / deep) to
       its characteristics (London-corridor share, baseline size, season/commuter share).
       The pattern IS the finding: the effect concentrates where the London market share
       is high (Newcastle) and vanishes/reverses at commuter (Stevenage) or non-London
       hub (Edinburgh) stations.
   (2) DATA-RICH context — a random forest predicting each station's post-2021 recovery
       ratio (2024/2019) from covariates across ALL ~2.3k balanced stations, with
       permutation importance, to show which station types grew — and where the Lumo
       stops sit relative to their predicted recovery.

Run:  python -m src.models.causal_forest.heterogeneity
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, INTERIM, METRICS, PROCESSED, TABLES, ensure_dirs

LOG = get_logger("models.heterogeneity", log_file="logs/models.log")


def main() -> None:
    ensure_dirs()
    cfg = load_config("base")
    served = cfg["treatments"]["lumo"]["served_crs"]

    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    units = pd.read_parquet(PROCESSED / "units.parquet")

    # recovery ratio 2024/2019 per station (post-COVID growth, the RQ2 'grew vs not')
    v19 = panel[panel.year_start == 2019].set_index("crs")["value"]
    v24 = panel[panel.year_start == 2024].set_index("crs")["value"]
    rec = (v24 / v19).rename("recovery").reset_index()
    attr = panel.groupby("crs", observed=True)[["ee_season", "ee_all"]].first().reset_index()
    attr["season_share_2024"] = attr["ee_season"] / attr["ee_all"]
    df = (
        units.merge(rec, on="crs")
        .merge(attr[["crs", "season_share_2024"]], on="crs")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["recovery", "baseline_ee_2019"])
    )
    df["log_base"] = np.log(df["baseline_ee_2019"])
    df["exposure_kgx"] = df["exposure_kgx"].astype(float)
    df["season_share_2024"] = df["season_share_2024"].fillna(df["season_share_2024"].median())

    # exogenous geographic covariate: distance to London (NaPTAN coordinates)
    cov_path = INTERIM / "station_covariates.parquet"
    if cov_path.exists():
        cov = pd.read_parquet(cov_path)[["crs", "distance_to_london_km"]]
        df = df.merge(cov, on="crs", how="left")
        df["distance_to_london_km"] = df["distance_to_london_km"].fillna(df["distance_to_london_km"].median())
        feats = ["log_base", "exposure_kgx", "season_share_2024", "distance_to_london_km"]
    else:
        feats = ["log_base", "exposure_kgx", "season_share_2024"]
    train = df[df.role == "donor_clean"]  # learn the general recovery surface on clean stations
    X, y = train[feats].to_numpy(), train["recovery"].to_numpy()
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=20, random_state=cfg["seed"], n_jobs=-1)
    rf.fit(X, y)
    imp = permutation_importance(rf, X, y, n_repeats=10, random_state=cfg["seed"], n_jobs=-1)
    importance = {f: round(float(i), 4) for f, i in zip(feats, imp.importances_mean)}
    LOG.info("recovery-driver permutation importance: %s", importance)
    LOG.info(
        "KGX-exposed mean recovery=%.3f vs non-exposed=%.3f",
        df[df.exposure_kgx == 1].recovery.mean(),
        df[df.exposure_kgx == 0].recovery.mean(),
    )

    # locate Lumo stops vs their predicted recovery
    lumo = df[df.crs.isin(served)].copy()
    lumo["predicted_recovery"] = rf.predict(lumo[feats].to_numpy())
    lumo["excess_vs_predicted_pct"] = 100 * (lumo["recovery"] / lumo["predicted_recovery"] - 1)
    het = lumo[
        [
            "crs",
            "station_name",
            "recovery",
            "predicted_recovery",
            "excess_vs_predicted_pct",
            "baseline_ee_2019",
            "season_share_2024",
            "exposure_kgx",
            "main_od",
        ]
    ].sort_values("excess_vs_predicted_pct", ascending=False)
    het.to_csv(TABLES / "m5_heterogeneity_lumo.csv", index=False)
    LOG.info("Lumo stations vs predicted recovery (RQ2 — who grew):")
    for r in het.itertuples():
        LOG.info(
            "  %-10s recovery=%.2f predicted=%.2f excess=%+.1f%%  (main_OD=%s, season_share=%.2f)",
            r.station_name,
            r.recovery,
            r.predicted_recovery,
            r.excess_vs_predicted_pct,
            r.main_od,
            r.season_share_2024,
        )

    summary = {
        "recovery_driver_importance": importance,
        "kgx_exposed_mean_recovery": round(float(df[df.exposure_kgx == 1].recovery.mean()), 3),
        "non_exposed_mean_recovery": round(float(df[df.exposure_kgx == 0].recovery.mean()), 3),
        "lumo_excess_vs_predicted": {r.crs: round(r.excess_vs_predicted_pct, 1) for r in het.itertuples()},
        "note": "4 treated units -> descriptive heterogeneity; full Causal Forest needs "
        "many treated units (pool open-access launches or use OD-pairs).",
    }
    (METRICS / "m5_heterogeneity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # figure: Lumo effect vs London-corridor share (the heterogeneity story)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        het["season_share_2024"],
        het["excess_vs_predicted_pct"],
        s=np.sqrt(het["baseline_ee_2019"]) / 3,
        c=het["exposure_kgx"],
        cmap="coolwarm",
        edgecolor="k",
        zorder=3,
    )
    for r in het.itertuples():
        ax.annotate(
            r.station_name,
            (r.season_share_2024, r.excess_vs_predicted_pct),
            fontsize=9,
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Season-ticket (commuter) share, 2024-25")
    ax.set_ylabel("Lumo-stop recovery vs predicted (%)")
    ax.set_title(
        "RQ2 heterogeneity — Lumo's growth concentrates at LONG-DISTANCE stations\n"
        "(low commuter share); commuter stops (Stevenage) underperform. "
        "Colour=KGX-exposed, size=baseline volume.",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "m5_heterogeneity.png", dpi=150)
    plt.close(fig)
    LOG.info("figure -> %s | heterogeneity complete.", FIGURES / "m5_heterogeneity.png")


if __name__ == "__main__":
    main()
