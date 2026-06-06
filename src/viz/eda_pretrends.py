"""
M2 EDA: pre-treatment trajectories — Newcastle (treated) vs candidate donor groups.

THINK -> RESEARCH -> CODE
  WHAT: Two views of log entries+exits, 2004-2024:
        (L) absolute log levels — does Newcastle sit among comparable big cities?
        (R) indexed to 2019=100 — the post-COVID recovery window where Lumo's
            Oct-2021 entry lands; a Lumo *abstraction* effect on the incumbent would
            show ECML stations recovering DIFFERENTLY from off-ECML donors.
  WHY : Synthetic control / DiD credibility rests on pre-treatment parallelism.
        This is the eyeball test before the formal donor match at M3. It also makes
        COVID confound visible: 2020-21 is a crater for everyone.
  GROUPS (data-driven from main_od + line knowledge):
        treated      = Newcastle (Lumo stop, KGX-dominant, LNER incumbent)
        ecml_through = York/Doncaster/Darlington/Grantham (LNER, NOT Lumo -> spillover risk)
        clean_donors = Manchester/Liverpool/Bristol/Sheffield/Cardiff/Glasgow/Birmingham
                       (large intercity, OFF the East Coast Main Line)
  Out: results/figures/m2_pretrends_newcastle.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, PROCESSED

LOG = get_logger("viz.eda_pretrends", log_file="logs/features.log")

GROUPS = {
    "treated": {"crs": ["NCL"], "color": "#d62728", "lw": 2.6, "ls": "-", "label": "Newcastle (Lumo-served, treated)"},
    "ecml_through": {
        "crs": ["YRK", "DON", "DAR", "GRA"],
        "color": "#ff7f0e",
        "lw": 1.3,
        "ls": "--",
        "label": "ECML through (LNER, not Lumo) — spillover risk",
    },
    "clean_donors": {
        "crs": ["MAN", "LIV", "BRI", "SHF", "CDF", "GLC", "BHM"],
        "color": "#1f77b4",
        "lw": 1.1,
        "ls": ":",
        "label": "Off-ECML intercity donors",
    },
}


def main() -> None:
    panel = pl.read_parquet(PROCESSED / "panel.parquet")
    present = set(panel["crs"].unique().to_list())
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for _gname, g in GROUPS.items():
        for crs in g["crs"]:
            if crs not in present:
                LOG.warning("CRS %s not in panel — skipping", crs)
                continue
            s = panel.filter(pl.col("crs") == crs).sort("year_start").select(["year_start", "value", "station_name"])
            yrs = s["year_start"].to_list()
            val = s["value"].to_list()
            name = s["station_name"][0]
            import numpy as np

            logv = np.log(np.array(val, dtype=float))
            axes[0].plot(yrs, logv, color=g["color"], lw=g["lw"], ls=g["ls"], alpha=0.9)

            base = dict(zip(yrs, val)).get(2019)
            if base:
                idx = [100 * v / base for v in val]
                axes[1].plot(yrs, idx, color=g["color"], lw=g["lw"], ls=g["ls"], alpha=0.9)
            if crs == "NCL":
                axes[0].annotate(
                    name, (yrs[-1], logv[-1]), color=g["color"], fontsize=9, fontweight="bold", va="center"
                )

    for ax in axes:
        ax.axvspan(2020, 2021, color="grey", alpha=0.15, lw=0)
        ax.axvline(2021, color="#2166ac", ls="-", lw=1.4, alpha=0.8)
        ax.text(2021.1, ax.get_ylim()[0], " Lumo (Oct 2021)", color="#2166ac", fontsize=8, rotation=90, va="bottom")
        ax.set_xlabel("Financial year (start)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("log(entries + exits)")
    axes[0].set_title("(a) Absolute levels — Newcastle sits among comparable big cities")
    axes[1].set_ylabel("Entries + exits, indexed to 2019 = 100")
    axes[1].set_title("(b) Post-COVID recovery — where the Lumo effect would appear")
    axes[1].axhline(100, color="k", lw=0.6, alpha=0.4)

    handles = [plt.Line2D([0], [0], color=g["color"], lw=2, ls=g["ls"], label=g["label"]) for g in GROUPS.values()]
    handles.append(plt.Line2D([0], [0], color="grey", lw=8, alpha=0.3, label="COVID trough (2020-21)"))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Pre-treatment parallelism check — Newcastle vs candidate donors "
        "(log entries+exits, LENNON era). Formal donor match at M3.",
        fontsize=11,
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.99))
    out = FIGURES / "m2_pretrends_newcastle.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("figure -> %s", out)


if __name__ == "__main__":
    main()
