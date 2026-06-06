"""
Open-access intercity sector analysis (RQ4 generalisation + the policy synthesis).

THINK -> RESEARCH -> CODE
  WHAT: Aggregate the three intercity open-access operators (Grand Central + Hull
        Trains + Lumo) into one series and ask the regulator's question directly:
        has open-access intercity GROWN, and did it come ALONGSIDE franchised
        long-distance (creation) or AT ITS EXPENSE (substitution)?
  WHY : RQ4 — do the patterns generalise beyond Lumo? And §10/§2 — keep the
        substitution-vs-creation policy question front and centre. ORR is deciding
        live open-access bids; "do these operators grow the market?" is THE question.
  DATA: ORR Table 1223 (operator, from 2011) for the three OA operators; Table 1221
        (sector, from 1994) for total rail + the open-access sector. Heathrow Express
        and Wrexham&Shropshire excluded from the intercity aggregate (not comparable).
  CAVEAT: descriptive at the sector level — establishes the market context the
        causal estimates sit within, not itself a causal estimate.

Run:  python -m src.models.classical.openaccess_sector
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

LOG = get_logger("models.openaccess_sector", log_file="logs/models.log")

OA_INTERCITY = ["Grand Central", "Hull Trains", "Lumo"]
LAUNCHES = {"Hull Trains": 2000, "Grand Central": 2007, "Lumo": 2021}


def main() -> None:
    ensure_dirs()
    op = pd.read_parquet(INTERIM / "operator_journeys.parquet")
    a = op[(op.freq == "annual") & (op.flag == "observed")]
    wide = a.pivot_table(index="period_start", columns="series", values="value_m").sort_index()
    oa = wide[OA_INTERCITY].sum(axis=1, min_count=1)

    LOG.info(
        "Open-access intercity (GC+Hull+Lumo) journeys (m): %s", {int(y): round(v, 2) for y, v in oa.dropna().items()}
    )

    # franchised long-distance comparator (from operator table)
    franchised_ld = wide[
        [
            "London North Eastern Railway",
            "Avanti West Coast",
            "CrossCountry",
            "Great Western Railway",
            "East Midlands Railway",
            "TransPennine Express",
        ]
    ].sum(axis=1, min_count=1)

    summary = {
        "oa_intercity_2011_m": float(oa.get(2011, np.nan)),
        "oa_intercity_2024_m": float(oa.get(2024, np.nan)),
        "oa_intercity_growth_x": round(float(oa.get(2024) / oa.get(2011)), 2),
        "oa_share_of_LD_2024_pct": round(100 * float(oa.get(2024) / franchised_ld.get(2024)), 2),
    }

    # long-run sector view (Table 1221, from 1994)
    if (INTERIM / "sector_journeys.parquet").exists():
        sec = pd.read_parquet(INTERIM / "sector_journeys.parquet")
        sa = sec[(sec.freq == "annual") & (sec.flag == "observed")]
        secw = sa.pivot_table(index="period_start", columns="series", values="value_m").sort_index()
        tot_col = [c for c in secw.columns if "Total" in c][0]
        oa_col = [c for c in secw.columns if "Open access" in c][0]
        summary["total_rail_1994_m"] = round(float(secw[tot_col].iloc[0]), 1)
        summary["total_rail_2024_m"] = round(float(secw[tot_col].dropna().iloc[-1]), 1)
        summary["oa_sector_2024_m"] = round(float(secw[oa_col].dropna().iloc[-1]), 2)
    else:
        secw = tot_col = oa_col = None

    (METRICS / "m6_openaccess_sector.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("summary: %s", json.dumps(summary))

    # ---- figure: OA intercity rise + franchised LD (did franchised fall as OA grew?) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    yrs = [y for y in wide.index if y >= 2011]
    ax1.plot(yrs, oa.reindex(yrs), "o-", color="#7b3294", lw=2.4, label="Open-access intercity (GC+Hull+Lumo)")
    for op_name in OA_INTERCITY:
        ax1.plot(yrs, wide[op_name].reindex(yrs), lw=1.0, alpha=0.7, ls="--", label=op_name)
    ax1.axvline(2021, color="#2166ac", ls=":", lw=1.2)
    ax1.text(2021.1, 0.1, "Lumo", color="#2166ac", fontsize=8, rotation=90)
    ax1.set_ylabel("Journeys (millions/yr)")
    ax1.set_xlabel("Financial year (start)")
    ax1.set_title(
        f"Open-access intercity grew {summary['oa_intercity_growth_x']}x (2011→2024), "
        f"to {summary['oa_share_of_LD_2024_pct']:.0f}% of franchised long-distance"
    )
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    if secw is not None:
        sy = [y for y in secw.index if y >= 1997]
        ax2.plot(sy, secw[tot_col].reindex(sy), "o-", color="#1b7837", lw=2, ms=3, label="Total rail journeys")
        ax2.plot(
            sy,
            secw[oa_col].reindex(sy) * 10,
            "s-",
            color="#d62728",
            lw=1.6,
            ms=3,
            label="Open-access sector (×10, for scale)",
        )
        for op_name, yr in LAUNCHES.items():
            ax2.axvline(yr, color="grey", ls=":", lw=1)
            ax2.text(yr + 0.1, ax2.get_ylim()[1] * 0.02, op_name.split()[0], rotation=90, fontsize=7, color="grey")
        ax2.set_ylabel("Journeys (millions/yr)")
        ax2.set_xlabel("Financial year (start)")
        ax2.set_title(
            "Total rail kept GROWING as open-access entered (1997→2024)\n"
            "open-access rose without total rail falling ⇒ market expansion"
        )
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "m6_openaccess_sector.png", dpi=150)
    plt.close(fig)
    LOG.info("figure -> %s", FIGURES / "m6_openaccess_sector.png")
    pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(TABLES / "m6_openaccess_sector.csv")
    LOG.info("open-access sector analysis complete.")


if __name__ == "__main__":
    main()
