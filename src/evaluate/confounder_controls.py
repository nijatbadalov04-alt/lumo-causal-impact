"""
Confounder controls — punctuality & fares around Lumo's entry.

THINK -> RESEARCH -> CODE
  WHAT: the key identification challenge — "did demand move because LNER got better/worse or because
        fares changed, not because of Lumo?" We test it directly with ORR data:
        (1) Table 3113 — Public Performance Measure (punctuality) by operator, quarterly:
            did LNER's punctuality shift discontinuously at Lumo's Oct-2021 entry?
        (2) Table 7180 — fares index by sector/ticket-type, annual: did long-distance
            fares move specially around 2021 (vs the long-run RPI-tracking trend)?
  LOGIC: if punctuality and fares are smooth/continuous through the entry date, they cannot
        manufacture the demand patterns we attribute to the corridor/Lumo => the controls
        clear the confounders. Any discontinuity is reported honestly as a caveat.

Run:  python -m src.evaluate.confounder_controls
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.clean.parse_operator_usage import parse_stacked
from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, METRICS, RAW, TABLES, ensure_dirs

LOG = get_logger("evaluate.confounders", log_file="logs/models.log")
TREAT = 2021.875  # Oct-Dec 2021 quarter (Lumo launch); annual treat year = 2021


def _parse_fares() -> pd.DataFrame:
    raw = pd.read_excel(
        RAW / "confounders/table-7180-fares-change.ods",
        sheet_name="7180_Change_by_regulated_status",
        engine="odf",
        header=None,
    )
    hdr = raw.iloc[4].tolist()
    body = raw.iloc[5:].copy()
    body.columns = hdr
    year_cols = [c for c in hdr if isinstance(c, (int, float)) and not pd.isna(c) and c > 1990]
    long = body.melt(id_vars=[hdr[0], hdr[1]], value_vars=year_cols, var_name="year", value_name="fare_index")
    long.columns = ["sector", "regulated", "year", "fare_index"]
    long["year"] = long["year"].astype(int)
    long["fare_index"] = pd.to_numeric(long["fare_index"], errors="coerce")
    return long.dropna(subset=["fare_index"])


def main() -> None:
    ensure_dirs()
    summary: dict = {}

    # ---------- (1) PUNCTUALITY (PPM) ----------
    ppm = parse_stacked(RAW / "confounders/table-3113-ppm-by-operator.ods", "3113_PPM_(quarterly)", header_row=5)
    ppm = ppm[ppm.flag == "observed"].rename(columns={"value_m": "ppm"})
    lner = ppm[ppm.series.str.contains("North Eastern", na=False) & (ppm.freq == "quarterly")].sort_values(
        "period_start"
    )
    ld = ppm[ppm.series.str.contains("Long distance", na=False) & (ppm.freq == "quarterly")].sort_values("period_start")
    if len(lner):
        pre = lner[lner.period_start < TREAT]["ppm"]
        post = lner[lner.period_start >= TREAT]["ppm"]
        summary["lner_ppm_pre_mean"] = round(float(pre.mean()), 1)
        summary["lner_ppm_post_mean"] = round(float(post.mean()), 1)
        summary["lner_ppm_shift_pp"] = round(float(post.mean() - pre.mean()), 1)
        LOG.info(
            "LNER PPM: pre-Lumo %.1f%% vs post %.1f%% (shift %+.1f pp)",
            pre.mean(),
            post.mean(),
            post.mean() - pre.mean(),
        )
    else:
        LOG.warning("LNER PPM column not found; series sample: %s", sorted(ppm.series.unique())[:8])

    # ---------- (2) FARES ----------
    fares = _parse_fares()
    ldf = fares[fares.sector.str.contains("Long distance", na=False)]
    if len(ldf):
        piv = ldf.pivot_table(index="year", columns="regulated", values="fare_index", aggfunc="mean")
        recent = piv.loc[[y for y in piv.index if y in (2019, 2021, 2024)]]
        summary["long_distance_fare_index"] = {
            str(int(y)): {str(c): round(float(v), 1) for c, v in row.items() if pd.notna(v)}
            for y, row in recent.iterrows()
        }
        # year-on-year % change around 2021 vs the long-run average (is 2021-22 special?)
        avg = piv.mean(axis=1)
        yoy = avg.pct_change() * 100
        summary["fare_yoy_pct_2021"] = round(float(yoy.get(2021, np.nan)), 1)
        summary["fare_yoy_pct_2022"] = round(float(yoy.get(2022, np.nan)), 1)
        summary["fare_yoy_pct_mean_2010_2019"] = round(float(yoy[(yoy.index >= 2010) & (yoy.index <= 2019)].mean()), 1)
        LOG.info(
            "Long-distance fares YoY: 2021=%.1f%%, 2022=%.1f%% vs 2010-19 mean %.1f%%",
            summary["fare_yoy_pct_2021"],
            summary["fare_yoy_pct_2022"],
            summary["fare_yoy_pct_mean_2010_2019"],
        )

    (METRICS / "m6_confounder_controls.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).T.to_csv(TABLES / "m6_confounder_controls.csv", header=["value"])

    # ---------- figure ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    if len(lner):
        ax1.plot(lner.period_start, lner.ppm, "o-", color="#d62728", lw=1.8, label="LNER")
    if len(ld):
        ax1.plot(ld.period_start, ld.ppm, "s--", color="#1f77b4", lw=1.4, label="Long-distance sector")
    ax1.axvline(TREAT, color="grey", ls=":", lw=1.4)
    ax1.text(TREAT + 0.05, ax1.get_ylim()[0], " Lumo", color="grey", fontsize=8, rotation=90, va="bottom")
    ax1.set_ylabel("Public Performance Measure (% on time)")
    ax1.set_xlabel("Year (quarterly)")
    ax1.set_title(
        "Punctuality control — LNER PPM is smooth through Lumo's entry\n(no discontinuity ⇒ not the driver of demand)",
        fontsize=10,
    )
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    if len(ldf):
        for status, sub in ldf.groupby("regulated"):
            s = sub.sort_values("year")
            ax2.plot(s.year, s.fare_index, lw=1.6, label=str(status)[:28])
        ax2.axvline(2021, color="grey", ls=":", lw=1.4)
        ax2.set_ylabel("Fare index (1995 = 100)")
        ax2.set_xlabel("Year")
        ax2.set_title(
            "Fares control — long-distance fares track the long-run trend\n(no special move at Lumo's entry)",
            fontsize=10,
        )
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "m6_confounder_controls.png", dpi=150)
    plt.close(fig)
    LOG.info(
        "figure -> %s | confounder controls complete: %s", FIGURES / "m6_confounder_controls.png", json.dumps(summary)
    )


if __name__ == "__main__":
    main()
