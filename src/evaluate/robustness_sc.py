"""
Robustness / weakness investigation for the M3 synthetic-control result.

THINK -> RESEARCH -> CODE
  Three threats to the Newcastle +17% headline, tested empirically:

  (A) PLACEBO-IN-TIME — pretend Lumo launched 2018 (fit 2004-2017, check 2018-19,
      pre-COVID). A real effect must be ~0 here. If it isn't, the method invents
      effects.

  (B) DONOR CONTAMINATION by concurrent shocks (the dangerous one):
      - Avanti West Coast meltdown (2022-23): WCML donors were *depressed* ⇒ would
        push the synthetic DOWN ⇒ INFLATE the measured Lumo effect.
      - Elizabeth line opening (May 2022): Elizabeth donors were *inflated* ⇒ push
        the synthetic UP ⇒ DEFLATE the effect.
      We re-estimate Newcastle excluding donors whose ORR `facility_owner` is
      Avanti / Elizabeth line, and report how much the effect moves. Stable ⇒ robust.

  (C) COMMON ECML-RECOVERY CONFOUND (the decisive one): estimate the SAME SC for
      ECML *through* stations that Lumo does NOT serve (York, Doncaster, Darlington,
      Grantham). If they show a large positive 'effect' too, then Newcastle's gap is
      (partly) a corridor-wide recovery the off-ECML donors miss — NOT Lumo-specific.
      If they're ~0, the Newcastle effect is Lumo-specific. This is the cleanest
      falsification test we can run without operator-level data.

Run:  python -m src.evaluate.robustness_sc
Out:  results/tables/m3_robustness_sc.csv, results/metrics/m3_robustness_sc.json,
      results/figures/m3_placebo_in_time.png, m3_spillover_ecml_through.png
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.models.classical.synthetic_control import (
    K_DONORS,
    SIZE_BAND,
    build_outcome_matrix,
    run_sc,
)
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, METRICS, PROCESSED, TABLES, ensure_dirs

LOG = get_logger("evaluate.robustness_sc", log_file="logs/models.log")


def select_donors(panel, units, tcrs, years, treat_year, exclude_owner_regex=None):
    """K nearest clean donors by pre-treatment log-trajectory; optional owner exclusion."""
    base_map = dict(units.select("crs", "baseline_ee_2019").drop_nulls().iter_rows())
    tbase = base_map.get(tcrs)
    donors = (
        units.filter(pl.col("role") == "donor_clean")
        .select("crs", "baseline_ee_2019", "facility_owner")
        .drop_nulls("baseline_ee_2019")
    )
    if exclude_owner_regex:
        donors = donors.filter(~pl.col("facility_owner").fill_null("").str.contains(f"(?i){exclude_owner_regex}"))
    pool = donors.filter(
        (pl.col("baseline_ee_2019") >= SIZE_BAND[0] * tbase) & (pl.col("baseline_ee_2019") <= SIZE_BAND[1] * tbase)
    )["crs"].to_list()
    yarr = np.array(years)
    pre = yarr < treat_year
    M = build_outcome_matrix(panel, [tcrs] + pool, years)
    y1, Y0full = M[0], M[1:]
    dist = np.sqrt(((Y0full[:, pre] - y1[pre]) ** 2).sum(axis=1))
    order = np.argsort(dist)[:K_DONORS]
    return [pool[i] for i in order]


def sc_for(panel, units, tcrs, years, treat_year, exclude_owner_regex=None):
    donor_crs = select_donors(panel, units, tcrs, years, treat_year, exclude_owner_regex)
    M = build_outcome_matrix(panel, [tcrs] + donor_crs, years)
    res = run_sc(M[0], M[1:], np.array(years), treat_year)
    post = np.array(years) >= treat_year
    avg = float(np.exp(res["gap"][post].mean()) - 1)
    return {
        "avg_effect_pct": 100 * avg,
        "pre_rmspe": res["pre_rmspe"],
        "n_donors": len(donor_crs),
        "gap": res["gap"],
        "synth": res["synth"],
        "y1": M[0],
        "ratio": res["ratio"],
    }


def main() -> None:
    ensure_dirs()
    cfg = load_config("base")
    treat = int(cfg["treatments"]["lumo"]["treat_year_start"])
    lennon_min = int(cfg["panel"]["lennon_era_min"])
    ymax = int(cfg["panel"]["year_max"])
    full_years = list(range(lennon_min, ymax + 1))

    panel = pl.read_parquet(PROCESSED / "panel.parquet")
    units = pl.read_parquet(PROCESSED / "units.parquet")
    rows = []

    # ---------- (A) PLACEBO-IN-TIME: fake 2018, pre-COVID window only ----------
    LOG.info("=== (A) Placebo-in-time (fake 2018, years 2004-2019) ===")
    pit_years = list(range(lennon_min, 2020))  # 2004..2019 (excludes COVID + real Lumo)
    pit_fig = []
    for tcrs in ["NCL", "SVG", "MPT"]:
        r = sc_for(panel, units, tcrs, pit_years, 2018)
        eff = {int(y): round(100 * (np.exp(g) - 1), 1) for y, g in zip(pit_years, r["gap"]) if y >= 2018}
        LOG.info(
            "  %s: fake-2018 avg effect = %+.1f%% (pre_RMSPE=%.4f)  per-yr=%s",
            tcrs,
            r["avg_effect_pct"],
            r["pre_rmspe"],
            eff,
        )
        rows.append(
            {
                "test": "placebo_in_time_2018",
                "unit": tcrs,
                "effect_pct": round(r["avg_effect_pct"], 1),
                "pre_rmspe": round(r["pre_rmspe"], 4),
            }
        )
        pit_fig.append((tcrs, pit_years, r["gap"]))

    # ---------- (B) DONOR CONTAMINATION sensitivity (Newcastle) ----------
    LOG.info("=== (B) Donor-pool sensitivity — concurrent-shock contamination ===")
    for label, rgx in [
        ("baseline", None),
        ("excl_Avanti", "avanti"),
        ("excl_Elizabeth", "elizabeth"),
        ("excl_both", "avanti|elizabeth"),
    ]:
        r = sc_for(panel, units, "NCL", full_years, treat, exclude_owner_regex=rgx)
        LOG.info(
            "  NCL [%-14s]: effect=%+.1f%%  pre_RMSPE=%.4f  n_donors=%d",
            label,
            r["avg_effect_pct"],
            r["pre_rmspe"],
            r["n_donors"],
        )
        rows.append(
            {
                "test": f"donor_sensitivity_{label}",
                "unit": "NCL",
                "effect_pct": round(r["avg_effect_pct"], 1),
                "pre_rmspe": round(r["pre_rmspe"], 4),
            }
        )

    # ---------- (C) ECML-through 'placebo' — common-recovery confound ----------
    LOG.info("=== (C) ECML-through spillover/confound test (Lumo does NOT serve these) ===")
    spill_fig = []
    for tcrs, tname in [("YRK", "York"), ("DON", "Doncaster"), ("DAR", "Darlington"), ("GRA", "Grantham")]:
        r = sc_for(panel, units, tcrs, full_years, treat)
        LOG.info("  %s (%s): SC 'effect' = %+.1f%%  pre_RMSPE=%.4f", tcrs, tname, r["avg_effect_pct"], r["pre_rmspe"])
        rows.append(
            {
                "test": "ecml_through_confound",
                "unit": tcrs,
                "effect_pct": round(r["avg_effect_pct"], 1),
                "pre_rmspe": round(r["pre_rmspe"], 4),
            }
        )
        spill_fig.append((tname, full_years, r["gap"]))

    pl.DataFrame(rows).write_csv(TABLES / "m3_robustness_sc.csv")
    (METRICS / "m3_robustness_sc.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _plot_gaps(
        pit_fig, 2018, "Placebo-in-time (fake 2018 launch) — effect must be ~0", FIGURES / "m3_placebo_in_time.png"
    )
    _plot_gaps(
        spill_fig,
        treat,
        "ECML through-stations (NOT Lumo-served) — confound test",
        FIGURES / "m3_spillover_ecml_through.png",
    )
    LOG.info("robustness investigation complete -> %s", TABLES / "m3_robustness_sc.csv")


def _plot_gaps(series, treat_year, title, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, years, gap in series:
        ax.plot(years, (np.exp(np.asarray(gap)) - 1) * 100, "o-", lw=1.8, label=name)
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(treat_year, color="#2166ac", ls=":", lw=1.4)
    ax.set_ylabel("Gap (obs − synthetic, %)")
    ax.set_xlabel("Financial year (start)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOG.info("figure -> %s", path)


if __name__ == "__main__":
    main()
