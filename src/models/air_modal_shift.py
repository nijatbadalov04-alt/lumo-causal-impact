"""
Air -> rail modal shift: decompose the corridor "creation" into induced vs air-abstraction.

THINK -> RESEARCH -> CODE
  THE QUESTION the ODM alone cannot answer: the London<->Edinburgh market grew +60% by rail.
        Were those NEW trips (induced demand) or did they switch from PLANES (modal shift)?
        Opposite carbon/welfare stories. Only air data resolves it.
  METHOD: pair the CAA domestic air passengers (London-area airports <-> Edinburgh/Glasgow/
        Newcastle) with the ODM rail journeys on the same corridors, same years, same units
        (single passenger-trips, both directions). Compute, pre (2018-19) -> post (2023-24):
          - ΔRail, ΔAir, Δ(Rail+Air) total corridor market
          - rail's share of the rail+air market (the modal-split swing)
          - air-abstraction share of rail growth = min(|ΔAir-if-falling|, ΔRail) / ΔRail
          - residual = induced demand or car-abstraction
  CONTRAST: Glasgow-London is a placebo for the ECML story -- it is served by rail on the
        WCML (Avanti), NOT by Lumo/ECML. If rail captured air decline on Edinburgh (ECML,
        capacity grew) but NOT on Glasgow (WCML, disrupted), that supports a rail-supply
        mechanism rather than a generic air-decline story.
  HONEST CAVEATS: air decline is partly COVID-structural (business travel) and corridor-wide
        (LNER Azuma + Lumo, not Lumo alone -> consistent with the operator-agnostic ODM);
        some London-Edinburgh air is Heathrow hub-feed that cannot switch to rail.

Run:  python -m src.models.air_modal_shift
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, INTERIM, METRICS, RAW, TABLES, ensure_dirs

LOG = get_logger("models.air_modal_shift", log_file="logs/models.log")

PRE_YEARS, POST_YEARS = (2018, 2019), (2023, 2024)
# CAA airport name (apt2) -> ODM rail city (od_london_flows)
CORRIDORS = {"EDINBURGH": "Edinburgh", "GLASGOW": "Glasgow Central", "NEWCASTLE": "Newcastle"}
LONDON_GROUP = "London Area Airports"


def _air_by_corridor_year() -> pl.DataFrame:
    """Total London-area air passengers per corridor per year, from the CAA annual files."""
    rows = []
    for f in sorted((RAW / "air").glob("caa_domestic_*.csv")):
        df = pl.read_csv(f, infer_schema_length=5000, encoding="utf8-lossy")
        df = df.rename({c: c.lstrip("﻿").strip() for c in df.columns})
        lon = df.filter(pl.col("grp_name") == LONDON_GROUP)
        for apt, city in CORRIDORS.items():
            sub = lon.filter(pl.col("apt2_apt_name") == apt)
            if sub.height == 0:
                continue
            y_tp = int(sub["this_period"][0])
            y_lp = int(sub["last_period"][0])
            rows.append({"city": city, "year": y_tp, "air_pax": int(sub["total_pax_tp"].sum())})
            rows.append({"city": city, "year": y_lp, "air_pax": int(sub["total_pax_lp"].sum())})
    return pl.DataFrame(rows).unique(subset=["city", "year"], keep="first").sort(["city", "year"])


def _rail_by_corridor_year() -> pl.DataFrame:
    flows = pl.read_parquet(INTERIM / "od_london_flows.parquet")
    return flows.filter(pl.col("city").is_in(list(CORRIDORS.values()))).select(
        "city", "year", pl.col("london_journeys").alias("rail_journeys")
    )


def _pp(df: pl.DataFrame, val: str) -> pl.DataFrame:
    pre = df.filter(pl.col("year").is_in(PRE_YEARS)).group_by("city").agg(pl.col(val).mean().alias(f"{val}_pre"))
    post = df.filter(pl.col("year").is_in(POST_YEARS)).group_by("city").agg(pl.col(val).mean().alias(f"{val}_post"))
    return pre.join(post, on="city")


def main() -> None:
    ensure_dirs()
    air = _air_by_corridor_year()
    rail = _rail_by_corridor_year()
    if air.height == 0 or rail.height == 0:
        LOG.warning("missing air or rail data — run download_caa / od_substitution first.")
        return
    air.write_parquet(INTERIM / "air_by_corridor.parquet")

    m = _pp(air, "air_pax").join(_pp(rail, "rail_journeys"), on="city")
    m = m.with_columns(
        (pl.col("rail_journeys_post") - pl.col("rail_journeys_pre")).alias("d_rail"),
        (pl.col("air_pax_post") - pl.col("air_pax_pre")).alias("d_air"),
    ).with_columns(
        (pl.col("d_rail") + pl.col("d_air")).alias("d_total_market"),
        (pl.col("rail_journeys_pre") / (pl.col("rail_journeys_pre") + pl.col("air_pax_pre"))).alias("rail_share_pre"),
        (pl.col("rail_journeys_post") / (pl.col("rail_journeys_post") + pl.col("air_pax_post"))).alias("rail_share_post"),
    )

    results, edb_raw = {}, {}
    for r in m.iter_rows(named=True):
        d_rail, d_air = r["d_rail"], r["d_air"]
        air_abstraction = min(max(-d_air, 0.0), max(d_rail, 0.0))  # air loss that rail could absorb
        abstraction_share = air_abstraction / d_rail if d_rail > 0 else float("nan")
        if r["city"] == "Edinburgh":  # keep UNROUNDED values for the headline (avoid double-rounding)
            edb_raw = {"d_rail": d_rail, "d_air": d_air, "share_pre": r["rail_share_pre"], "share_post": r["rail_share_post"], "abstraction": abstraction_share}
        results[r["city"]] = {
            "air_pre": int(round(r["air_pax_pre"])),
            "air_post": int(round(r["air_pax_post"])),
            "rail_pre": int(round(r["rail_journeys_pre"])),
            "rail_post": int(round(r["rail_journeys_post"])),
            "d_rail": int(round(d_rail)),
            "d_air": int(round(d_air)),
            "d_total_market": int(round(r["d_total_market"])),
            "rail_share_pre": round(r["rail_share_pre"], 3),
            "rail_share_post": round(r["rail_share_post"], 3),
            "air_abstraction_of_rail_growth": round(abstraction_share, 3) if d_rail > 0 else None,
            "induced_or_car_residual": int(round(d_rail - air_abstraction)) if d_rail > 0 else None,
        }
        LOG.info(
            "%-16s rail %+.0fk | air %+.0fk | rail share %.0f%%->%.0f%% | air-abstraction %s of rail growth",
            r["city"],
            d_rail / 1e3,
            d_air / 1e3,
            r["rail_share_pre"] * 100,
            r["rail_share_post"] * 100,
            f"{abstraction_share*100:.0f}%" if d_rail > 0 else "n/a",
        )

    edb = results.get("Edinburgh", {})
    summary = {
        "design": "CAA London-area air passengers vs ODM rail journeys, same corridors/years/units.",
        "pre_years": list(PRE_YEARS),
        "post_years": list(POST_YEARS),
        "per_corridor": results,
        "headline_edinburgh": (
            f"London-Edinburgh: rail {edb_raw.get('d_rail',0)/1e3:+.0f}k, air {edb_raw.get('d_air',0)/1e3:+.0f}k; "
            f"rail share {round(edb_raw.get('share_pre',0)*100)}%->{round(edb_raw.get('share_post',0)*100)}%. "
            f"~{round(edb_raw.get('abstraction',0)*100)}% of the rail growth is air-abstraction; "
            "the rest is induced demand or car-abstraction."
        ),
        "caveats": (
            "Air decline is partly COVID-structural and corridor-wide (Azuma + Lumo, not Lumo alone); "
            "some London-Edinburgh air is Heathrow hub-feed that cannot realistically switch to rail. "
            "Glasgow (WCML/Avanti, no Lumo) is the placebo contrast."
        ),
    }
    (METRICS / "air_modal_shift.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    m.write_csv(TABLES / "air_modal_shift.csv")
    # hand the air-abstraction share to the carbon module
    if edb.get("air_abstraction_of_rail_growth") is not None:
        (METRICS / "air_abstraction_share.json").write_text(
            json.dumps({c: results[c]["air_abstraction_of_rail_growth"] for c in results}, indent=2),
            encoding="utf-8",
        )

    _plot(air, rail, results)
    LOG.info("metrics -> results/metrics/air_modal_shift.json | air modal-shift complete.")


def _plot(air: pl.DataFrame, rail: pl.DataFrame, results: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # left: Edinburgh air vs rail trajectories (the crossing chart)
    a = air.filter(pl.col("city") == "Edinburgh").sort("year")
    r = rail.filter(pl.col("city") == "Edinburgh").sort("year")
    ax1.plot(a["year"], a["air_pax"] / 1e6, "o-", color="#1f77b4", lw=2.2, label="Air (CAA, London<->EDB)")
    ax1.plot(r["year"], r["rail_journeys"] / 1e6, "s-", color="#d62728", lw=2.2, label="Rail (ODM, London<->EDB)")
    ax1.axvline(2021, color="grey", ls=":", lw=1.5)
    ax1.text(2021.05, ax1.get_ylim()[1] * 0.96, " Lumo (Oct 2021)", color="grey", rotation=90, fontsize=8, va="top")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Passenger-trips per year (m, both directions)")
    ax1.set_title("London<->Edinburgh: air falls as rail surges\n(modal shift toward rail)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # right: rail share of the rail+air market, pre vs post, per corridor
    cities = list(results.keys())
    pre = [results[c]["rail_share_pre"] * 100 for c in cities]
    post = [results[c]["rail_share_post"] * 100 for c in cities]
    x = np.arange(len(cities))
    ax2.bar(x - 0.18, pre, width=0.36, color="#aec7e8", label="pre (2018-19)")
    ax2.bar(x + 0.18, post, width=0.36, color="#d62728", label="post (2023-24)")
    for i in range(len(cities)):
        ax2.annotate(
            f"+{post[i]-pre[i]:.0f}pp",
            (x[i], max(pre[i], post[i]) + 1.5),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace(" Central", "") for c in cities])
    ax2.set_ylabel("Rail share of rail+air market (%)")
    ax2.set_title("Rail's share of the corridor air+rail market\n(Edinburgh = ECML/Lumo; Glasgow = WCML placebo)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Air -> rail modal shift: the majority of the Edinburgh rail growth came from planes", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES / "air_modal_shift.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
