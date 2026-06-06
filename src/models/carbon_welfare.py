"""
Carbon / welfare quantification of the Lumo-corridor market growth.

THINK -> RESEARCH -> CODE
  QUESTION: the ODM showed the London<->Edinburgh market grew ~+60% (+0.83m journeys/yr) and
        London<->Newcastle ~+20% (+0.26m/yr) after Lumo. What is the CO2 consequence? That
        depends ENTIRELY on where the new journeys came from:
          - abstracted from AIR  -> big saving  (rail 13.8 vs air 165.0 kg/pax on Edinburgh)
          - abstracted from CAR  -> saving       (rail 13.8 vs car 132.4 kg/pax)
          - INDUCED (new trips)  -> a COST       (+13.8 kg/pax of rail that would not exist)
  METHOD: take the per-OD-pair emissions by mode (GTD), and the ODM journey growth (delta),
        and compute net CO2 under end-member scenarios. The robust, assumption-light result
        is the BREAK-EVEN INDUCED SHARE: the fraction of new journeys that could be purely
        induced before the corridor stops being net carbon-beneficial. Because long-distance
        rail is ~12x cleaner per passenger than flying this route, abstraction dominates
        unless almost everything is induced -- a result that holds without knowing the exact
        modal-shift split (which the CAA air test, src/models/air_modal_shift.py, pins down).

  HONEST SCOPE: ODM journeys are operator-agnostic, so this is the carbon of the *corridor
        market growth that Lumo catalysed*, not of Lumo's trains alone. Emissions are a
        2025-26 per-pair snapshot (GTD). 1 journey := 1 one-way passenger trip.

Run:  python -m src.models.carbon_welfare
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, INTERIM, METRICS, TABLES, ensure_dirs

LOG = get_logger("models.carbon_welfare", log_file="logs/models.log")

# Lumo-served markets that GREW (the carbon story); Stevenage is commuter & shrank -> excluded
LUMO_GROWTH_CITIES = ["Edinburgh", "Newcastle"]
PRE_YEARS, POST_YEARS = (2018, 2019), (2023, 2024)

# UK average car: ~2.2 tCO2e/yr (DfT/BEIS per-vehicle); for a relatable equivalent only.
TONNES_PER_CAR_YEAR = 2.2
# When abstracted journeys are split between modes, baseline assumption = half air / half car.
# (Edinburgh-London is one of the UK's largest domestic AIR markets, so this is conservative.)
BASELINE_AIR_FRACTION_AMONG_ABSTRACTED = 0.5


def _growth_table() -> pl.DataFrame:
    """Per-city pre/post journeys and delta from the ODM London-flows panel."""
    flows = pl.read_parquet(INTERIM / "od_london_flows.parquet")
    pre = (
        flows.filter(pl.col("year").is_in(PRE_YEARS))
        .group_by("city")
        .agg(pl.col("london_journeys").mean().alias("pre"))
    )
    post = (
        flows.filter(pl.col("year").is_in(POST_YEARS))
        .group_by("city")
        .agg(pl.col("london_journeys").mean().alias("post"))
    )
    return (
        pre.join(post, on="city")
        .with_columns((pl.col("post") - pl.col("pre")).alias("delta"))
        .with_columns((pl.col("post") / pl.col("pre")).alias("recovery"))
    )


def _scenarios(delta: float, e: dict) -> dict:
    """Net CO2 (tonnes/yr) for `delta` new journeys given per-pax emissions dict `e`."""
    sav_air = e["saving_rail_over_air_kg"]
    sav_car = e["saving_rail_over_car_kg"]
    rail = e["rail_standard_kg"]
    kg_to_t = 1e-3
    a = BASELINE_AIR_FRACTION_AMONG_ABSTRACTED
    mixed_abstracted = a * sav_air + (1 - a) * sav_car  # saving per abstracted journey (air/car mix)
    # break-even induced share s*: (1-s)*M = s*rail  ->  s* = M/(M+rail)
    breakeven_induced = mixed_abstracted / (mixed_abstracted + rail)
    return {
        "delta_journeys_per_yr": int(round(delta)),
        "saving_if_all_from_air_tonnes": delta * sav_air * kg_to_t,
        "saving_if_all_from_car_tonnes": delta * sav_car * kg_to_t,
        "cost_if_all_induced_tonnes": -delta * rail * kg_to_t,  # negative = net emissions ADDED
        "net_if_half_air_half_car_abstracted_tonnes": delta * mixed_abstracted * kg_to_t,
        "breakeven_induced_share": breakeven_induced,
        "_per_pax": {"rail_kg": rail, "saving_vs_air_kg": sav_air, "saving_vs_car_kg": sav_car},
    }


def _caa_informed(results: dict) -> dict:
    """If the CAA air test ran, turn the measured air-abstraction share into a point estimate.

    For each city: air_abstraction_share s of the rail growth is abstracted from air (big
    saving); the residual (1-s) is induced or car-abstracted. We bracket the residual:
    'induced' (a small carbon COST, conservative) vs 'car' (a saving, optimistic).
    """
    share_path = METRICS / "air_abstraction_share.json"
    if not share_path.exists():
        return {}
    shares = json.loads(share_path.read_text(encoding="utf-8"))
    out, tot_lo, tot_hi = {}, 0.0, 0.0
    for city, sc in results.items():
        s = shares.get(city)
        if s is None:
            continue
        delta = sc["delta_journeys_per_yr"]
        air_j, resid = s * delta, (1 - s) * delta
        sav_air, sav_car, rail = (
            sc["_per_pax"]["saving_vs_air_kg"],
            sc["_per_pax"]["saving_vs_car_kg"],
            sc["_per_pax"]["rail_kg"],
        )
        net_resid_induced = (air_j * sav_air - resid * rail) / 1e6  # ktonnes (kg->t /1e3, t->kt /1e3)
        net_resid_car = (air_j * sav_air + resid * sav_car) / 1e6
        out[city] = {
            "air_abstraction_share": round(s, 3),
            "net_saving_ktonnes_residual_induced": round(net_resid_induced, 1),
            "net_saving_ktonnes_residual_car": round(net_resid_car, 1),
        }
        tot_lo += net_resid_induced
        tot_hi += net_resid_car
    if out:
        out["corridor_total_ktonnes"] = {
            "central_conservative_residual_induced": round(tot_lo, 1),
            "upper_residual_car": round(tot_hi, 1),
        }
    return out


def _carbon_monte_carlo(results: dict, seed: int = 20251025, n: int = 100_000) -> dict:
    """Propagate the two real uncertainties into a carbon CI (instead of a point estimate).

    For each Lumo-growth city, the rail growth ΔR decomposes precisely into market-expansion
    (Δtotal market) + from-air (the air decline). The uncertainties are:
      - the air-abstraction share f_air around its MEASURED value (±0.10 sensitivity), and
      - whether the non-air residual is induced (carbon COST) or car-abstracted (SAVING) — a
        fraction we are honestly ignorant about, so ~U(0,1).
    Net CO2 per draw = ΔR·[ f_air·sav_air + (1-f_air)·( ind·(-rail) + (1-ind)·sav_car ) ].
    """
    share_path = METRICS / "air_abstraction_share.json"
    if not share_path.exists():
        return {}
    shares = json.loads(share_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    total = np.zeros(n)
    per_city = {}
    for city, sc in results.items():
        s = shares.get(city)
        if s is None:
            continue
        delta = sc["delta_journeys_per_yr"]
        sav_air, sav_car, rail = (
            sc["_per_pax"]["saving_vs_air_kg"],
            sc["_per_pax"]["saving_vs_car_kg"],
            sc["_per_pax"]["rail_kg"],
        )
        f_air = np.clip(rng.uniform(s - 0.10, s + 0.10, n), 0.0, 1.0)
        ind = rng.uniform(0.0, 1.0, n)
        net = delta * (f_air * sav_air + (1 - f_air) * (ind * (-rail) + (1 - ind) * sav_car)) / 1e6  # ktonnes
        total += net
        per_city[city] = {
            "p05_ktonnes": round(float(np.percentile(net, 5)), 1),
            "p50_ktonnes": round(float(np.percentile(net, 50)), 1),
            "p95_ktonnes": round(float(np.percentile(net, 95)), 1),
        }
    if per_city:
        per_city["corridor_total_ktonnes"] = {
            "p05": round(float(np.percentile(total, 5)), 1),
            "p50": round(float(np.percentile(total, 50)), 1),
            "p95": round(float(np.percentile(total, 95)), 1),
        }
    return per_city


def main() -> None:
    ensure_dirs()
    growth = _growth_table()
    carbon = pl.read_parquet(INTERIM / "carbon_kgx_pairs.parquet")
    cmap = {r["city"]: r for r in carbon.iter_rows(named=True)}

    results, total = {}, {
        "saving_if_all_from_air_tonnes": 0.0,
        "saving_if_all_from_car_tonnes": 0.0,
        "cost_if_all_induced_tonnes": 0.0,
        "net_if_half_air_half_car_abstracted_tonnes": 0.0,
    }
    for city in LUMO_GROWTH_CITIES:
        g = growth.filter(pl.col("city") == city)
        if g.height == 0 or city not in cmap:
            LOG.warning("  no growth/carbon match for %s — skipping", city)
            continue
        delta = float(g["delta"][0])
        sc = _scenarios(delta, cmap[city])
        results[city] = sc
        for k in total:
            total[k] += sc[k]
        LOG.info(
            "%-10s +%s journeys/yr | air-shift save %.0f kt | induced cost %.0f kt | break-even induced %.0f%%",
            city,
            f"{sc['delta_journeys_per_yr']:,}",
            sc["saving_if_all_from_air_tonnes"] / 1e3,
            -sc["cost_if_all_induced_tonnes"] / 1e3,
            sc["breakeven_induced_share"] * 100,
        )

    summary = {
        "scope": (
            "Net CO2e of the ODM corridor market growth Lumo catalysed (operator-agnostic). "
            "Per-pair emissions from GTD (2025-26 snapshot); 1 journey = 1 one-way pax trip."
        ),
        "pre_years": list(PRE_YEARS),
        "post_years": list(POST_YEARS),
        "per_city": results,
        "corridor_total": total,
        "caa_informed_central": _caa_informed(results),
        "carbon_monte_carlo_ci": _carbon_monte_carlo(results),
        "corridor_total_air_saving_equiv_cars_off_road": round(
            total["saving_if_all_from_air_tonnes"] / TONNES_PER_CAR_YEAR
        ),
        "headline": (
            "Long-distance rail is ~12x cleaner per passenger than flying this corridor, so the "
            "market growth is net carbon-BENEFICIAL unless the large majority of new journeys are "
            "purely induced. The CAA air test estimates the air-abstraction share to pin the point."
        ),
    }
    (METRICS / "carbon_welfare.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # tidy table
    rows = []
    for city, sc in results.items():
        rows.append(
            {
                "city": city,
                "delta_journeys_per_yr": sc["delta_journeys_per_yr"],
                "rail_kg_per_pax": sc["_per_pax"]["rail_kg"],
                "saving_vs_air_kg_per_pax": sc["_per_pax"]["saving_vs_air_kg"],
                "all_air_saving_ktonnes": round(sc["saving_if_all_from_air_tonnes"] / 1e3, 1),
                "all_car_saving_ktonnes": round(sc["saving_if_all_from_car_tonnes"] / 1e3, 1),
                "all_induced_cost_ktonnes": round(sc["cost_if_all_induced_tonnes"] / 1e3, 1),
                "breakeven_induced_share_pct": round(sc["breakeven_induced_share"] * 100, 1),
            }
        )
    pl.DataFrame(rows).write_csv(TABLES / "carbon_welfare.csv")

    # figure: net CO2 by scenario (corridor total)
    _plot(total, results)
    LOG.info(
        "CORRIDOR total: all-air %.0f kt saved | all-induced %.0f kt added | ~%s cars off road (air case)",
        total["saving_if_all_from_air_tonnes"] / 1e3,
        -total["cost_if_all_induced_tonnes"] / 1e3,
        f"{summary['corridor_total_air_saving_equiv_cars_off_road']:,}",
    )
    LOG.info("metrics -> results/metrics/carbon_welfare.json | carbon/welfare complete.")


def _plot(total: dict, results: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # left: corridor net CO2 by scenario
    labels = ["All from\nair", "All from\ncar", "Half air /\nhalf car", "All\ninduced"]
    vals = [
        total["saving_if_all_from_air_tonnes"] / 1e3,
        total["saving_if_all_from_car_tonnes"] / 1e3,
        total["net_if_half_air_half_car_abstracted_tonnes"] / 1e3,
        total["cost_if_all_induced_tonnes"] / 1e3,
    ]
    colours = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
    ax1.bar(labels, vals, color=colours, alpha=0.85)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_ylabel("Net CO2e, kilotonnes / year  (+saved / −added)")
    ax1.set_title(
        "Carbon impact of the Edinburgh+Newcastle\nLondon-market growth, by composition scenario",
        fontsize=10,
    )
    for i, v in enumerate(vals):
        ax1.text(i, v + (3 if v >= 0 else -3), f"{v:,.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # right: break-even induced share per city
    cities = list(results.keys())
    be = [results[c]["breakeven_induced_share"] * 100 for c in cities]
    ax2.barh(cities, be, color="#1f77b4", alpha=0.85)
    for i, v in enumerate(be):
        ax2.text(v - 2, i, f"{v:.0f}%", ha="right", va="center", color="white", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Break-even induced share (%)")
    ax2.set_title(
        "New journeys could be this % purely induced\nbefore the growth stops cutting carbon",
        fontsize=10,
    )
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Open-access rail growth is carbon-beneficial across almost the whole plausible range",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES / "carbon_welfare.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
