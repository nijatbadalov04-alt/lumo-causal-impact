"""
Study-area map: the East Coast Main Line (treated corridor) and the West Coast
Glasgow placebo, drawn from NaPTAN station coordinates.

  WHAT: a setting map for the paper. The ECML London-Edinburgh corridor is the
        treated route (Lumo entered 25 Oct 2021); the WCML London-Glasgow corridor
        is the air-decline placebo (air fell there too, but with no Lumo entry rail
        did not capture it). Background grey points are the GB rail network, i.e.
        the pool from which the off-corridor comparator flows are drawn.
  IN  : data/interim/station_covariates.parquet  (crs, station_name, lat, lon)
  OUT : results/figures/study_area_map.png
"""
from pathlib import Path
import logging
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LOG = logging.getLogger("study_area_map")
ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
FIGURES = ROOT / "results" / "figures"

# East Coast Main Line, south -> north (London King's Cross to Edinburgh)
ECML = ["KGX", "SVG", "PBO", "GRA", "RET", "DON", "YRK", "NTR", "DAR", "DHM",
        "NCL", "MPT", "ALM", "BWK", "EDB"]
LUMO = {"EDB", "NCL", "MPT", "SVG"}                 # the four treated Lumo stops
# West Coast Main Line placebo, south -> north (London Euston to Glasgow)
WCML = ["EUS", "BHM", "CRE", "PRE", "LAN", "CAR", "GLC"]

LABELS = {
    "KGX": "London", "SVG": "Stevenage", "YRK": "York", "DAR": "Darlington",
    "NCL": "Newcastle", "MPT": "Morpeth", "EDB": "Edinburgh",
    "BHM": "Birmingham", "CAR": "Carlisle", "GLC": "Glasgow",
}
RED, BLUE = "#c0392b", "#1f4e9c"


def _coords(lookup, crs_list):
    return [(c, lookup[c][0], lookup[c][1]) for c in crs_list if c in lookup]


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = pl.read_parquet(INTERIM / "station_covariates.parquet").drop_nulls(["lat", "lon"])
    gb = df.filter((pl.col("lon") > -8.5) & (pl.col("lon") < 2.0)
                   & (pl.col("lat") > 49.8) & (pl.col("lat") < 59.0))
    lookup = {r["crs"]: (r["lon"], r["lat"]) for r in df.iter_rows(named=True)}
    ecml, wcml = _coords(lookup, ECML), _coords(lookup, WCML)

    fig, ax = plt.subplots(figsize=(6.4, 8.2))
    ax.scatter(gb["lon"], gb["lat"], s=3, c="0.80", alpha=0.6, linewidths=0, zorder=1)

    # West Coast placebo
    ax.plot([p[1] for p in wcml], [p[2] for p in wcml], "--", color=BLUE, lw=2.0, zorder=3)
    gc = next(p for p in wcml if p[0] == "GLC")
    ax.scatter([gc[1]], [gc[2]], marker="^", s=130, c=BLUE, edgecolors="white", linewidths=0.8, zorder=6)

    # East Coast treated corridor
    ax.plot([p[1] for p in ecml], [p[2] for p in ecml], "-", color=RED, lw=2.4, zorder=3)
    for c, x, y in ecml:
        if c in LUMO:
            ax.scatter([x], [y], marker="*", s=240, c=RED, edgecolors="white", linewidths=0.8, zorder=6)
        else:
            ax.scatter([x], [y], marker="o", s=22, c=RED, edgecolors="white", linewidths=0.4, zorder=5)
    kgx = next(p for p in ecml if p[0] == "KGX")
    ax.scatter([kgx[1]], [kgx[2]], marker="s", s=72, c="black", edgecolors="white", linewidths=0.6, zorder=7)

    allpts = {c: (x, y) for c, x, y in ecml + wcml}
    for c, name in LABELS.items():
        if c not in allpts:
            continue
        x, y = allpts[c]
        left = c in ("YRK", "DAR", "NCL", "MPT", "CAR")
        dx = -0.14 if left else 0.14
        col = RED if (c in LUMO or c == "KGX") else (BLUE if c in ("GLC", "BHM", "CAR") else "0.25")
        ax.annotate(name, (x, y), xytext=(x + dx, y), ha=("right" if left else "left"),
                    va="center", fontsize=8.5,
                    fontweight=("bold" if (c in LUMO or c in ("KGX", "GLC")) else "normal"), color=col)

    ax.set_aspect(1.0 / np.cos(np.radians(54.5)))
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Study area: the East Coast treated corridor and the\n"
                 "West Coast (Glasgow) placebo", fontsize=11, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="0.92", lw=0.6)

    legend = [
        Line2D([0], [0], color=RED, lw=2.4, marker="*", markersize=12, markerfacecolor=RED,
               markeredgecolor="white", label="East Coast Main Line (treated); ★ Lumo stop"),
        Line2D([0], [0], color=BLUE, lw=2.0, ls="--", marker="^", markersize=9, markerfacecolor=BLUE,
               markeredgecolor="white", label="West Coast / Glasgow (air-decline placebo)"),
        Line2D([0], [0], color="0.80", lw=0, marker="o", markersize=5,
               label="GB rail network (comparator pool)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=7.6, frameon=True, framealpha=0.92)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "study_area_map.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    LOG.info("study area map -> %s (ECML %d/%d, WCML %d/%d)", out, len(ecml), len(ECML), len(wcml), len(WCML))


if __name__ == "__main__":
    main()
