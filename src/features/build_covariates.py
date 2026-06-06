"""
Station covariates — distance-to-London from NaPTAN coordinates (for RQ2 / Causal Forest).

THINK -> RESEARCH -> CODE
  WHAT: NaPTAN (DfT open data) gives every rail station's lat/lon. We name-match it to the
        ORR stations and compute great-circle distance to London King's Cross — the single
        most relevant heterogeneity covariate (Lumo competes on London long-distance routes).
  WHY : upgrades RQ2 from proxy covariates (season-share, KGX-exposure) to an exogenous
        geographic one; feeds the heterogeneity model.
  MATCH: NaPTAN CommonName ("Newcastle Rail Station") -> normalised -> joined to ORR
        station_name. Name-matching is imperfect (~report the rate); major stations (all
        treated + most donors) match. CRS isn't in NaPTAN's node export, hence name-match.

Run:  python -m src.features.build_covariates
Out:  data/interim/station_covariates.parquet
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import requests

from src.utils.logging_setup import get_logger
from src.utils.paths import INTERIM, RAW, ensure_dirs

LOG = get_logger("features.build_covariates", log_file="logs/features.log")

KGX_LAT, KGX_LON = 51.5308, -0.1238  # London King's Cross
NAPTAN_URL = "https://naptan.api.dft.gov.uk/v1/access-nodes?dataFormat=csv"


def _ensure_naptan() -> object:
    """Return the rail-only NaPTAN CSV; download the national file and filter RLY if absent."""
    dest = RAW / "geography/naptan_rail.csv"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    full = RAW / "geography/naptan_all.csv"
    LOG.info("downloading NaPTAN national nodes (~100MB) to filter rail stations...")
    with requests.get(NAPTAN_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        with full.open("wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    pl.read_csv(full, infer_schema_length=0).filter(pl.col("StopType") == "RLY").write_csv(dest)
    LOG.info("filtered NaPTAN rail -> %s", dest)
    return dest


def _norm(name: str) -> str:
    s = str(name).lower()
    s = re.sub(r"\b(rail|railway)\s+station\b", "", s)
    s = re.sub(r"\(.*?\)", "", s)  # drop "(Herts)" etc.
    s = s.replace("&", "and").replace("'", "").replace(".", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _haversine(lat, lon, lat0, lon0):
    R = 6371.0
    p1, p2 = np.radians(lat), np.radians(lat0)
    dphi, dl = np.radians(lat0 - lat), np.radians(lon0 - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main() -> None:
    ensure_dirs()
    nap = pl.read_csv(_ensure_naptan(), infer_schema_length=2000)
    nap = nap.select(
        pl.col("CommonName").alias("nap_name"),
        pl.col("Latitude").cast(pl.Float64, strict=False).alias("lat"),
        pl.col("Longitude").cast(pl.Float64, strict=False).alias("lon"),
    ).drop_nulls()
    nap = nap.with_columns(pl.col("nap_name").map_elements(_norm, return_dtype=pl.Utf8).alias("key")).unique(
        subset="key", keep="first"
    )

    meta = pl.read_parquet(INTERIM / "station_meta.parquet").select("crs", "station_name")
    meta = meta.filter(pl.col("crs").str.contains(r"^[A-Z]{3}$")).with_columns(
        pl.col("station_name").map_elements(_norm, return_dtype=pl.Utf8).alias("key")
    )

    joined = meta.join(nap.select("key", "lat", "lon"), on="key", how="left")
    matched = joined.drop_nulls(["lat", "lon"])
    rate = matched.height / meta.height
    LOG.info("NaPTAN name-match: %d/%d ORR stations (%.0f%%)", matched.height, meta.height, 100 * rate)

    out = matched.with_columns(
        pl.struct(["lat", "lon"])
        .map_elements(lambda r: float(_haversine(r["lat"], r["lon"], KGX_LAT, KGX_LON)), return_dtype=pl.Float64)
        .alias("distance_to_london_km")
    ).select("crs", "station_name", "lat", "lon", "distance_to_london_km")
    out.write_parquet(INTERIM / "station_covariates.parquet", compression="zstd")

    # sanity: treated stations' distances
    for crs in ["NCL", "EDB", "MPT", "SVG", "YRK"]:
        r = out.filter(pl.col("crs") == crs)
        if r.height:
            LOG.info("  %s %-12s dist-to-London = %.0f km", crs, r["station_name"][0], r["distance_to_london_km"][0])
    LOG.info("covariates -> %s (%d stations)", INTERIM / "station_covariates.parquet", out.height)


if __name__ == "__main__":
    main()
