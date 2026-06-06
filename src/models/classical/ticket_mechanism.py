"""
Ticket-type mechanism — is Lumo's growth in Advance/leisure or commuter tickets?

THINK -> RESEARCH -> CODE
  WHAT: ORR Table 1410 splits entries+exits into Full / Reduced (≈Advance) / Season
        (≈commuter). Lumo is an Advance-fares, leisure-oriented operator, so if its entry
        *created* new (leisure/discretionary) demand we expect the **Reduced-ticket share**
        to rise at its stations relative to controls — whereas pure commuter substitution
        would not. We build the share trajectory 2020-21 → 2024-25 for Lumo stops vs ECML
        controls vs clean donors.
  DATA: 5 annual 1410 snapshots (headers vary year to year ⇒ matched by substring).
  CAVEAT: the only pre-Lumo year with a split is 2020-21 (COVID-distorted) ⇒ this is a
        DESCRIPTIVE mechanism read, not a clean pre/post causal test. Honest limitation.

Run:  python -m src.models.classical.ticket_mechanism
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, METRICS, RAW, TABLES, ensure_dirs

LOG = get_logger("models.ticket_mechanism", log_file="logs/models.log")

FILES = {
    2020: "table-1410-station-usage-2020-21.ods",
    2021: "table-1410-station-usage-2021-22.ods",
    2022: "table-1410-station-usage-2022-23.ods",
    2023: "table-1410-station-usage-2023-24.ods",
    2024: "table-1410-station-usage-2024-25.ods",
}


def _num(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.replace(",", "").str.strip(), errors="coerce")


def _parse_year(path) -> pd.DataFrame | None:
    xl = pd.ExcelFile(path, engine="odf")
    # pick the data sheet BY NAME (don't read every sheet — these ODS carry junk ref-sheets)
    cands = [
        s
        for s in xl.sheet_names
        if not s.startswith("'file") and s not in ("Cover_sheet", "Notes", "Cover", "Contents")
    ]
    sheet = next(
        (s for s in cands if "1410" in s or "usage" in s.lower() or "ntrie" in s.lower()),
        cands[0] if cands else xl.sheet_names[0],
    )
    raw = pd.read_excel(path, sheet_name=sheet, engine="odf", header=None)
    # find header row (contains a Three-Letter-Code column)
    hrow = next(
        (i for i in range(min(8, len(raw))) if raw.iloc[i].astype(str).str.contains("Three Letter", case=False).any()),
        3,
    )
    hdr = raw.iloc[hrow].astype(str)

    def find(*subs, exclude=()):
        for j, h in enumerate(hdr):
            hl = h.lower()
            if all(s in hl for s in subs) and not any(e in hl for e in exclude):
                return j
        return None

    crs = find("three letter")
    reduced = find("reduced")
    season = find("season")
    allc = find("all", "ticket") or find("total")
    if crs is None or reduced is None or allc is None:
        LOG.warning("could not locate ticket columns in %s", path.name)
        return None
    body = raw.iloc[hrow + 1 :]
    df = pd.DataFrame(
        {
            "crs": body.iloc[:, crs].astype(str).str.strip(),
            "reduced": _num(body.iloc[:, reduced]),
            "season": _num(body.iloc[:, season]) if season is not None else np.nan,
            "all": _num(body.iloc[:, allc]),
        }
    )
    df = df[df["crs"].str.fullmatch(r"[A-Z]{3}")].dropna(subset=["all"])
    df = df[df["all"] > 0]
    df["reduced_share"] = df["reduced"] / df["all"]
    return df[["crs", "reduced_share"]]


def main() -> None:
    ensure_dirs()
    cfg = load_config("base")
    served = cfg["treatments"]["lumo"]["served_crs"]
    units = pd.read_parquet("data/processed/units.parquet")[["crs", "role"]]

    frames = []
    for yr, fn in FILES.items():
        p = RAW / fn
        if not p.exists():
            continue
        d = _parse_year(p)
        if d is not None:
            d["year"] = yr
            frames.append(d)
    if not frames:
        LOG.error("no ticket data parsed")
        return
    tk = pd.concat(frames, ignore_index=True).merge(units, on="crs", how="left")

    # group trajectories: Lumo stops vs ECML controls vs clean donors
    tk["grp"] = np.where(
        tk["crs"].isin(served),
        "Lumo stops",
        np.where(
            tk["role"] == "ecml_corridor_control",
            "ECML controls",
            np.where(tk["role"] == "donor_clean", "Clean donors", "other"),
        ),
    )
    traj = tk[tk.grp != "other"].groupby(["grp", "year"])["reduced_share"].mean().reset_index()

    # key stat: change in Reduced share 2021->2024 by group
    def chg(g):
        s = traj[traj.grp == g].set_index("year")["reduced_share"]
        return round(float(s.get(2024, np.nan) - s.get(2021, np.nan)) * 100, 1)

    summary = {
        "reduced_share_change_pp_2021_2024": {g: chg(g) for g in ["Lumo stops", "ECML controls", "Clean donors"]},
        "newcastle_reduced_share": {
            int(r.year): round(float(r.reduced_share), 3) for r in tk[tk.crs == "NCL"].itertuples()
        },
    }
    (METRICS / "m6_ticket_mechanism.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    traj.to_csv(TABLES / "m6_ticket_mechanism.csv", index=False)
    LOG.info("Reduced-share change 2021->2024 (pp): %s", summary["reduced_share_change_pp_2021_2024"])
    LOG.info("Newcastle Reduced share by year: %s", summary["newcastle_reduced_share"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for g, c in [("Lumo stops", "#d62728"), ("ECML controls", "#ff7f0e"), ("Clean donors", "#1f77b4")]:
        s = traj[traj.grp == g].sort_values("year")
        ax.plot(s.year, s.reduced_share * 100, "o-", color=c, lw=2 if g == "Lumo stops" else 1.4, label=g)
    ax.axvline(2021, color="grey", ls=":", lw=1.3)
    ax.text(2021.05, ax.get_ylim()[0], " Lumo", color="grey", rotation=90, fontsize=8, va="bottom")
    ax.set_xlabel("Financial year (start)")
    ax.set_ylabel("Reduced (≈Advance) ticket share of entries+exits, %")
    ax.set_title(
        "Mechanism — do Lumo stops shift toward Advance/leisure tickets?\n(rising Reduced share ⇒ new leisure demand, not commuter substitution)",
        fontsize=10,
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "m6_ticket_mechanism.png", dpi=150)
    plt.close(fig)
    LOG.info("figure -> %s | ticket mechanism complete.", FIGURES / "m6_ticket_mechanism.png")


if __name__ == "__main__":
    main()
