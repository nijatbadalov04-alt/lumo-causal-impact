"""Exploratory: inspect raw ORR Table 1415 ODS structure (run once, unbuffered).
Writes a concise summary to data/interim/_inspect_1415.txt and prints it.
Exploration only (parsing logic lives in src/).
"""
from __future__ import annotations

import sys

import pandas as pd

from src.utils.paths import INTERIM, RAW

OUT = INTERIM / "_inspect_1415.txt"
lines: list[str] = []


def emit(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


f1415 = RAW / "table-1415-time-series-station-usage.ods"
emit(f"FILE: {f1415.name}  ({f1415.stat().st_size:,} bytes)")

xl = pd.ExcelFile(f1415, engine="odf")
emit(f"SHEETS ({len(xl.sheet_names)}): {xl.sheet_names}")

# Read every sheet fully once; report shape so we can pick the data sheet.
sheet_frames = {}
for sheet in xl.sheet_names:
    df = pd.read_excel(f1415, sheet_name=sheet, engine="odf", header=None)
    sheet_frames[sheet] = df
    emit(f"  - {sheet!r}: shape={df.shape}")

# Data sheet = the one with the most rows.
data_sheet = max(sheet_frames, key=lambda s: sheet_frames[s].shape[0])
emit(f"\nDATA SHEET = {data_sheet!r}")
df = sheet_frames[data_sheet]

emit("\nFIRST 12 ROWS x FIRST 10 COLS (header=None):")
emit(df.iloc[:12, :10].to_string(max_colwidth=26))

emit("\nFIRST 12 ROWS x LAST 8 COLS:")
emit(df.iloc[:12, -8:].to_string(max_colwidth=26))

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"\n[written] {OUT}", file=sys.stderr, flush=True)
