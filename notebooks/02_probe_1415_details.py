"""Precise probe of Table 1415a before writing the parser: full header names,
sentinel/shorthand note, footnote/tail rows, real station count, all non-numeric
tokens appearing in the value columns."""
from __future__ import annotations

import re

import pandas as pd

from src.utils.paths import RAW

f = RAW / "table-1415-time-series-station-usage.ods"
ee = pd.read_excel(f, sheet_name="1415a_Entries_and_Exits", engine="odf", header=None)

print("shape:", ee.shape, flush=True)

print("\n--- row 2 (shorthand note) ---", flush=True)
print(repr(ee.iloc[2, 0]), flush=True)

print("\n--- row 3 = HEADER, all columns ---", flush=True)
for i, v in enumerate(ee.iloc[3].tolist()):
    print(f"  col{i:2d}: {v!r}", flush=True)

print("\n--- TAIL 6 rows, first 6 cols (check footnotes) ---", flush=True)
print(ee.iloc[-6:, :6].to_string(max_colwidth=40), flush=True)

# Real data rows: 'Sort' (col0) is an integer-like value
sort_col = pd.to_numeric(ee.iloc[4:, 0], errors="coerce")
n_real = sort_col.notna().sum()
print(f"\n--- rows with numeric Sort (real stations): {n_real} ---", flush=True)

# All non-numeric tokens across the 28 year columns (rows 4+)
year_block = ee.iloc[4:, 6:34]
tokens = set()
for col in year_block.columns:
    s = year_block[col].astype(str)
    nonnum = s[~s.str.match(r"^-?\d+(\.\d+)?$", na=False)]
    tokens.update(nonnum.unique())
# strip obvious floats/nan
tokens = {t for t in tokens if not re.match(r"^-?\d+(\.\d+)?$", t)}
print("\n--- unique NON-NUMERIC tokens in year columns ---", flush=True)
for t in sorted(tokens):
    print(f"  {t!r}", flush=True)

# How many cells per sentinel
flat = year_block.astype(str).values.ravel()
import collections
c = collections.Counter(t for t in flat if not re.match(r"^-?\d+(\.\d+)?$", t))
print("\n--- sentinel cell counts ---", flush=True)
for k, v in c.most_common(15):
    print(f"  {k!r}: {v}", flush=True)
