"""Targeted inspection of the 4 REAL sheets in Table 1415 (skip 244 junk refs)."""
from __future__ import annotations

import pandas as pd

from src.utils.paths import INTERIM, RAW

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 240)

f = RAW / "table-1415-time-series-station-usage.ods"
out = []


def emit(s=""):
    print(s, flush=True)
    out.append(str(s))


# --- Notes sheet: documents columns / definitions ---
notes = pd.read_excel(f, sheet_name="Notes", engine="odf", header=None)
emit("===== NOTES SHEET (shape %s) =====" % (notes.shape,))
emit(notes.iloc[:, :2].to_string(max_colwidth=120))

# --- Main entries & exits time series ---
ee = pd.read_excel(f, sheet_name="1415a_Entries_and_Exits", engine="odf", header=None)
emit("\n\n===== 1415a_Entries_and_Exits (shape %s) =====" % (ee.shape,))
emit("FIRST 14 ROWS x FIRST 14 COLS:")
emit(ee.iloc[:14, :14].to_string(max_colwidth=22))
emit("\nFIRST 8 ROWS x LAST 12 COLS:")
emit(ee.iloc[:8, -12:].to_string(max_colwidth=22))

# --- Interchanges ---
ix = pd.read_excel(f, sheet_name="1415b_Interchanges", engine="odf", header=None)
emit("\n\n===== 1415b_Interchanges (shape %s) =====" % (ix.shape,))
emit(ix.iloc[:8, :12].to_string(max_colwidth=22))

(INTERIM / "_inspect_1415_real.txt").write_text("\n".join(out), encoding="utf-8")
emit("\n[done]")
