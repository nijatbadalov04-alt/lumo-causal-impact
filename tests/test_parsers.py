"""Unit tests for the M1 parsing logic (pure functions only — no file I/O).

Addresses Dr. Vasquez's M1 critique: transform code must be tested, not lucky.
Run:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.clean.parse_station_usage import _parse_year_header, parse_values


# ---- financial-year header parsing ----
def test_year_header_plain():
    assert _parse_year_header("Apr 1997 to Mar 1998") == (1997, "1997-98", False)


def test_year_header_break_flag():
    # The 2022-23 column carries a [b] break annotation in Table 1415.
    assert _parse_year_header("Apr 2022 to Mar 2023 [b]") == (2022, "2022-23", True)


def test_year_header_last_year():
    assert _parse_year_header("Apr 2024 to Mar 2025") == (2024, "2024-25", False)


def test_year_header_non_year_returns_none():
    for h in ["Station name", "Three Letter Code(TLC)", "Region", ""]:
        assert _parse_year_header(h) is None


# ---- value / flag mapping (the missing-value taxonomy) ----
def test_values_map_numbers_sentinels_and_true_zero():
    s = pd.Series(["123", "2284585.024", "[x]", "[z]", "0", "[c]", "weird"])
    val, flag = parse_values(s)

    assert val.iloc[0] == 123.0 and flag[0] == "observed"
    assert abs(val.iloc[1] - 2284585.024) < 1e-6 and flag[1] == "observed"  # CAPRI decimal
    assert np.isnan(val.iloc[2]) and flag[2] == "not_available"  # [x]
    assert np.isnan(val.iloc[3]) and flag[3] == "not_applicable"  # [z]
    assert val.iloc[4] == 0.0 and flag[4] == "observed"  # TRUE ZERO ≠ missing
    assert np.isnan(val.iloc[5]) and flag[5] == "confidential"  # [c]
    assert np.isnan(val.iloc[6]) and flag[6] == "unknown_missing"


def test_values_dtype_is_float():
    val, _ = parse_values(pd.Series(["1", "[x]"]))
    assert str(val.dtype) == "float64"
