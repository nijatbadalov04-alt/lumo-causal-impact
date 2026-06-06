"""Unit tests for operator/sector parsing helpers (pure functions, no file I/O)."""

from __future__ import annotations

import numpy as np

from src.clean.parse_operator_usage import _clean_series_name, _parse_value, _period_to_decimal


def test_period_annual():
    assert _period_to_decimal("Apr 2011 to Mar 2012") == (2011.0, "annual")


def test_period_annual_with_break_flag_still_parses():
    dec, freq = _period_to_decimal("Apr 2021 to Mar 2022 [b] [p]")
    assert dec == 2021.0 and freq == "annual"


def test_period_quarters_map_to_financial_year():
    # Apr-Jun starts FY; Jan-Mar belongs to the FY that began the previous calendar year
    assert _period_to_decimal("Apr to Jun 2011") == (2011.375, "quarterly")
    assert _period_to_decimal("Jul to Sep 2011") == (2011.625, "quarterly")
    assert _period_to_decimal("Oct to Dec 2011") == (2011.875, "quarterly")
    assert _period_to_decimal("Jan to Mar 2012") == (2011.125, "quarterly")  # FY2011-12 Q4


def test_period_non_period_returns_none():
    assert _period_to_decimal("Table 1223b: Passenger journeys") == (None, None)


def test_clean_series_name_strips_units_and_notes():
    assert _clean_series_name("London North Eastern Railway(million)") == "London North Eastern Railway"
    assert _clean_series_name("Lumo(million)[note 5]") == "Lumo"


def test_parse_value_numeric_and_sentinels():
    v, f = _parse_value("23.43")
    assert abs(v - 23.43) < 1e-9 and f == "observed"
    v, f = _parse_value("[z]")
    assert np.isnan(v) and f == "not_applicable"
    v, f = _parse_value("[x]")
    assert np.isnan(v) and f == "not_available"
