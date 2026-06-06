"""
Estimator-correctness tests (CRITIQUE F): inject a KNOWN effect into synthetic data and
check each estimator recovers it. These validate the maths against ground truth, unlike the
import smoke tests.
"""

from __future__ import annotations

import numpy as np

from src.models.carbon_welfare import _scenarios
from src.models.od_event_study import _demean, _slope


def _make_panel(nc=20, ny=6, treated=(0, 1), post=(4, 5), tau=0.30, noise=0.0, seed=0):
    """Balanced flow x year panel with flow FE + year FE + a known treated-post effect tau."""
    rng = np.random.default_rng(seed)
    flow_fe = rng.normal(10, 1.0, nc)
    year_fe = rng.normal(0, 0.5, ny)
    ci, yi, y = [], [], []
    tset, pset = set(treated), set(post)
    for i in range(nc):
        for t in range(ny):
            val = flow_fe[i] + year_fe[t] + (tau if (i in tset and t in pset) else 0.0)
            if noise:
                val += rng.normal(0, noise)
            ci.append(i)
            yi.append(t)
            y.append(val)
    return np.array(ci), np.array(yi), np.array(y, dtype=float), tset, pset


def test_twfe_recovers_known_effect_exactly_without_noise():
    """With no noise the two-way FE DiD must recover tau to machine precision."""
    nc, ny, tau = 25, 6, 0.42
    ci, yi, y, tset, pset = _make_panel(nc=nc, ny=ny, tau=tau, noise=0.0)
    treated_post = np.array([1.0 if (c in tset and t in pset) else 0.0 for c, t in zip(ci, yi)])
    beta = _slope(_demean(treated_post, ci, yi, nc, ny), _demean(y, ci, yi, nc, ny))
    assert abs(beta - tau) < 1e-9, f"recovered {beta}, expected {tau}"


def test_twfe_recovers_known_effect_with_noise():
    """With modest noise the estimate stays close to the injected tau."""
    nc, ny, tau = 60, 6, 0.30
    ci, yi, y, tset, pset = _make_panel(nc=nc, ny=ny, treated=(0, 1, 2), post=(4, 5), tau=tau, noise=0.05, seed=7)
    treated_post = np.array([1.0 if (c in tset and t in pset) else 0.0 for c, t in zip(ci, yi)])
    beta = _slope(_demean(treated_post, ci, yi, nc, ny), _demean(y, ci, yi, nc, ny))
    assert abs(beta - tau) < 0.03, f"recovered {beta}, expected ~{tau}"


def test_twfe_zero_effect_is_near_zero():
    """No treatment -> estimate ~0. Use many treated cells so the DiD is well-identified."""
    nc, ny = 60, 6
    ci, yi, y, tset, pset = _make_panel(
        nc=nc, ny=ny, treated=tuple(range(8)), post=(3, 4, 5), tau=0.0, noise=0.02, seed=3
    )
    treated_post = np.array([1.0 if (c in tset and t in pset) else 0.0 for c, t in zip(ci, yi)])
    beta = _slope(_demean(treated_post, ci, yi, nc, ny), _demean(y, ci, yi, nc, ny))
    assert abs(beta) < 0.02


def test_dml_ate_recovers_known_effect_under_confounding():
    """The causal-forest DML/partialling-out ATE must recover a known tau despite confounding."""
    from src.models.causal_forest.od_causal_forest import _dml_ate

    rng = np.random.default_rng(0)
    n = 1500
    X = rng.normal(0, 1, (n, 2))
    prop = 1 / (1 + np.exp(-(0.8 * X[:, 0] - 0.5 * X[:, 1])))  # treatment depends on X (confounding)
    w = (rng.uniform(size=n) < prop).astype(float)
    tau = 0.40
    y = 1.0 + 1.5 * X[:, 0] - 0.7 * X[:, 1] + tau * w + rng.normal(0, 0.3, n)
    ate, _, _ = _dml_ate(X, w, y, seed=0)
    # naive difference-in-means would be badly biased by the confounding; DML should be close
    assert abs(ate - tau) < 0.08, f"DML recovered {ate}, expected {tau}"


def test_carbon_scenarios_have_correct_signs():
    """Air/car abstraction must SAVE (>0), pure induced must COST (<0), break-even in (0,1)."""
    e = {"saving_rail_over_air_kg": 151.2, "saving_rail_over_car_kg": 118.6, "rail_standard_kg": 13.8}
    sc = _scenarios(delta=1_000_000, e=e)
    assert sc["saving_if_all_from_air_tonnes"] > sc["saving_if_all_from_car_tonnes"] > 0
    assert sc["cost_if_all_induced_tonnes"] < 0  # induced adds emissions
    assert 0.0 < sc["breakeven_induced_share"] < 1.0
    # because rail << air, break-even induced share must be high (abstraction dominates)
    assert sc["breakeven_induced_share"] > 0.8


def test_carbon_breakeven_zero_point():
    """At the break-even induced share, the net (air/car mix) saving must vanish by definition."""
    e = {"saving_rail_over_air_kg": 151.2, "saving_rail_over_car_kg": 118.6, "rail_standard_kg": 13.8}
    sc = _scenarios(delta=1_000_000, e=e)
    s = sc["breakeven_induced_share"]
    a = 0.5  # matches BASELINE_AIR_FRACTION_AMONG_ABSTRACTED in carbon_welfare
    mixed = a * e["saving_rail_over_air_kg"] + (1 - a) * e["saving_rail_over_car_kg"]
    net_per_journey = (1 - s) * mixed - s * e["rail_standard_kg"]
    assert abs(net_per_journey) < 1e-6
