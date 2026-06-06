"""Smoke test: every pipeline stage module imports and exposes main() (reproducibility)."""

from __future__ import annotations

import importlib

import pytest

STAGES = [
    "src.ingest.download_orr",
    "src.clean.parse_station_usage",
    "src.clean.parse_1410_snapshot",
    "src.clean.characterise_missingness",
    "src.clean.parse_operator_usage",
    "src.features.build_panel",
    "src.features.select_units",
    "src.viz.eda_pretrends",
    "src.models.classical.synthetic_control",
    "src.models.classical.within_corridor",
    "src.models.classical.operator_analysis",
    "src.models.classical.openaccess_sector",
    "src.models.classical.did_event_study",
    "src.models.classical.causal_impact_operator",
    "src.evaluate.robustness_sc",
    "src.evaluate.triangulation",
    "src.models.causal_forest.heterogeneity",
    "src.models.deep.deep_counterfactual",
    "src.ingest.download_supplementary",
    "src.models.classical.augmented_sc",
    "src.evaluate.confounder_controls",
    "src.features.build_covariates",
    "src.models.deep.deep_counterfactual_gru",
    "src.models.deep.deep_counterfactual_crn",
    "src.models.od_substitution",
    "src.models.od_inference",
    "src.models.od_event_study",
    "src.models.od_corridor_robustness",
    "src.evaluate.multiple_testing",
    "src.evaluate.master_synthesis",
    "src.evaluate.sensitivity_confounding",
    "src.models.classical.cs_did",
    "src.models.classical.gc_hull_replication",
    "src.models.causal_forest.od_causal_forest",
    "src.models.air_modal_shift",
    "src.models.carbon_welfare",
    "src.evaluate.footfall_validation",
    "src.clean.parse_carbon",
    "src.clean.parse_footfall",
    "src.ingest.download_caa",
]


@pytest.mark.parametrize("mod", STAGES)
def test_stage_imports_and_has_main(mod):
    m = importlib.import_module(mod)
    assert hasattr(m, "main"), f"{mod} missing main()"
