"""
End-to-end pipeline runner — the single-command reproduction entry point (§10).

    python run_pipeline.py            # run all available stages, in order
    python run_pipeline.py --from M1.parse_1415
    python run_pipeline.py --only M1.missingness

Every stage is an idempotent module exposing `main()`. Stages are added as each
milestone lands; today the M1 (data) stages are wired up.
"""
from __future__ import annotations

import argparse
import importlib
import time

from src.utils.logging_setup import get_logger

LOG = get_logger("run_pipeline", log_file="logs/pipeline.log")

# (stage_id, module) — order matters.
STAGES: list[tuple[str, str]] = [
    ("M1.download",     "src.ingest.download_orr"),
    ("M1.download_supp", "src.ingest.download_supplementary"),  # operator + confounder data
    ("M1.download_caa", "src.ingest.download_caa"),  # CAA domestic air route data (OGL)
    ("M1.parse_1415",   "src.clean.parse_station_usage"),
    ("M1.parse_1410",   "src.clean.parse_1410_snapshot"),
    ("M1.missingness",  "src.clean.characterise_missingness"),
    ("M1.parse_carbon", "src.clean.parse_carbon"),      # GTD per-pair emissions (needs data/raw/carbon/)
    ("M1.parse_footfall", "src.clean.parse_footfall"),  # NR daily concourse footfall (needs data/raw/footfall/)
    ("M2.build_panel",  "src.features.build_panel"),
    ("M2.select_units", "src.features.select_units"),
    ("M2.pretrends",    "src.viz.eda_pretrends"),
    ("M3.synth_control",     "src.models.classical.synthetic_control"),
    ("M3.within_corridor",   "src.models.classical.within_corridor"),
    ("M3.augmented_sc",      "src.models.classical.augmented_sc"),
    ("M3.generalised_sc",    "src.models.classical.generalised_sc"),  # interactive fixed effects
    ("M3.robustness_sc",     "src.evaluate.robustness_sc"),
    ("M3.parse_operator",    "src.clean.parse_operator_usage"),
    ("M3.operator_analysis", "src.models.classical.operator_analysis"),
    ("M6.confounder_controls", "src.evaluate.confounder_controls"),  # punctuality + fares
    ("RQ1.od_substitution",  "src.models.od_substitution"),  # DECISIVE OD-pair test (needs ODM in data/raw/odm/)
    ("RQ1.od_inference",     "src.models.od_inference"),      # randomization inference on the OD result
    ("RQ1.od_event_study",   "src.models.od_event_study"),    # OD-flow panel TWFE DiD + permutation
    ("RQ1.air_modal_shift",  "src.models.air_modal_shift"),   # CAA air->rail decomposition (induced vs abstraction)
    ("M7.carbon_welfare",    "src.models.carbon_welfare"),    # CO2 of the corridor growth (needs air_modal_shift)
    ("M6.footfall_validation", "src.evaluate.footfall_validation"),  # real NR counts vs modelled usage
    ("M3.did_event_study",   "src.models.classical.did_event_study"),
    ("M3.cs_did",            "src.models.classical.cs_did"),  # Callaway-Sant'Anna staggered adoption
    ("M3.causal_impact",     "src.models.classical.causal_impact_operator"),  # BSTS, quarterly
    ("M6.openaccess_sector", "src.models.classical.openaccess_sector"),  # RQ4 generalisation
    ("RQ4.gc_hull_replication", "src.models.classical.gc_hull_replication"),  # GC pre-COVID external validity
    ("M4.deep_counterfactual", "src.models.deep.deep_counterfactual"),  # GPU (attention ensemble)
    ("M4.deep_crn",          "src.models.deep.deep_counterfactual_crn"),  # GPU (CRN-lite, domain-adversarial)
    ("M4.triangulation",     "src.evaluate.triangulation"),
    ("M5.covariates",        "src.features.build_covariates"),  # distance-to-London (NaPTAN)
    ("RQ1.od_corridor_robust", "src.models.od_corridor_robustness"),  # distance-matched RI (needs covariates)
    ("RQ2.od_causal_forest", "src.models.causal_forest.od_causal_forest"),  # real CATE on 391 flows
    ("M7.sensitivity_confounding", "src.evaluate.sensitivity_confounding"),  # Cinelli-Hazlett/Oster/E-value
    ("RQ3.ticket_mechanism", "src.models.classical.ticket_mechanism"),  # ticket-type mechanism read
    ("M5.heterogeneity",     "src.models.causal_forest.heterogeneity"),  # RQ2
    ("M7.multiple_testing",  "src.evaluate.multiple_testing"),  # multiplicity correction over headline p-values
    ("M7.master_synthesis",  "src.evaluate.master_synthesis"),  # every estimate -> one table + forest plot (runs last)
    # Pending: modern SC (AugSC/GSC, CausalImpact); GC/Hull station replication; full paper.
]


def run(stages: list[tuple[str, str]]) -> None:
    LOG.info("pipeline: %d stage(s) -> %s", len(stages), [s for s, _ in stages])
    for name, mod in stages:
        LOG.info("===== STAGE %s (%s) =====", name, mod)
        t0 = time.time()
        importlib.import_module(mod).main()
        LOG.info("----- stage %s done in %.1fs -----", name, time.time() - t0)
    LOG.info("pipeline complete.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="from_stage", help="start at this stage id")
    ap.add_argument("--only", dest="only_stage", help="run just this stage id")
    args = ap.parse_args()

    stages = STAGES
    ids = [s for s, _ in STAGES]
    if args.only_stage:
        stages = [(s, m) for s, m in STAGES if s == args.only_stage]
    elif args.from_stage:
        if args.from_stage not in ids:
            raise SystemExit(f"unknown stage {args.from_stage!r}; choices: {ids}")
        stages = STAGES[ids.index(args.from_stage):]
    if not stages:
        raise SystemExit(f"no matching stages; choices: {ids}")
    run(stages)


if __name__ == "__main__":
    main()
