from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load_json(name: str) -> dict:
    with (TEMP_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    g4 = load_json("g4_brain_direct_falsification_master_20260311.json")
    f2 = load_json("f2_spatial_brain_experiment_design_20260311.json")
    p7c = load_json("p7c_brain_spatial_falsification_minimal_core_20260311.json")
    p9c = load_json("p9c_hard_spatial_brain_forecasts_20260311.json")
    p10c = load_json("p10c_final_brain_falsifier_checklist_20260311.json")
    s8d = load_json("stage8d_brain_high_risk_falsification_20260311.json")

    protocol_specificity_score = mean(
        [
            g4["headline_metrics"]["directional_specificity_score"],
            p9c["headline_metrics"]["forecast_specificity_score"],
            f2["headline_metrics"]["local_vs_bridge_separability_score"],
            s8d["headline_metrics"]["brain_specificity_score"],
        ]
    )

    observable_coverage_score = mean(
        [
            f2["headline_metrics"]["measurable_mapping_score"],
            f2["headline_metrics"]["intervention_targetability_score"],
            g4["headline_metrics"]["experiment_readiness_score"],
            p9c["headline_metrics"]["testability_score"],
        ]
    )

    failure_criterion_sharpness_score = mean(
        [
            g4["headline_metrics"]["falsifier_sharpness_score"],
            p10c["headline_metrics"]["checklist_sharpness_score"],
            s8d["headline_metrics"]["hard_falsifier_spec_score"],
            p7c["headline_metrics"]["falsifier_sharpness_score"],
        ]
    )

    equation_binding_score = mean(
        [
            g4["headline_metrics"]["theory_term_mapping_score"],
            p7c["headline_metrics"]["minimal_core_alignment_score"],
            p7c["headline_metrics"]["geometry_constraint_score"],
            f2["headline_metrics"]["measurable_mapping_score"],
        ]
    )

    overall_g5_score = mean(
        [
            protocol_specificity_score,
            observable_coverage_score,
            failure_criterion_sharpness_score,
            equation_binding_score,
        ]
    )

    observables = {
        "local_neighborhood_perturbation": {
            "primary_readouts": [
                "family_topology_margin",
                "local_feature_separability",
                "shared_family_residual_shift",
            ],
            "equation_terms": [
                "L_t(i)",
                "C_local",
                "f_{t+1}(i)",
            ],
            "failure_rule": "If family_topology_margin does not drop before bridge-level measures, local reuse weakens.",
        },
        "long_range_bridge_cut": {
            "primary_readouts": [
                "compact_boundary_relation_bridge_score",
                "endpoint_support",
                "cross_region_integration_success",
            ],
            "equation_terms": [
                "d_t(i,j)",
                "D_3d(i,j)",
                "A_{t+1}(i,j)",
            ],
            "failure_rule": "If compact-boundary relation bridge score is not hit first, sparse bridge theory weakens.",
        },
        "geometry_only_vs_targeted_bridge_enhancement": {
            "primary_readouts": [
                "E_3d",
                "relation_bridge_specificity",
                "generalization_and_recovery",
            ],
            "equation_terms": [
                "D_3d(i,j)",
                "E_3d",
                "m_{t+1}(i,j)",
            ],
            "failure_rule": "If geometry-only smoothing beats targeted bridge enhancement, dynamic effective topology weakens.",
        },
        "fast_mid_slow_timescale_intervention": {
            "primary_readouts": [
                "short_term_feature_drift",
                "effective_topology_reorganization_speed",
                "recovery_and_retention",
            ],
            "equation_terms": [
                "f_{t+1}",
                "A_{t+1}",
                "m_{t+1}",
            ],
            "failure_rule": "If fast, mid, and slow interventions do not separate feature, topology, and recovery effects, timescale law weakens.",
        },
    }

    formulas = {
        "observable_binding": "Obs_k = Readout(Delta f_t, Delta A_t, Delta m_t, SpatialState, BridgeState)",
        "failure_margin": "FailMargin = Score(predicted_effect_order) - Score(observed_null_or_reversed_order)",
        "protocol_quality": "ProtocolQuality = mean(ProtocolSpecificity, ObservableCoverage, FailureSharpness, EquationBinding)",
        "experiment_pass": "Pass(experiment) = 1[primary_readout shifts first on the theory-predicted axis]",
    }

    verdict = {
        "status": (
            "protocol_ready_for_brain_side_execution"
            if overall_g5_score >= 0.78
            else "protocol_partially_ready"
        ),
        "best_first_protocol": "local_neighborhood_perturbation",
        "highest_information_protocol": "long_range_bridge_cut",
        "hardest_measurement": "bridge_specificity_strength",
        "can_bind_observables_to_core_equations": overall_g5_score >= 0.75,
    }

    hypotheses = {
        "H1_protocol_specificity_is_nontrivial": protocol_specificity_score >= 0.67,
        "H2_observable_coverage_is_ready": observable_coverage_score >= 0.78,
        "H3_failure_criteria_are_sharp": failure_criterion_sharpness_score >= 0.9,
        "H4_equation_binding_is_ready": equation_binding_score >= 0.77,
        "H5_g5_protocol_master_is_positive": overall_g5_score >= 0.75,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G5_brain_experiment_protocol_observable_mapping",
        },
        "headline_metrics": {
            "protocol_specificity_score": protocol_specificity_score,
            "observable_coverage_score": observable_coverage_score,
            "failure_criterion_sharpness_score": failure_criterion_sharpness_score,
            "equation_binding_score": equation_binding_score,
            "overall_g5_score": overall_g5_score,
        },
        "observables": observables,
        "formulas": formulas,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    with (TEMP_DIR / "g5_brain_experiment_protocol_observable_mapping_20260311.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
