from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load_json(name: str) -> dict:
    path = TEMP_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    f2 = load_json("f2_spatial_brain_experiment_design_20260311.json")
    p7c = load_json("p7c_brain_spatial_falsification_minimal_core_20260311.json")
    s7c = load_json("stage7c_brain_falsifiable_predictions_20260311.json")
    s8d = load_json("stage8d_brain_high_risk_falsification_20260311.json")
    p9c = load_json("p9c_hard_spatial_brain_forecasts_20260311.json")
    p10c = load_json("p10c_final_brain_falsifier_checklist_20260311.json")

    experiment_readiness_score = mean(
        [
            f2["headline_metrics"]["intervention_targetability_score"],
            f2["headline_metrics"]["measurable_mapping_score"],
            f2["headline_metrics"]["local_vs_bridge_separability_score"],
            f2["headline_metrics"]["geometry_rejection_readiness_score"],
        ]
    )

    falsifier_sharpness_score = mean(
        [
            p9c["headline_metrics"]["forecast_sharpness_score"],
            p10c["headline_metrics"]["checklist_sharpness_score"],
            s8d["headline_metrics"]["hard_falsifier_spec_score"],
            s7c["headline_metrics"]["falsifiability_quality_score"],
        ]
    )

    directional_specificity_score = mean(
        [
            s8d["headline_metrics"]["directional_falsifier_score"],
            s8d["headline_metrics"]["brain_specificity_score"],
            s7c["headline_metrics"]["bridge_alignment_score"],
            p9c["headline_metrics"]["forecast_specificity_score"],
        ]
    )

    theory_term_mapping_score = mean(
        [
            p7c["headline_metrics"]["geometry_constraint_score"],
            p7c["headline_metrics"]["minimal_core_alignment_score"],
            p7c["headline_metrics"]["brain_spatial_plausibility_score"],
            p9c["headline_metrics"]["testability_score"],
        ]
    )

    overall_g4_score = mean(
        [
            experiment_readiness_score,
            falsifier_sharpness_score,
            directional_specificity_score,
            theory_term_mapping_score,
        ]
    )

    formulas = {
        "intervention_to_equation": (
            "Measure(Delta f_t, Delta A_t, Delta m_t) under local perturbation, bridge cut, "
            "targeted bridge enhancement, and timescale-specific intervention."
        ),
        "brain_falsifier_margin": (
            "FalsifierMargin = Score(targeted_prediction) - Score(generic_support_or_null)"
        ),
        "mapping_quality": (
            "MapQuality = mean(LocalVsBridgeSeparation, EquationTermReadout, DirectInterventionTargetability)"
        ),
        "direct_testability": (
            "DirectTestability = mean(ExperimentReadiness, FalsifierSharpness, DirectionalSpecificity, TheoryTermMapping)"
        ),
    }

    candidate_experiments = [
        "local_neighborhood_perturbation",
        "long_range_bridge_cut",
        "geometry_only_vs_targeted_bridge_enhancement",
        "fast_mid_slow_timescale_intervention",
    ]

    strongest_falsifiers = [
        "If local perturbation does not first reduce family topology margin, local reuse weakens.",
        "If long-range bridge cut does not first reduce compact-boundary relation bridge measures, sparse bridge theory weakens.",
        "If geometry-only smoothing beats targeted bridge enhancement, dynamic effective topology weakens.",
        "If high-value long-range bridges are broad and diffuse rather than sparse and bundled, 3D efficiency weakens.",
    ]

    verdict = {
        "status": (
            "direct_brain_falsification_ready"
            if overall_g4_score >= 0.74
            else "direct_brain_falsification_partially_ready"
        ),
        "best_start": "local_neighborhood_perturbation",
        "highest_value": "long_range_bridge_cut",
        "main_open_gap": "bridge_specificity_strength",
        "can_now_test_core_claim_directly": overall_g4_score >= 0.72,
    }

    hypotheses = {
        "H1_experiment_suite_is_ready": experiment_readiness_score >= 0.7,
        "H2_falsifiers_are_sharp": falsifier_sharpness_score >= 0.85,
        "H3_directional_specificity_is_nontrivial": directional_specificity_score >= 0.68,
        "H4_theory_terms_map_to_measurements": theory_term_mapping_score >= 0.74,
        "H5_g4_master_is_positive": overall_g4_score >= 0.72,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G4_brain_direct_falsification_master",
        },
        "headline_metrics": {
            "experiment_readiness_score": experiment_readiness_score,
            "falsifier_sharpness_score": falsifier_sharpness_score,
            "directional_specificity_score": directional_specificity_score,
            "theory_term_mapping_score": theory_term_mapping_score,
            "overall_g4_score": overall_g4_score,
        },
        "candidate_experiments": candidate_experiments,
        "strongest_falsifiers": strongest_falsifiers,
        "formulas": formulas,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "g4_brain_direct_falsification_master_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
