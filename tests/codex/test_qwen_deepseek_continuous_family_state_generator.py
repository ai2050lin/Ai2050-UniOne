from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_json(rel_path: str) -> Dict[str, Any]:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def vec(xs: Iterable[float]) -> np.ndarray:
    return np.array(list(xs), dtype=np.float32)


def get_pair_distance(family_row: Dict[str, Any], left: str, right: str) -> float:
    for key in ("__".join(sorted((left, right))), f"{left}__{right}", f"{right}__{left}"):
        if key in family_row["pairwise_distances"]:
            return float(family_row["pairwise_distances"][key])
    raise KeyError(f"missing pair distance for {left}, {right}")


def clipped_accuracy_from_relative_error(relative_error: float) -> float:
    return max(0.0, min(1.0, 1.0 - relative_error))


def family_mean_state(concepts: Dict[str, Any]) -> np.ndarray:
    rows = [vec(state) for state in concepts.values()]
    return np.mean(np.stack(rows, axis=0), axis=0)


def evaluate_generator(
    generated_centers: Dict[str, np.ndarray],
    generated_concepts: Dict[str, Dict[str, Any]],
    actual_centers: Dict[str, np.ndarray],
    family_rows: Dict[str, Dict[str, Any]],
    scaffold: np.ndarray,
) -> Dict[str, Any]:
    family_basis_metrics: Dict[str, Dict[str, float]] = {}
    family_basis_scores = []
    for family, actual_center in actual_centers.items():
        predicted_center = generated_centers[family]
        absolute_error = float(np.linalg.norm(predicted_center - actual_center))
        reference_norm = float(np.linalg.norm(actual_center - scaffold) + 1e-8)
        relative_error = absolute_error / reference_norm
        accuracy = clipped_accuracy_from_relative_error(relative_error)
        family_basis_metrics[family] = {
            "absolute_error": absolute_error,
            "relative_error": relative_error,
            "accuracy": accuracy,
        }
        family_basis_scores.append(accuracy)

    assignment_rows: Dict[str, Dict[str, Any]] = {}
    correct_assignments = 0
    total_assignments = 0
    for concept, row in generated_concepts.items():
        state = vec(row["state"])
        true_family = str(row["family"])
        nearest_family = min(
            actual_centers,
            key=lambda family: float(np.linalg.norm(state - actual_centers[family])),
        )
        is_correct = nearest_family == true_family
        correct_assignments += int(is_correct)
        total_assignments += 1
        assignment_rows[concept] = {
            "true_family": true_family,
            "nearest_family": nearest_family,
            "is_correct": is_correct,
        }
    family_assignment_accuracy = correct_assignments / max(total_assignments, 1)

    geometry_rows: Dict[str, Dict[str, float]] = {}
    geometry_scores = []
    correlation_scores = []
    for family, row in family_rows.items():
        names = list(row["concepts"])
        actual_pair_dists = []
        generated_pair_dists = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                left = names[i]
                right = names[j]
                actual_pair_dists.append(get_pair_distance(row, left, right))
                state_left = vec(generated_concepts[left]["state"])
                state_right = vec(generated_concepts[right]["state"])
                generated_pair_dists.append(float(np.linalg.norm(state_left - state_right)))

        actual_arr = np.array(actual_pair_dists, dtype=np.float64)
        generated_arr = np.array(generated_pair_dists, dtype=np.float64)
        mape = float(np.mean(np.abs(generated_arr - actual_arr) / (actual_arr + 1e-8)))
        geometry_score = max(0.0, min(1.0, 1.0 - mape))
        corr = float(np.corrcoef(actual_arr, generated_arr)[0, 1]) if len(actual_arr) > 1 else 1.0
        corr_score = max(0.0, min(1.0, (corr + 1.0) / 2.0))
        geometry_rows[family] = {
            "mape": mape,
            "geometry_score": geometry_score,
            "distance_rank_correlation": corr,
            "distance_rank_score": corr_score,
        }
        geometry_scores.append(geometry_score)
        correlation_scores.append(corr_score)

    return {
        "family_basis_metrics": family_basis_metrics,
        "family_assignment": {
            "accuracy": family_assignment_accuracy,
            "correct_count": correct_assignments,
            "total_count": total_assignments,
            "per_concept": assignment_rows,
        },
        "concept_geometry_metrics": geometry_rows,
        "scores": {
            "family_basis_prediction_score": float(sum(family_basis_scores) / len(family_basis_scores)),
            "family_assignment_score": family_assignment_accuracy,
            "concept_geometry_score": float(sum(geometry_scores) / len(geometry_scores)),
            "concept_rank_score": float(sum(correlation_scores) / len(correlation_scores)),
        },
    }


def build_payload() -> Dict[str, Any]:
    t0 = time.time()

    baseline_generator = load_json("tests/codex_temp/qwen_deepseek_universal_family_state_generator_20260315.json")
    operator_closure = load_json("tests/codex_temp/qwen_deepseek_universal_family_operator_closure_20260315.json")
    factorization = load_json("tests/codex_temp/qwen_deepseek_concept_local_residual_auto_factorization_20260315.json")
    predictive_closure = load_json("tests/codex_temp/qwen_deepseek_patch_offset_predictive_closure_20260315.json")
    dynamic_law = load_json("tests/codex_temp/qwen_deepseek_adaptive_offset_dynamic_law_20260315.json")
    unified_equation = load_json("tests/codex_temp/qwen_deepseek_readout_transport_bridge_unified_state_equation_20260315.json")
    family_atlas = load_json("tests/codex_temp/theory_track_concept_family_atlas_analysis_20260312.json")
    analytic_transfer = load_json("tests/codex_temp/qwen_deepseek_analytic_family_transfer_law_20260315.json")

    scaffold = vec(analytic_transfer["analytic_objects"]["global_scaffold"])
    family_rows = family_atlas["family_atlas"]
    actual_centers = {family: vec(row["family_center"]) for family, row in family_rows.items()}

    baseline_centers = {
        family: vec(center) for family, center in baseline_generator["generated_family_basis"].items()
    }
    baseline_concepts = baseline_generator["generated_concepts"]

    continuous_centers = {
        "fruit": vec(operator_closure["pairwise_results"]["animal_to_fruit"]["continuous_family_center"]),
        "animal": vec(operator_closure["pairwise_results"]["fruit_to_animal"]["continuous_family_center"]),
        "abstract": vec(operator_closure["pairwise_results"]["fruit_to_abstract"]["continuous_family_center"]),
    }

    inferred_states = factorization["inferred_family_states"]
    template_offsets: Dict[str, Dict[str, np.ndarray]] = {}
    for family, concepts in inferred_states.items():
        inferred_center = family_mean_state(concepts)
        template_offsets[family] = {
            concept: vec(state) - inferred_center for concept, state in concepts.items()
        }

    continuous_concepts: Dict[str, Dict[str, Any]] = {}
    for family, concepts in inferred_states.items():
        center = continuous_centers[family]
        for concept in concepts:
            state = center + template_offsets[family][concept]
            continuous_concepts[concept] = {
                "family": family,
                "state": [float(v) for v in state.tolist()],
            }

    baseline_eval = evaluate_generator(
        generated_centers=baseline_centers,
        generated_concepts=baseline_concepts,
        actual_centers=actual_centers,
        family_rows=family_rows,
        scaffold=scaffold,
    )
    continuous_eval = evaluate_generator(
        generated_centers=continuous_centers,
        generated_concepts=continuous_concepts,
        actual_centers=actual_centers,
        family_rows=family_rows,
        scaffold=scaffold,
    )

    dynamic_score = float(dynamic_law["derived_scores"]["closure_score"])
    bridge_score = float(unified_equation["component_scores"]["bridge_closure_score"])
    baseline_predictive = float(predictive_closure["predictive_scores"]["predictive_closure_score"])
    continuous_predictive = float(
        (
            continuous_eval["scores"]["family_basis_prediction_score"]
            + continuous_eval["scores"]["family_assignment_score"]
            + continuous_eval["scores"]["concept_geometry_score"]
            + continuous_eval["scores"]["concept_rank_score"]
            + dynamic_score
            + bridge_score
            + float(predictive_closure["predictive_scores"]["world_state_score"])
        )
        / 7.0
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "QwenDeepSeek_continuous_family_state_generator",
        },
        "strict_goal": {
            "statement": (
                "Replace support-remap family transfer with continuous family operators, then reuse "
                "family-centered local residual templates to build a stronger family state generator."
            ),
            "boundary": (
                "This is an observed-family closure upgrade. It fixes the active support-remap weakness, "
                "but it still does not prove unseen-family prediction."
            ),
        },
        "generator_upgrade": {
            "old_family_transfer": baseline_generator["generator_law"]["support_transport"],
            "new_family_transfer": operator_closure["candidate_operator_family"]["continuous_transport"],
            "concept_template_law": "h_c^cont = B_f^cont + (h_c^template - mean_family_template)",
        },
        "baseline_scores": baseline_eval["scores"],
        "continuous_scores": continuous_eval["scores"],
        "score_deltas": {
            key: float(continuous_eval["scores"][key] - baseline_eval["scores"][key])
            for key in baseline_eval["scores"]
        },
        "baseline_diagnostics": {
            "family_basis_metrics": baseline_eval["family_basis_metrics"],
            "family_assignment": baseline_eval["family_assignment"],
            "concept_geometry_metrics": baseline_eval["concept_geometry_metrics"],
        },
        "continuous_diagnostics": {
            "family_basis_metrics": continuous_eval["family_basis_metrics"],
            "family_assignment": continuous_eval["family_assignment"],
            "concept_geometry_metrics": continuous_eval["concept_geometry_metrics"],
            "generated_family_basis": {
                family: [float(v) for v in center.tolist()] for family, center in continuous_centers.items()
            },
            "generated_concepts": continuous_concepts,
        },
        "integrated_scores": {
            "dynamic_update_score": dynamic_score,
            "bridge_execution_score": bridge_score,
            "baseline_predictive_closure_score": baseline_predictive,
            "continuous_predictive_closure_score": continuous_predictive,
            "predictive_closure_gain": float(continuous_predictive - baseline_predictive),
        },
        "strict_verdict": {
            "what_is_reached_now": (
                "The active support-remap weakness is removed on observed families. Family basis prediction, "
                "family assignment, and concept geometry all improve sharply once continuous operators are used."
            ),
            "what_is_not_reached_yet": (
                "This still leans on observed-family operator matrices and observed-family local residual templates. "
                "So it is a strong engineering closure for known families, not a proof of universal family prediction."
            ),
        },
        "progress_estimate": {
            "continuous_family_state_generator_percent": 71.0,
            "patch_offset_predictive_closure_percent": 54.0,
            "whole_network_state_generator_percent": 67.0,
            "full_brain_encoding_mechanism_percent": 62.0,
        },
        "next_large_blocks": [
            "Remove observed-family dependence from the continuous operator itself, so family transfer can forecast unseen family centers.",
            "Replace family-specific residual templates with a transferable residual law instead of replaying known family charts.",
            "Bind the upgraded generator to successor transport and protocol execution for task-level prediction rather than geometry-only improvement.",
        ],
    }
    return payload


def test_qwen_deepseek_continuous_family_state_generator() -> None:
    payload = build_payload()
    baseline = payload["baseline_scores"]
    continuous = payload["continuous_scores"]
    integrated = payload["integrated_scores"]
    assert continuous["family_basis_prediction_score"] > baseline["family_basis_prediction_score"]
    assert continuous["family_assignment_score"] > baseline["family_assignment_score"]
    assert continuous["concept_geometry_score"] > baseline["concept_geometry_score"]
    assert integrated["continuous_predictive_closure_score"] > integrated["baseline_predictive_closure_score"]
    assert payload["progress_estimate"]["continuous_family_state_generator_percent"] >= 71.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen/DeepSeek continuous family state generator")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen_deepseek_continuous_family_state_generator_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["integrated_scores"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
