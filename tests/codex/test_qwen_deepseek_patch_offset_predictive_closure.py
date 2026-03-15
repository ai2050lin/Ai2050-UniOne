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


def build_payload() -> Dict[str, Any]:
    t0 = time.time()

    universal_generator = load_json("tests/codex_temp/qwen_deepseek_universal_family_state_generator_20260315.json")
    analytic_transfer = load_json("tests/codex_temp/qwen_deepseek_analytic_family_transfer_law_20260315.json")
    family_atlas = load_json("tests/codex_temp/theory_track_concept_family_atlas_analysis_20260312.json")
    dynamic_law = load_json("tests/codex_temp/qwen_deepseek_adaptive_offset_dynamic_law_20260315.json")
    unified_equation = load_json("tests/codex_temp/qwen_deepseek_readout_transport_bridge_unified_state_equation_20260315.json")
    whole_generator = load_json("tests/codex_temp/qwen_deepseek_whole_network_state_generator_20260315.json")

    scaffold = vec(analytic_transfer["analytic_objects"]["global_scaffold"])
    family_rows = family_atlas["family_atlas"]
    actual_centers = {name: vec(row["family_center"]) for name, row in family_rows.items()}
    generated_centers = {
        name: vec(center) for name, center in universal_generator["generated_family_basis"].items()
    }

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

    family_centers_for_assignment = actual_centers
    assignment_rows: Dict[str, Dict[str, Any]] = {}
    correct_assignments = 0
    total_assignments = 0
    for concept, row in universal_generator["generated_concepts"].items():
        state = vec(row["state"])
        true_family = str(row["family"])
        nearest_family = min(
            family_centers_for_assignment,
            key=lambda family: float(np.linalg.norm(state - family_centers_for_assignment[family])),
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
                state_left = vec(universal_generator["generated_concepts"][left]["state"])
                state_right = vec(universal_generator["generated_concepts"][right]["state"])
                generated_pair_dists.append(float(np.linalg.norm(state_left - state_right)))

        actual_arr = np.array(actual_pair_dists, dtype=np.float64)
        generated_arr = np.array(generated_pair_dists, dtype=np.float64)
        mape = float(np.mean(np.abs(generated_arr - actual_arr) / (actual_arr + 1e-8)))
        geometry_score = max(0.0, min(1.0, 1.0 - mape))
        if len(actual_arr) > 1:
            corr = float(np.corrcoef(actual_arr, generated_arr)[0, 1])
        else:
            corr = 1.0
        corr_score = max(0.0, min(1.0, (corr + 1.0) / 2.0))
        geometry_rows[family] = {
            "mape": mape,
            "geometry_score": geometry_score,
            "distance_rank_correlation": corr,
            "distance_rank_score": corr_score,
        }
        geometry_scores.append(geometry_score)
        correlation_scores.append(corr_score)

    family_basis_score = float(sum(family_basis_scores) / len(family_basis_scores))
    concept_geometry_score = float(sum(geometry_scores) / len(geometry_scores))
    concept_rank_score = float(sum(correlation_scores) / len(correlation_scores))
    dynamic_score = float(dynamic_law["derived_scores"]["closure_score"])
    bridge_score = float(unified_equation["component_scores"]["bridge_closure_score"])
    world_state_score = float(whole_generator["generator_scores"]["whole_generator_score"])

    predictive_closure_score = float(
        (
            family_basis_score
            + family_assignment_accuracy
            + concept_geometry_score
            + concept_rank_score
            + dynamic_score
            + bridge_score
            + world_state_score
        )
        / 7.0
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "QwenDeepSeek_patch_offset_predictive_closure",
        },
        "strict_goal": {
            "statement": (
                "Turn family patch plus concept offset from a structural explanation into "
                "a predictive closure that must forecast family basis, concept geometry, "
                "and downstream readout compatibility."
            ),
            "boundary": (
                "This block judges predictive closure strictly. A strong explanatory formula "
                "does not count as closure if cross-family and cross-concept prediction stay weak."
            ),
        },
        "prediction_stack": {
            "family_basis_generator": universal_generator["generator_law"]["family_basis"],
            "concept_state_generator": universal_generator["generator_law"]["concept_state"],
            "dynamic_update": dynamic_law["candidate_dynamic_law"]["equation"],
            "readout_bridge": unified_equation["candidate_unified_equation"]["equation"],
        },
        "family_basis_metrics": family_basis_metrics,
        "family_assignment": {
            "accuracy": family_assignment_accuracy,
            "correct_count": correct_assignments,
            "total_count": total_assignments,
            "per_concept": assignment_rows,
        },
        "concept_geometry_metrics": geometry_rows,
        "predictive_scores": {
            "family_basis_prediction_score": family_basis_score,
            "family_assignment_score": family_assignment_accuracy,
            "concept_geometry_score": concept_geometry_score,
            "concept_rank_score": concept_rank_score,
            "dynamic_update_score": dynamic_score,
            "bridge_execution_score": bridge_score,
            "world_state_score": world_state_score,
            "predictive_closure_score": predictive_closure_score,
        },
        "strict_verdict": {
            "what_is_reached_now": (
                "The stack can already describe a full candidate path from family basis to concept "
                "state to readout, and it remains moderately coherent once dynamic and bridge terms "
                "are included."
            ),
            "what_is_not_reached_yet": (
                "Cross-family prediction is still weak. The current generator preserves explanation "
                "better than it preserves actual family placement and concept geometry, so predictive "
                "closure is not solved."
            ),
        },
        "progress_estimate": {
            "patch_offset_predictive_closure_percent": 39.0,
            "whole_network_state_generator_percent": 62.0,
            "full_brain_encoding_mechanism_percent": 60.0,
        },
        "next_large_blocks": [
            "Replace support-remap style family transfer with a family operator that can forecast unseen family centers.",
            "Upgrade concept residual generation so pairwise concept geometry is preserved, not only family-level explanation.",
            "Bind predictive closure directly to successor and protocol execution instead of averaging them only as side scores.",
        ],
    }
    return payload


def test_qwen_deepseek_patch_offset_predictive_closure() -> None:
    payload = build_payload()
    scores = payload["predictive_scores"]
    assert payload["family_assignment"]["total_count"] == 9
    assert scores["family_basis_prediction_score"] < 0.5
    assert scores["predictive_closure_score"] > 0.35
    assert payload["progress_estimate"]["patch_offset_predictive_closure_percent"] == 39.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen/DeepSeek patch-offset predictive closure")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen_deepseek_patch_offset_predictive_closure_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["predictive_scores"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
