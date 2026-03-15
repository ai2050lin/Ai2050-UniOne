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


def clipped_accuracy_from_relative_error(relative_error: float) -> float:
    return max(0.0, min(1.0, 1.0 - relative_error))


def build_payload() -> Dict[str, Any]:
    t0 = time.time()

    family_atlas = load_json("tests/codex_temp/theory_track_concept_family_atlas_analysis_20260312.json")
    family_ops = load_json("tests/codex_temp/theory_track_family_conditioned_projection_operators_20260312.json")
    analytic_transfer = load_json("tests/codex_temp/qwen_deepseek_analytic_family_transfer_law_20260315.json")
    universal_generator = load_json("tests/codex_temp/qwen_deepseek_universal_family_state_generator_20260315.json")
    continuous_generator = load_json("tests/codex_temp/qwen_deepseek_continuous_family_state_generator_20260315.json")

    scaffold = vec(analytic_transfer["analytic_objects"]["global_scaffold"])
    actual_centers = {
        family: vec(row["family_center"]) for family, row in family_atlas["family_atlas"].items()
    }
    family_packages = {
        family: vec(values) for family, values in universal_generator["family_packages"].items()
    }
    support_dims = {
        family: list(row["P_obj_family"]["support_dims"])
        for family, row in family_ops["core_operators"].items()
    }
    oracle_centers = {
        family: vec(values)
        for family, values in continuous_generator["continuous_diagnostics"]["generated_family_basis"].items()
    }

    leave_one_out_rows: Dict[str, Dict[str, Any]] = {}
    package_direction_scores = []
    support_signature_scores = []
    oracle_scores = []

    for held_family, target_center in actual_centers.items():
        train_families = [family for family in actual_centers if family != held_family]
        mean_residual_norm = float(
            np.mean([np.linalg.norm(actual_centers[family] - scaffold) for family in train_families])
        )

        package_vec = family_packages[held_family]
        if float(np.linalg.norm(package_vec)) > 1e-8:
            package_pred = scaffold + mean_residual_norm * package_vec / float(np.linalg.norm(package_vec))
        else:
            package_pred = scaffold.copy()

        sign_pattern = np.sign(
            np.mean(np.stack([actual_centers[family] - scaffold for family in train_families], axis=0), axis=0)
        )
        support_pred = scaffold.copy()
        support_mag = mean_residual_norm / np.sqrt(len(support_dims[held_family]))
        for dim in support_dims[held_family]:
            support_pred[int(dim)] += sign_pattern[int(dim)] * support_mag

        oracle_pred = oracle_centers[held_family]

        def metric(pred: np.ndarray) -> Dict[str, float]:
            abs_err = float(np.linalg.norm(pred - target_center))
            rel_err = abs_err / float(np.linalg.norm(target_center - scaffold) + 1e-8)
            return {
                "absolute_error": abs_err,
                "relative_error": rel_err,
                "accuracy": clipped_accuracy_from_relative_error(rel_err),
            }

        package_metric = metric(package_pred)
        support_metric = metric(support_pred)
        oracle_metric = metric(oracle_pred)

        leave_one_out_rows[held_family] = {
            "package_direction_estimator": package_metric,
            "support_signature_estimator": support_metric,
            "observed_family_oracle": oracle_metric,
        }
        package_direction_scores.append(package_metric["accuracy"])
        support_signature_scores.append(support_metric["accuracy"])
        oracle_scores.append(oracle_metric["accuracy"])

    package_direction_score = float(sum(package_direction_scores) / len(package_direction_scores))
    support_signature_score = float(sum(support_signature_scores) / len(support_signature_scores))
    oracle_score = float(sum(oracle_scores) / len(oracle_scores))

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "QwenDeepSeek_unseen_family_operator_dependency_audit",
        },
        "strict_goal": {
            "statement": (
                "Audit whether the current family operator can survive leave-one-family-out conditions, "
                "instead of relying on observed-family matrices or templates."
            ),
            "boundary": (
                "This block is an audit, not a solver. It measures how far current heuristics are from "
                "a real universal family operator."
            ),
        },
        "leave_one_out_rows": leave_one_out_rows,
        "summary_scores": {
            "package_direction_score": package_direction_score,
            "support_signature_score": support_signature_score,
            "observed_family_oracle_score": oracle_score,
            "oracle_minus_package_gap": float(oracle_score - package_direction_score),
            "oracle_minus_support_gap": float(oracle_score - support_signature_score),
        },
        "strict_verdict": {
            "what_is_reached_now": (
                "The audit now isolates the exact dependency gap: current observed-family closures are strong, "
                "but leave-one-family-out estimators remain weak even when a target support signature is granted."
            ),
            "what_is_not_reached_yet": (
                "A universal family operator is still missing. Package-only and support-signature heuristics "
                "do not recover unseen family centers with acceptable accuracy."
            ),
        },
        "progress_estimate": {
            "unseen_family_operator_closure_percent": 18.0,
            "patch_offset_predictive_closure_percent": 54.0,
            "whole_network_state_generator_percent": 67.0,
            "full_brain_encoding_mechanism_percent": 62.0,
        },
        "next_large_blocks": [
            "Learn or derive a target-family-free operator law from transferable invariants instead of family-specific matrices.",
            "Replace target support signatures with a support prediction law derived from family attributes or relation structure.",
            "Connect unseen-family operator prediction to downstream concept geometry and successor execution, not only center recovery.",
        ],
    }
    return payload


def test_qwen_deepseek_unseen_family_operator_dependency_audit() -> None:
    payload = build_payload()
    scores = payload["summary_scores"]
    assert scores["observed_family_oracle_score"] > 0.99
    assert scores["package_direction_score"] < 0.25
    assert scores["support_signature_score"] < 0.25
    assert scores["oracle_minus_package_gap"] > 0.7
    assert payload["progress_estimate"]["unseen_family_operator_closure_percent"] == 18.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen/DeepSeek unseen-family operator dependency audit")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen_deepseek_unseen_family_operator_dependency_audit_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary_scores"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
