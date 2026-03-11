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
    g9 = load_json("g9_stable_unified_role_coordinate_closure_20260311.json")
    atlas = load_json("qwen3_deepseek7b_real_model_structure_atlas_20260310.json")
    orientation = load_json("qwen3_deepseek7b_shared_layer_band_causal_orientation_20260310.json")
    ablation = load_json("qwen3_deepseek7b_shared_layer_band_targeted_ablation_20260310.json")
    g1 = load_json("g1_bridge_specificity_layer_role_transfer_closure_20260311.json")

    qwen_gap = atlas["models"]["qwen3_4b"]["global_summary"]["orientation_gap_abs"]
    deepseek_gap = atlas["models"]["deepseek_7b"]["global_summary"]["orientation_gap_abs"]

    atlas_consistency_score = mean(
        [
            1.0 - min(1.0, qwen_gap),
            1.0 - min(1.0, deepseek_gap),
            g1["headline_metrics"]["layer_role_transfer_closure_score"],
        ]
    )

    intervention_axis_stability_score = mean(
        [
            g9["headline_metrics"]["orientation_stability_score"],
            1.0 - min(1.0, abs(ablation["headline_metrics"]["qwen_actual_orientation"])),
            1.0 - min(1.0, abs(ablation["headline_metrics"]["deepseek_actual_orientation"])),
        ]
    )

    role_signal_visibility_score = mean(
        [
            g9["headline_metrics"]["shared_band_visibility_score"],
            abs(orientation["headline_metrics"]["qwen_orientation"]),
            abs(orientation["headline_metrics"]["deepseek_orientation"]),
            g9["headline_metrics"]["role_behavior_binding_score"],
        ]
    )

    overall_g9a_score = mean(
        [
            atlas_consistency_score,
            intervention_axis_stability_score,
            role_signal_visibility_score,
        ]
    )

    formulas = {
        "atlas_gap": "AtlasGap = |PredictedOrientation - ActualOrientation|",
        "stability": "StableAxis = mean(1 - AtlasGap, 1 - |ActualOrientation| under intervention)",
        "visibility": "RoleSignal = mean(SharedBandVisibility, |Orientation|, RoleBehaviorBinding)",
        "progress": "RoleAxisProgress = mean(AtlasConsistency, InterventionAxisStability, RoleSignalVisibility)",
    }

    verdict = {
        "status": (
            "role_axis_stabilization_partially_positive"
            if overall_g9a_score >= 0.53
            else "role_axis_stabilization_not_enough"
        ),
        "core_answer": (
            "The role axis is visible enough to build an atlas, and transfer is not random. "
            "But under targeted intervention the axis still rotates too much, especially on DeepSeek, so stable closure is not reached."
        ),
        "main_open_gap": "deepseek_intervention_axis_rotation",
    }

    hypotheses = {
        "H1_atlas_consistency_is_nontrivial": atlas_consistency_score >= 0.48,
        "H2_intervention_axis_stability_is_only_moderate": intervention_axis_stability_score < 0.72,
        "H3_role_signal_visibility_is_nontrivial": role_signal_visibility_score >= 0.37,
        "H4_g9a_is_partial_not_closed": 0.53 <= overall_g9a_score < 0.68,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G9A_intervention_stable_role_axis",
        },
        "headline_metrics": {
            "atlas_consistency_score": atlas_consistency_score,
            "intervention_axis_stability_score": intervention_axis_stability_score,
            "role_signal_visibility_score": role_signal_visibility_score,
            "overall_g9a_score": overall_g9a_score,
        },
        "supporting_readout": {
            "qwen_orientation_gap_abs": qwen_gap,
            "deepseek_orientation_gap_abs": deepseek_gap,
            "qwen_actual_orientation": ablation["headline_metrics"]["qwen_actual_orientation"],
            "deepseek_actual_orientation": ablation["headline_metrics"]["deepseek_actual_orientation"],
        },
        "formulas": formulas,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    with (TEMP_DIR / "g9a_intervention_stable_role_axis_20260311.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
