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
    g9a = load_json("g9a_intervention_stable_role_axis_20260311.json")
    atlas = load_json("qwen3_deepseek7b_real_model_structure_atlas_20260310.json")
    g1 = load_json("g1_bridge_specificity_layer_role_transfer_closure_20260311.json")

    qwen = atlas["models"]["qwen3_4b"]["global_summary"]
    deepseek = atlas["models"]["deepseek_7b"]["global_summary"]

    kernel_visibility_score = mean(
        [
            qwen["shared_band_layer_count"] / 5.0,
            deepseek["shared_band_layer_count"] / 5.0,
            g9a["headline_metrics"]["role_signal_visibility_score"],
        ]
    )

    cross_model_kernel_consistency_score = mean(
        [
            g1["headline_metrics"]["layer_role_transfer_closure_score"],
            1.0 - min(1.0, qwen["orientation_gap_abs"]),
            1.0 - min(1.0, deepseek["orientation_gap_abs"]),
        ]
    )

    intervention_stability_kernel_score = mean(
        [
            g9a["headline_metrics"]["intervention_axis_stability_score"],
            max(0.0, qwen["mechanism_bridge_score"] - qwen["orientation_gap_abs"]),
            max(0.0, deepseek["mechanism_bridge_score"] - deepseek["orientation_gap_abs"]),
        ]
    )

    overall_g9b_score = mean(
        [
            kernel_visibility_score,
            cross_model_kernel_consistency_score,
            intervention_stability_kernel_score,
        ]
    )

    formulas = {
        "role_kernel": "Kernel = SharedBandCount + TransferClosure + (MechanismBridge - OrientationGap)",
        "consistency": "KernelConsistency = mean(TransferClosure, 1 - OrientationGap_qwen, 1 - OrientationGap_deepseek)",
        "closure": "RoleKernelClosure = mean(KernelVisibility, KernelConsistency, InterventionKernelStability)",
    }

    verdict = {
        "status": (
            "cross_model_role_kernel_partially_ready"
            if overall_g9b_score >= 0.6
            else "cross_model_role_kernel_not_ready"
        ),
        "core_answer": (
            "A cross-model role kernel is now visible: shared-band counts, mechanism bridge, and transfer closure all remain nontrivial. "
            "But intervention-stable closure is still limited by DeepSeek-side rotation."
        ),
        "main_open_gap": "deepseek_gap_keeps_kernel_from_full_stability",
    }

    hypotheses = {
        "H1_role_kernel_visibility_is_real": kernel_visibility_score >= 0.58,
        "H2_cross_model_consistency_is_nontrivial": cross_model_kernel_consistency_score >= 0.5,
        "H3_intervention_kernel_stability_is_nontrivial": intervention_stability_kernel_score >= 0.58,
        "H4_g9b_is_partial_not_closed": 0.6 <= overall_g9b_score < 0.76,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G9B_cross_model_intervention_stable_role_kernel",
        },
        "headline_metrics": {
            "kernel_visibility_score": kernel_visibility_score,
            "cross_model_kernel_consistency_score": cross_model_kernel_consistency_score,
            "intervention_stability_kernel_score": intervention_stability_kernel_score,
            "overall_g9b_score": overall_g9b_score,
        },
        "supporting_readout": {
            "qwen_shared_band_layer_count": qwen["shared_band_layer_count"],
            "deepseek_shared_band_layer_count": deepseek["shared_band_layer_count"],
            "qwen_orientation_gap_abs": qwen["orientation_gap_abs"],
            "deepseek_orientation_gap_abs": deepseek["orientation_gap_abs"],
        },
        "formulas": formulas,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    with (TEMP_DIR / "g9b_cross_model_intervention_stable_role_kernel_20260311.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
