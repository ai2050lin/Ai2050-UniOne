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
    grounding = load_json("continuous_input_grounding_precision_scan_20260309.json")
    multimodal = load_json("continuous_multimodal_grounding_proto_20260309.json")
    open_world = load_json("open_world_continuous_grounding_stream_20260310.json")
    g4 = load_json("g4_brain_direct_falsification_master_20260311.json")
    g5 = load_json("g5_brain_experiment_protocol_observable_mapping_20260311.json")

    direct = grounding["systems"]["direct_prototype"]
    dual_store = grounding["systems"]["dual_store_route"]
    cross_modal = grounding["systems"]["cross_modal_dual_store"]

    continuous_grounding_score = mean(
        [
            float(open_world["systems"]["direct_stream"]["closure_score"]),
            float(direct["grounding_score"]),
            float(dual_store["retention_concept_accuracy"]),
            float(cross_modal["retention_concept_accuracy"]),
        ]
    )

    multimodal_consistency_score = mean(
        [
            float(multimodal["systems"]["direct_multimodal"]["crossmodal_consistency"]),
            float(multimodal["systems"]["shared_offset_multimodal"]["crossmodal_consistency"]),
            float(multimodal["systems"]["shared_offset_multimodal"]["overall_concept_accuracy"]),
            float(multimodal["systems"]["direct_multimodal"]["grounding_score"]),
        ]
    )

    brain_execution_readiness_score = mean(
        [
            float(g4["headline_metrics"]["overall_g4_score"]),
            float(g5["headline_metrics"]["overall_g5_score"]),
            float(g4["headline_metrics"]["directional_specificity_score"]),
            float(g5["headline_metrics"]["observable_coverage_score"]),
        ]
    )

    overall_stage_c_score = mean(
        [
            continuous_grounding_score,
            multimodal_consistency_score,
            brain_execution_readiness_score,
        ]
    )

    hypotheses = {
        "H1_continuous_grounding_is_nontrivial": continuous_grounding_score >= 0.5,
        "H2_multimodal_consistency_is_still_weak": multimodal_consistency_score < 0.45,
        "H3_brain_execution_path_is_ready": brain_execution_readiness_score >= 0.75,
        "H4_stage_c_is_not_closed": overall_stage_c_score < 0.65,
    }

    if overall_stage_c_score >= 0.7 and multimodal_consistency_score >= 0.55:
        status = "stage_c_joint_closure_ready"
    elif overall_stage_c_score >= 0.58 and brain_execution_readiness_score >= 0.75:
        status = "stage_c_partial_external_closure"
    else:
        status = "stage_c_external_closure_not_ready"

    verdict = {
        "status": status,
        "core_answer": (
            "Stage C is asymmetric: continuous grounding and brain-side execution readiness are both real, but multimodal consistency is still too weak to count as a closed external loop."
        ),
        "main_open_gap": "multimodal_consistency_and_real_execution_gap",
        "next_step": (
            "Do not re-score brain readiness again. Build one C1 master that joins continuous grounding, multimodal consistency, and executable brain protocol in the same external loop."
        ),
    }

    interpretation = {
        "grounding": "Continuous grounding already exists as a nontrivial stream capability, but it is still not a strong novel-retention closed loop.",
        "multimodal": "Multimodal evidence exists, but consistency remains weak and does not yet dominate the decision boundary.",
        "brain": "Brain-side falsification and protocol design are already ready enough to execute; they are not the main bottleneck anymore.",
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "StageC_external_closure_master",
        },
        "headline_metrics": {
            "continuous_grounding_score": continuous_grounding_score,
            "multimodal_consistency_score": multimodal_consistency_score,
            "brain_execution_readiness_score": brain_execution_readiness_score,
            "overall_stage_c_score": overall_stage_c_score,
        },
        "supporting_readout": {
            "open_world_direct_closure": float(open_world["systems"]["direct_stream"]["closure_score"]),
            "direct_grounding_score": float(direct["grounding_score"]),
            "dual_store_retention": float(dual_store["retention_concept_accuracy"]),
            "cross_modal_retention": float(cross_modal["retention_concept_accuracy"]),
            "direct_multimodal_consistency": float(multimodal["systems"]["direct_multimodal"]["crossmodal_consistency"]),
            "shared_offset_multimodal_consistency": float(multimodal["systems"]["shared_offset_multimodal"]["crossmodal_consistency"]),
            "g4_score": float(g4["headline_metrics"]["overall_g4_score"]),
            "g5_score": float(g5["headline_metrics"]["overall_g5_score"]),
        },
        "hypotheses": hypotheses,
        "interpretation": interpretation,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "stage_c_external_closure_master_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
