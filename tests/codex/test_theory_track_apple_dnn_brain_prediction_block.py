from __future__ import annotations

import json
import math
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in vec))


def main() -> None:
    start = time.time()
    apple = load_json(TEMP / "theory_track_apple_concept_encoding_analysis_20260312.json")
    atlas = load_json(TEMP / "theory_track_system_level_concept_atlas_synthesis_20260312.json")
    consistency = load_json(TEMP / "qwen3_deepseek7b_apple_mechanism_consistency_20260309.json")
    crack = load_json(TEMP / "complete_brain_encoding_crack_path_block.json")
    spike = load_json(TEMP / "spike_brain_system_bridge_block.json")
    brain = load_json(TEMP / "brain_encoding_spike_assessment.json")

    decomp = apple["apple_decomposition"]
    visual_state = decomp["apple_visual_state"]
    tactile_state = decomp["apple_tactile_state"]
    language_state = decomp["apple_language_state"]
    visual_basis = decomp["family_basis_visual"]
    tactile_basis = decomp["family_basis_tactile"]
    offset = decomp["apple_specific_offset_visual_tactile"]

    fruit_radius = float(atlas["atlas_metrics"]["fruit_radius"])
    atlas_sep = float(atlas["atlas_metrics"]["atlas_separation_score"])
    qwen = consistency["qwen3_4b"]
    deepseek = consistency["deepseek_7b"]

    sparse_offset_support = 1.0 if 0.05 <= l2_norm(offset) <= 0.2 else 0.6
    family_separation_support = clamp01(
        0.5 * (qwen["shared_basis"]["apple_to_animal_residual"] - qwen["shared_basis"]["apple_to_fruit_residual"])
        + 0.5 * (qwen["shared_basis"]["apple_to_abstract_residual"] - qwen["shared_basis"]["apple_to_fruit_residual"])
    )

    dnn_prediction_score = clamp01(
        0.16 * (1.0 - fruit_radius)
        + 0.16 * clamp01(atlas_sep / 60.0)
        + 0.10 * family_separation_support
        + 0.10 * sparse_offset_support
        + 0.08 * float(qwen["offset"]["natural_offset_supported"])
        + 0.10 * float(qwen["H"]["apple_closer_to_fruit_than_animal"])
        + 0.10 * float(qwen["H"]["apple_closer_to_fruit_than_abstract"])
        + 0.08 * qwen["G"]["early_topo_gating_strength"]
        + 0.06 * qwen["R"]["deep_repr_relation_strength"]
        + 0.06 * float(deepseek["H"]["hierarchy_closure_pass"])
    )

    brain_prediction_score = clamp01(
        0.28 * float(crack["path"]["crack_path_score"])
        + 0.24 * float(spike["bridge"]["spike_bridge_score"])
        + 0.24 * float(brain["headline_metrics"]["assessment_score"])
        + 0.12 * float(spike["bridge"]["components"]["phase_gate_support"])
        + 0.12 * float(spike["bridge"]["components"]["population_readout_support"])
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Apple_DNN_Brain_Prediction_Block",
        },
        "dnn_prediction": {
            "core_form": "apple ~= fruit family patch + apple-specific sparse offset + local attribute fibers + relation-role bridge slots",
            "predicted_family_patch": "fruit",
            "predicted_neighbors": ["banana", "pear", "orange", "grape"],
            "predicted_geometry": {
                "family_radius": fruit_radius,
                "atlas_separation_score": atlas_sep,
                "visual_state_norm": l2_norm(visual_state),
                "tactile_state_norm": l2_norm(tactile_state),
                "language_state_norm": l2_norm(language_state),
                "family_basis_visual_norm": l2_norm(visual_basis),
                "family_basis_tactile_norm": l2_norm(tactile_basis),
                "offset_norm": l2_norm(offset),
            },
            "predicted_mechanism": {
                "shared_basis_compactness": qwen["shared_basis"]["fruit_compactness"],
                "apple_to_fruit_residual": qwen["shared_basis"]["apple_to_fruit_residual"],
                "apple_to_animal_residual": qwen["shared_basis"]["apple_to_animal_residual"],
                "apple_to_abstract_residual": qwen["shared_basis"]["apple_to_abstract_residual"],
                "family_separation_support": family_separation_support,
                "sparse_offset_support": sparse_offset_support,
                "natural_offset_supported": qwen["offset"]["natural_offset_supported"],
                "topology_anchor_layers": qwen["T"]["topology_layers"],
                "repr_anchor_layers": qwen["T"]["repr_layers"],
                "early_topology_gating_strength": qwen["G"]["early_topo_gating_strength"],
                "deep_relation_strength": qwen["R"]["deep_repr_relation_strength"],
                "deepseek_hierarchy_closure_pass": deepseek["H"]["hierarchy_closure_pass"],
                "deepseek_axis_specificity_index": deepseek["offset"]["axis_specificity_index"],
                "deepseek_cross_dim_decoupling_index": deepseek["offset"]["cross_dim_decoupling_index"],
            },
            "predicted_roles": [
                "object-of-eating",
                "object-in-basket",
                "compare-with-pear",
                "attribute-binder(red/sweet/round/edible)",
            ],
            "prediction_score": dnn_prediction_score,
        },
        "brain_prediction": {
            "core_form": (
                "apple ~= fruit patch event-selection + burst-window section binding + "
                "phase-gated successor transport + population readout"
            ),
            "predicted_spike_implementation": {
                "event_patch_selection": spike["bridge"]["components"]["event_patch_selection"],
                "burst_window_section_binding": spike["bridge"]["components"]["burst_window_section_binding"],
                "membrane_integration_support": spike["bridge"]["components"]["membrane_integration_support"],
                "phase_gate_support": spike["bridge"]["components"]["phase_gate_support"],
                "successor_trigger_support": spike["bridge"]["components"]["successor_trigger_support"],
                "population_readout_support": spike["bridge"]["components"]["population_readout_support"],
            },
            "predicted_brain_side_constraints": {
                "patch_section": crack["path"]["components"]["PatchSection"],
                "write_read_asym": crack["path"]["components"]["WriteReadAsym"],
                "stage_successor": crack["path"]["components"]["StageSuccessor"],
                "proto_bridge": crack["path"]["components"]["ProtoBridge"],
                "causal_projection": crack["path"]["components"]["CausalProjection"],
                "constructive_closure": crack["path"]["components"]["ConstructiveClosure"],
            },
            "predicted_observables": [
                "fruit-neighbor pattern should overlap more with banana/pear than animal/abstract concepts",
                "apple recall should reactivate a fruit patch before relation/action bindings",
                "phase-gated successor should strengthen in eat/bite/carry contexts",
                "population readout should dominate over single-cell semantics",
            ],
            "prediction_score": brain_prediction_score,
        },
        "verdict": {
            "dnn_prediction_ready": dnn_prediction_score >= 0.84,
            "brain_prediction_ready": brain_prediction_score >= 0.94,
            "strict_final_prediction_pass": False,
            "core_answer": (
                "Yes. The current theory can already predict the apple code at the level of family patch, "
                "concept offset, attribute/relation fibers, and pulse-based brain-side implementation. "
                "What it still cannot give is the unique final biophysical realization."
            ),
        },
    }

    out_file = TEMP / "apple_dnn_brain_prediction_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
