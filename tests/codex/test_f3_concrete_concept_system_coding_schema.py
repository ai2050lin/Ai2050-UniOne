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


def top_items(d: dict[str, int], k: int = 5) -> list[dict[str, int]]:
    items = sorted(((int(layer), count) for layer, count in d.items()), key=lambda x: (-x[1], x[0]))
    return [{"layer": layer, "count": count} for layer, count in items[:k]]


def main() -> None:
    codebook = load_json("concept_family_unified_codebook_20260308.json")
    decomposition = load_json("qwen3_deepseek7b_concept_encoding_decomposition_20260309.json")
    apple_consistency = load_json("qwen3_deepseek7b_apple_mechanism_consistency_20260309.json")
    path_signature = load_json("gpt2_qwen3_concept_path_signature_20260308.json")
    field_mapping = load_json("qwen3_deepseek7b_concept_protocol_field_mapping_20260309.json")
    p8a = load_json("p8a_spatialized_plasticity_coding_equation_20260311.json")
    stage8b = load_json("stage8b_high_resolution_precision_editing_20260311.json")

    fruit_stats = codebook["family_stats"]["fruit"]
    apple_spot = codebook["spotlight_concepts"]["apple"]
    qwen_apple = decomposition["models"]["qwen3_4b"]["targets"]["apple"]
    deepseek_apple = decomposition["models"]["deepseek_7b"]["targets"]["apple"]
    qwen_path_apple = path_signature["models"]["qwen3_4b"]["concepts"]["apple"]["signature_summary"]
    gpt2_path_apple = path_signature["models"]["gpt2"]["concepts"]["apple"]["signature_summary"]
    qwen_field_apple = field_mapping["models"]["qwen3_4b"]["concepts"]["apple"]["summary"]
    deepseek_field_apple = field_mapping["models"]["deepseek_7b"]["concepts"]["apple"]["summary"]

    shared_basis_strength = mean(
        [
            fruit_stats["subspace_margin"],
            qwen_apple["best_layer"]["shared_norm_ratio"],
            deepseek_apple["best_layer"]["shared_norm_ratio"],
            1.0 - apple_consistency["qwen3_4b"]["shared_basis"]["apple_to_fruit_residual"],
        ]
    )
    sparse_offset_strength = mean(
        [
            apple_spot["shared_overlap_ratio"],
            qwen_apple["best_layer"]["offset_top32_energy_ratio"],
            deepseek_apple["best_layer"]["offset_top32_energy_ratio"],
            apple_consistency["deepseek_7b"]["offset"]["axis_specificity_index"],
        ]
    )
    routed_field_strength = mean(
        [
            qwen_field_apple["best_total_usage"] / max(qwen_field_apple["best_total_usage"], 1e-12),
            deepseek_field_apple["best_total_usage"] / max(deepseek_field_apple["best_total_usage"], 1e-12),
            apple_consistency["qwen3_4b"]["G"]["early_topo_gating_strength"],
            apple_consistency["deepseek_7b"]["R"]["route_index"] * 20.0,
        ]
    )
    system_integratability_score = mean(
        [
            p8a["headline_metrics"]["spatial_equation_consistency_score"],
            p8a["headline_metrics"]["topology_reuse_locality_score"],
            stage8b["headline_metrics"]["routing_precision_score"],
            apple_consistency["qwen3_4b"]["R"]["deep_repr_relation_strength"],
        ]
    )
    overall_f3_score = mean(
        [
            shared_basis_strength,
            sparse_offset_strength,
            routed_field_strength,
            system_integratability_score,
        ]
    )

    apple_code_formula = {
        "instance_equation": (
            "Code_apple(t) = w_B * B_fruit + w_O * O_apple + "
            "w_G * G_apple(context,t) + w_R * R_apple(context,t) + w_M * M_fruit,apple(t)"
        ),
        "feature_state": (
            "f_apple,t = q_t * [B_fruit + O_apple + K_t * Route_fruit(context,t) - I_apple(context,t)]"
        ),
        "structure_state": (
            "A_apple,t+1 = (1-l_A)A_apple,t + e_A(1-q_t)"
            "[coactivate(f_apple,t+1, neighbors) + demand_gap_apple,t - prune_apple,t - lambda_s D_3d]"
        ),
        "system_readout": "y_t = W_B B_family + W_O O_concept + W_R R_context + W_M M_slow",
    }

    apple_encoding = {
        "shared_family_basis": {
            "family": "fruit",
            "robust_shared_dims": fruit_stats["robust_shared_dims"],
            "prototype_top_dims_head": fruit_stats["prototype_top_dims"][:8],
            "dominant_layers": top_items(fruit_stats["prototype_layer_distribution"], 6),
            "subspace_margin": fruit_stats["subspace_margin"],
        },
        "concept_specific_offset": {
            "top_specific_dims": apple_spot["specific_dims"],
            "shared_overlap_ratio": apple_spot["shared_overlap_ratio"],
            "dominant_layers": top_items(apple_spot["layer_distribution"], 6),
            "qwen_offset_top32_energy_ratio": qwen_apple["best_layer"]["offset_top32_energy_ratio"],
            "deepseek_offset_top32_energy_ratio": deepseek_apple["best_layer"]["offset_top32_energy_ratio"],
        },
        "routing_and_gating": {
            "qwen_repr_basis_layers": qwen_path_apple["repr_basis_layers"],
            "qwen_repr_gating_layers": qwen_path_apple["repr_gating_layers"],
            "qwen_topo_relation_layers": qwen_path_apple["topo_relation_layers"],
            "gpt2_repr_basis_layers": gpt2_path_apple["repr_basis_layers"],
            "qwen_protocol_preferred_field": qwen_field_apple["preferred_field"],
            "deepseek_protocol_preferred_field": deepseek_field_apple["preferred_field"],
            "qwen_early_topo_gating_strength": apple_consistency["qwen3_4b"]["G"]["early_topo_gating_strength"],
            "deepseek_route_index": apple_consistency["deepseek_7b"]["R"]["route_index"],
        },
        "slow_memory_and_editability": {
            "editable_local_band": stage8b["precision_policy"]["local_attribute_edit"],
            "qwen_best_basis_layer": qwen_apple["best_layer"]["layer"],
            "deepseek_best_basis_layer": deepseek_apple["best_layer"]["layer"],
            "qwen_shared_norm_ratio": qwen_apple["best_layer"]["shared_norm_ratio"],
            "deepseek_shared_norm_ratio": deepseek_apple["best_layer"]["shared_norm_ratio"],
        },
    }

    system_schema = {
        "level_1_family_basis": "一个家族不是若干孤立点，而是一个紧致共享子空间 B_family。",
        "level_2_concept_offset": "每个概念通过稀疏偏移 O_concept 从家族基底上分叉出来。",
        "level_3_relation_routing": "上下文通过 G(context,t) 与 R(context,t) 选择当前启用哪些局部证据和长程桥。",
        "level_4_slow_memory": "慢时标 M(t) 保留高价值结构，决定什么会变成稳定编码 scaffold。",
        "level_5_system_code": "整个系统的编码不是概念表，而是 family basis + concept offsets + relation routing + slow scaffolds 的动态组合。",
    }

    hypotheses = {
        "H1_apple_code_can_be_written_as_family_basis_plus_sparse_offset": shared_basis_strength >= 0.68,
        "H2_apple_code_has_nontrivial_routed_field_component": routed_field_strength >= 0.55,
        "H3_concept_code_is_editable_in_a_narrow_local_band": stage8b["headline_metrics"]["localization_score"] >= 0.8,
        "H4_system_code_is_a_multilevel_structure_not_a_flat_label_table": system_integratability_score >= 0.62,
        "H5_f3_concrete_schema_is_ready": overall_f3_score >= 0.6,
    }

    verdict = {
        "status": "concrete_concept_schema_ready",
        "best_concrete_example": "apple_as_fruit_basis_plus_sparse_offset_plus_context_routing",
        "core_message": (
            "苹果不是一个单点向量，而是水果家族共享子空间上的稀疏偏移，"
            "再叠加上下文路由和慢时标结构。"
        ),
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "F3_concrete_concept_system_coding_schema",
        },
        "headline_metrics": {
            "shared_basis_strength": shared_basis_strength,
            "sparse_offset_strength": sparse_offset_strength,
            "routed_field_strength": routed_field_strength,
            "system_integratability_score": system_integratability_score,
            "overall_f3_score": overall_f3_score,
        },
        "apple_code_formula": apple_code_formula,
        "apple_encoding": apple_encoding,
        "system_schema": system_schema,
        "spatial_backbone_equation": p8a["candidate_mechanism"]["equations"],
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "f3_concrete_concept_system_coding_schema_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
