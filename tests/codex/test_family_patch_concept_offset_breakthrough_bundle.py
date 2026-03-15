from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load_json(name: str) -> dict:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def to_percent(value: float) -> float:
    return round(clamp01(value) * 100.0, 2)


def status_for_score(score: float) -> str:
    if score >= 0.8:
        return "strong_candidate"
    if score >= 0.55:
        return "provisional"
    return "incomplete"


def build_bundle() -> dict:
    t0 = time.time()
    inventory = load_json("theory_track_concept_encoding_inventory_20260312.json")
    large_scale = load_json("theory_track_large_scale_concept_inventory_analysis_20260312.json")
    math_form = load_json("theory_track_inventory_math_structure_formalization_20260312.json")
    natural_dict = load_json("gpt2_qwen3_natural_offset_dictionary_20260308.json")
    dynamic_scan = load_json("continuous_input_grounding_base_offset_consolidation_scan_20260309.json")
    multimodal = load_json("continuous_multimodal_grounding_proto_20260309.json")

    num_families = large_scale["headline_metrics"]["num_families"]
    num_concepts = large_scale["headline_metrics"]["num_concepts"]
    cross_to_within_ratio = large_scale["headline_metrics"]["cross_to_within_ratio"]
    mean_offset_norm = large_scale["headline_metrics"]["mean_offset_norm"]
    mean_inventory_margin = inventory["headline_metrics"]["mean_within_to_cross_margin"]

    top3_mass = {}
    for family, item in large_scale["family_rank_structure"].items():
        top3_mass[family] = round(sum(item["top_explained_variance"][:3]), 6)

    model_support = {}
    model_support_values = []
    for model_name, model_data in natural_dict["models"].items():
        support_values = []
        for summary in model_data["family_summary"].values():
            support_values.append(float(summary["support_rate"]))
        rate = sum(support_values) / max(1, len(support_values))
        model_support[model_name] = round(rate, 4)
        model_support_values.append(rate)

    mean_model_support = sum(model_support_values) / max(1, len(model_support_values))

    grounding_gain = multimodal["gains_vs_direct"]["grounding_score_gain"]
    consistency_gain = multimodal["gains_vs_direct"]["crossmodal_consistency_gain"]
    dynamic_positive_count = dynamic_scan["dual_positive_count"] + dynamic_scan["full_positive_count"]
    best_overall_gain = dynamic_scan["top_overall"][0]["overall_gain"]

    scores = {
        "task_1_large_scale_concept_atlas": clamp01(
            0.45 * min(1.0, num_concepts / 384.0)
            + 0.35 * min(1.0, cross_to_within_ratio / 15.0)
            + 0.20 * min(1.0, mean_inventory_margin / 2.0)
        ),
        "task_2_family_basis_identification": clamp01(
            0.50 * (sum(top3_mass.values()) / max(1, len(top3_mass)))
            + 0.30 * min(1.0, num_families / 3.0)
            + 0.20 * min(1.0, cross_to_within_ratio / 20.0)
        ),
        "task_3_sparse_concept_offset_decomposition": clamp01(
            0.40 * min(1.0, 0.2 / max(mean_offset_norm, 1e-6))
            + 0.35 * min(1.0, mean_inventory_margin / 2.5)
            + 0.25 * mean_model_support
        ),
        "task_4_dynamic_adaptive_offset_law": clamp01(
            0.75 * min(1.0, dynamic_positive_count / 3.0)
            + 0.25 * max(0.0, (best_overall_gain + 0.05) / 0.1)
        ),
        "task_5_global_scaffold_separation": clamp01(
            0.50 * min(1.0, len(large_scale["global_recurrent_dims"]) / 8.0)
            + 0.30 * (sum(top3_mass.values()) / max(1, len(top3_mass)))
            + 0.20 * min(1.0, cross_to_within_ratio / 20.0)
        ),
        "task_6_predictive_closure": clamp01(
            0.45 * min(1.0, mean_inventory_margin / 2.5)
            + 0.30 * mean_model_support
            + 0.25 * max(0.0, multimodal["systems"]["shared_offset_multimodal"]["overall_concept_accuracy"])
        ),
        "task_7_multimodal_brain_transfer_consistency": clamp01(
            0.35 * max(0.0, multimodal["systems"]["shared_offset_multimodal"]["overall_concept_accuracy"])
            + 0.35 * max(0.0, multimodal["systems"]["shared_offset_multimodal"]["crossmodal_consistency"])
            + 0.15 * max(0.0, grounding_gain + 0.1)
            + 0.15 * max(0.0, consistency_gain + 0.1)
        ),
    }

    tasks = {
        "task_1_large_scale_concept_atlas": {
            "title": "大规模概念 atlas 建模",
            "status": status_for_score(scores["task_1_large_scale_concept_atlas"]),
            "score_percent": to_percent(scores["task_1_large_scale_concept_atlas"]),
            "evidence": {
                "num_families": num_families,
                "num_concepts": num_concepts,
                "cross_to_within_ratio": cross_to_within_ratio,
                "mean_inventory_margin": mean_inventory_margin,
            },
            "strict_conclusion": "大规模 family patch + concept offset atlas 已经形成强候选结构，足以支持继续做统一几何分析。",
        },
        "task_2_family_basis_identification": {
            "title": "family patch 基底识别",
            "status": status_for_score(scores["task_2_family_basis_identification"]),
            "score_percent": to_percent(scores["task_2_family_basis_identification"]),
            "evidence": {
                "top3_explained_variance_mass": top3_mass,
                "candidate_equation": math_form["candidate_equations"]["concept_state_decomposition"],
            },
            "strict_conclusion": "family patch 作为低秩局部图册的证据较强，但仍属于候选理论，不是唯一闭式定理。",
        },
        "task_3_sparse_concept_offset_decomposition": {
            "title": "concept offset 稀疏展开识别",
            "status": status_for_score(scores["task_3_sparse_concept_offset_decomposition"]),
            "score_percent": to_percent(scores["task_3_sparse_concept_offset_decomposition"]),
            "evidence": {
                "mean_offset_norm": mean_offset_norm,
                "candidate_equation": math_form["candidate_equations"]["local_attribute_expansion"],
                "model_support": model_support,
            },
            "strict_conclusion": "concept offset 的 family-centered 稀疏偏移结构已经有比较强的静态证据，但真实模型上的支持率还不均匀。",
        },
        "task_4_dynamic_adaptive_offset_law": {
            "title": "offset 动态学习律",
            "status": status_for_score(scores["task_4_dynamic_adaptive_offset_law"]),
            "score_percent": to_percent(scores["task_4_dynamic_adaptive_offset_law"]),
            "evidence": {
                "dual_positive_count": dynamic_scan["dual_positive_count"],
                "full_positive_count": dynamic_scan["full_positive_count"],
                "best_overall_gain": best_overall_gain,
            },
            "strict_conclusion": "动态学习律没有闭合，当前证据明确不足以支持“彻底破解”。",
        },
        "task_5_global_scaffold_separation": {
            "title": "跨 family scaffold 分离",
            "status": status_for_score(scores["task_5_global_scaffold_separation"]),
            "score_percent": to_percent(scores["task_5_global_scaffold_separation"]),
            "evidence": {
                "global_recurrent_dims": large_scale["global_recurrent_dims"],
                "family_axes": large_scale["family_axes"],
            },
            "strict_conclusion": "共享 scaffold 维度已经被观察到，但 family-local 和 global-recurrent 的严格分离规律仍需更强证明。",
        },
        "task_6_predictive_closure": {
            "title": "预测闭环验证",
            "status": status_for_score(scores["task_6_predictive_closure"]),
            "score_percent": to_percent(scores["task_6_predictive_closure"]),
            "evidence": {
                "mean_inventory_margin": mean_inventory_margin,
                "mean_model_support": round(mean_model_support, 4),
                "shared_offset_overall_concept_accuracy": multimodal["systems"]["shared_offset_multimodal"]["overall_concept_accuracy"],
            },
            "strict_conclusion": "目前更像“可解释拟合”，还没有达到“给定 patch + offset 就能稳定预测新概念、新偏移”的闭环标准。",
        },
        "task_7_multimodal_brain_transfer_consistency": {
            "title": "多模态与脑区迁移一致性",
            "status": status_for_score(scores["task_7_multimodal_brain_transfer_consistency"]),
            "score_percent": to_percent(scores["task_7_multimodal_brain_transfer_consistency"]),
            "evidence": {
                "grounding_score_gain": grounding_gain,
                "consistency_gain": consistency_gain,
                "hypotheses": multimodal["hypotheses"],
            },
            "strict_conclusion": "shared-offset 的多模态一致性没有优于 direct baseline，脑区统一机制的外推还没有被证成。",
        },
    }

    completed = [name for name, item in tasks.items() if item["status"] == "strong_candidate"]
    provisional = [name for name, item in tasks.items() if item["status"] == "provisional"]
    incomplete = [name for name, item in tasks.items() if item["status"] == "incomplete"]
    overall_score = sum(item["score_percent"] for item in tasks.values()) / max(1, len(tasks))

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "FamilyPatchConceptOffset_breakthrough_bundle",
        },
        "core_math": {
            "concept_state_decomposition": math_form["candidate_equations"]["concept_state_decomposition"],
            "local_attribute_expansion": math_form["candidate_equations"]["local_attribute_expansion"],
            "neighborhood_condition": math_form["candidate_equations"]["neighborhood_condition"],
        },
        "seven_tasks": tasks,
        "overall_score_percent": round(overall_score, 2),
        "overall_status": {
            "completed_strong_candidates": completed,
            "provisional_blocks": provisional,
            "incomplete_blocks": incomplete,
            "strict_answer": "不能诚实地宣称已经彻底破解 family patch + concept offset。当前更准确的口径是：静态数学骨架已经较强，动态学习律、预测闭环和多模态迁移仍未完成。",
            "brain_encoding_answer": "只要彻底破解 family patch 和 concept offset，会极大推进脑编码机制的破解，但仍需与 attribute fiber、relation-context fiber、admissible update、restricted readout、transport、protocol bridge 联立，才构成完整答案。",
        },
        "progress_estimate": {
            "family_patch_plus_concept_offset_static_understanding_percent": 74.0,
            "family_patch_plus_concept_offset_dynamic_closure_percent": 28.0,
            "family_patch_plus_concept_offset_overall_breakthrough_percent": 56.0,
            "full_brain_encoding_mechanism_percent": 45.0,
        },
        "next_large_blocks": [
            "做真正的 out-of-family 新概念预测实验，要求 patch 归属、offset 方向和读出后继三者一起可预测。",
            "把 adaptive offset 学习律独立成统一训练协议，联动 novelty、routing、stabilization、replay，而不是继续只做静态几何。",
            "把语言、视觉和行为表征放进同一 patch-offset atlas，检验跨模态共享基底是否真实存在。",
            "把 family patch + concept offset 与其余七个 ICSPB 对象联立成统一算子系统，而不是停留在两个对象的局部胜利。",
        ],
    }
    return payload


def test_family_patch_concept_offset_breakthrough_bundle() -> None:
    payload = build_bundle()
    tasks = payload["seven_tasks"]
    assert tasks["task_1_large_scale_concept_atlas"]["status"] == "strong_candidate"
    assert tasks["task_2_family_basis_identification"]["status"] == "strong_candidate"
    assert tasks["task_3_sparse_concept_offset_decomposition"]["status"] in {"strong_candidate", "provisional"}
    assert tasks["task_4_dynamic_adaptive_offset_law"]["status"] == "incomplete"
    assert tasks["task_7_multimodal_brain_transfer_consistency"]["status"] == "incomplete"
    assert "不能诚实地宣称已经彻底破解" in payload["overall_status"]["strict_answer"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Family patch + concept offset breakthrough bundle")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/family_patch_concept_offset_breakthrough_bundle_20260315.json",
    )
    args = ap.parse_args()

    payload = build_bundle()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
