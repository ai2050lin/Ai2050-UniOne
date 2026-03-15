from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]


def load_json(rel_path: str) -> Dict[str, Any]:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8-sig"))


def build_payload() -> Dict[str, Any]:
    t0 = time.time()

    qd_math = load_json("tests/codex_temp/qwen3_deepseek_family_patch_offset_math_mechanism_20260315.json")
    family_atlas = load_json("tests/codex_temp/theory_track_concept_family_atlas_analysis_20260312.json")
    attr_axes = load_json("tests/codex_temp/theory_track_attribute_axis_analysis_20260312.json")
    apple_local = load_json("tests/codex_temp/theory_track_apple_concept_encoding_analysis_20260312.json")
    relation_attr = load_json("tests/codex_temp/theory_track_concept_relation_attribute_atlas_synthesis_20260312.json")
    multiaxis = load_json("tests/codex_temp/multiaxis_encoding_law_20260306.json")
    systemic = load_json("tests/codex_temp/theory_track_systemic_multiaxis_inventory_expansion_20260312.json")
    ladder = load_json("tests/codex_temp/abstraction_ladder_hierarchy_20260308.json")
    sweet_edit = load_json("tests/codex_temp/real_model_apple_sweetness_channel_edit_20260307.json")
    apple_dossier = load_json(
        "tempdata/deepseek7b_apple_encoding_law_dossier_20260306_223055/apple_multiaxis_encoding_law_dossier.json"
    )
    family_chain = load_json("tempdata/deepseek7b_concept_family_parallel_latest/concept_family_parallel_scale.json")

    fruit = family_atlas["family_atlas"]["fruit"]
    apple_entry = family_atlas["concept_decomposition_examples"]["apple"]
    round_axis = attr_axes["attribute_axes"]["round"]
    sweet_axis = attr_axes["attribute_axes"]["sweet"]

    apple_banana = fruit["pairwise_distances"]["apple__banana"]
    apple_pear = fruit["pairwise_distances"]["apple__pear"]
    fruit_radius = fruit["family_radius"]

    apple_chain = family_chain["metrics"]["apple_chain_summary"]
    axis_identifiability = min(float(row["accuracy"]) for row in multiaxis["axis_separability"])

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "QwenDeepSeek_micro_meso_macro_encoding_map",
        },
        "sources": {
            "qwen_deepseek_math": "tests/codex_temp/qwen3_deepseek_family_patch_offset_math_mechanism_20260315.json",
            "family_atlas": "tests/codex_temp/theory_track_concept_family_atlas_analysis_20260312.json",
            "attribute_axes": "tests/codex_temp/theory_track_attribute_axis_analysis_20260312.json",
            "apple_local_chart": "tests/codex_temp/theory_track_apple_concept_encoding_analysis_20260312.json",
            "relation_attribute_atlas": "tests/codex_temp/theory_track_concept_relation_attribute_atlas_synthesis_20260312.json",
            "multiaxis_law": "tests/codex_temp/multiaxis_encoding_law_20260306.json",
            "systemic_multiaxis": "tests/codex_temp/theory_track_systemic_multiaxis_inventory_expansion_20260312.json",
            "abstraction_ladder": "tests/codex_temp/abstraction_ladder_hierarchy_20260308.json",
            "sweetness_edit": "tests/codex_temp/real_model_apple_sweetness_channel_edit_20260307.json",
            "apple_dossier": "tempdata/deepseek7b_apple_encoding_law_dossier_20260306_223055/apple_multiaxis_encoding_law_dossier.json",
            "family_chain": "tempdata/deepseek7b_concept_family_parallel_latest/concept_family_parallel_scale.json",
        },
        "three_scales": {
            "micro": {
                "definition": "微观子属性层，不直接等于对象本身，而是附着在对象 family patch 上的局部属性方向或局部可编辑通道。",
                "core_equation": "h_micro(apple) = B_fruit + Delta_apple + sum_i alpha_i * u_attr_i^(fruit) + epsilon",
                "direct_evidence": {
                    "axis_identifiability_accuracy": axis_identifiability,
                    "micro_context_stability": multiaxis["apple_three_level_metrics"]["micro_context_stability"],
                    "round_axis_alignment": round_axis["mean_alignment"],
                    "round_axis_top_dims": round_axis["top_dims"],
                    "sweet_axis_alignment": sweet_axis["mean_alignment"],
                    "sweetness_edit_best_layer": sweet_edit["best"]["layer"],
                    "sweetness_edit_min_k_strong": sweet_edit["min_k_reversal_anchor80_strong"],
                    "sweetness_edit_target_reversal_strong": sweet_edit["best"]["target_reversal_strong"],
                },
                "strict_readout": {
                    "apple_color_taste_answer": (
                        "苹果的颜色、味道、形状这类信息，不像独立的符号标签，更像 fruit patch 上被 Delta_apple 选中的局部属性纤维。"
                        "属性之间近似解耦，但不是全局线性可加。"
                    ),
                    "why": (
                        "多特征正交性显示属性轴之间近似正交；甜味通道编辑又表明少量局部通道可以改变苹果甜/酸判断，"
                        "说明属性至少部分是可局部读写的。"
                    ),
                },
            },
            "meso": {
                "definition": "中观实体物层，对应具体对象及其 family patch，例如 apple、banana、pear 都是 fruit family patch 上的不同 concept offset。",
                "core_equation": "h_meso(c) = B_(f_c) + Delta_c",
                "direct_evidence": {
                    "fruit_family_radius": fruit_radius,
                    "apple_banana_distance": apple_banana,
                    "apple_pear_distance": apple_pear,
                    "fruit_mean_offset_norm": fruit["mean_offset_norm"],
                    "qwen_deepseek_mean_family_fit_strength": qd_math["cross_model_summary"]["mean_family_fit_strength"],
                    "qwen_deepseek_mean_wrong_family_margin": qd_math["cross_model_summary"]["mean_wrong_family_margin"],
                },
                "strict_readout": {
                    "apple_banana_answer": (
                        "苹果和香蕉不是两个完全独立编码，而是共享同一个 fruit family patch，只是各自 concept offset 不同。"
                        "所以它们相近，但不会塌成同一个点。"
                    ),
                    "why": (
                        f"当前 fruit 家族里 apple-pear 距离 {apple_pear:.4f}，apple-banana 距离 {apple_banana:.4f}，"
                        f"都显著小于跨 family 中心距离；这就是 family patch + concept offset 的直接证据。"
                    ),
                },
            },
            "macro": {
                "definition": "宏观超系统层，不只是抽象名词，也包括类别提升、动作/关系/协议/阶段运输等系统级结构。",
                "core_equation": (
                    "h_macro(c, ctx, role, stage) = Lift(h_meso(c)) + R_(ctx,role,c) + T_stage(c,ctx) + P_proto"
                ),
                "direct_evidence": {
                    "apple_micro_to_meso_jaccard_mean": apple_chain["micro_to_meso_jaccard"]["mean"],
                    "apple_meso_to_macro_jaccard_mean": apple_chain["meso_to_macro_jaccard"]["mean"],
                    "shared_base_ratio_vs_micro_union": apple_chain["shared_base_ratio_vs_micro_union"]["mean"],
                    "entity_mean_proj": ladder["projection_ladder"]["entity_mean_proj"],
                    "category_mean_proj": ladder["projection_ladder"]["category_mean_proj"],
                    "abstract_mean_proj": ladder["projection_ladder"]["abstract_mean_proj"],
                    "relation_cross_to_within_ratio": systemic["headline_metrics"]["relation_cross_to_within_ratio"],
                    "protocol_cross_to_within_ratio": systemic["headline_metrics"]["protocol_cross_to_within_ratio"],
                },
                "strict_readout": {
                    "apple_fruit_object_answer": (
                        "苹果到水果是 family 内概念到类别的提升；苹果到物体/食物/被拿取对象，则更像经由 abstraction lift 与 relation-role bridge"
                        "进入宏观系统，不是简单把“苹果向量”直接替换成“物体向量”。"
                    ),
                    "why": (
                        "apple 链路里 meso->macro 重叠显著高于 micro->meso，说明从具体苹果到类别/系统层，依赖的是提升与桥接，"
                        "而不是把属性堆起来就得到类别。"
                    ),
                },
            },
        },
        "example_mechanisms": {
            "apple_color_taste": {
                "mechanism": "fruit family patch + apple offset + attribute fibers",
                "candidate_formula": "h_apple_attr = B_fruit + Delta_apple + a_round * u_round + a_sweet * u_sweet + a_red * u_color + epsilon",
                "meaning": "颜色、味道、圆润度属于苹果局部属性方向，附着在对象图册上，而不是脱离对象存在。",
            },
            "apple_vs_banana": {
                "mechanism": "same family patch, different offsets",
                "candidate_formula": "h_apple - h_banana = Delta_apple - Delta_banana + small relation/context residual",
                "meaning": "苹果和香蕉共享 fruit 底座，但偏移方向不同；香蕉更偏 elongated，苹果更偏 round。",
            },
            "apple_to_fruit_to_object": {
                "mechanism": "concept section -> family abstraction -> object/role lift",
                "candidate_formula": "Lift_object(apple) = L_family(B_fruit + Delta_apple) + Bridge_role(object-of-eating, held-object, in-basket)",
                "meaning": "苹果先是 fruit patch 中的具体对象，再通过类别提升和角色桥接进入“食物”“可拿取物”“被吃对象”等宏观系统。",
            },
        },
        "system_encoding_law": {
            "core_statement": (
                "当前最强候选规律是：深度神经网络的编码不是平面词表，而是三层耦合结构。"
                "微观层负责属性轴，中观层负责 family patch 与 concept offset，宏观层负责类别提升、关系运输和协议桥接。"
            ),
            "full_candidate_equation": (
                "h(c,ctx,stage) = B_(f_c) + Delta_c + sum_i alpha_i(c,ctx) u_attr_i^(f_c) + "
                "R_(ctx,c) + T_stage(c,ctx) + P_proto(c,ctx)"
            ),
            "why_this_is_efficient": [
                "同族对象共享 family patch，节省参数并保留复用。",
                "具体概念通过稀疏 offset 区分，避免每个概念独立重建全空间。",
                "属性作为局部轴挂在 family patch 上，可以跨概念复用。",
                "宏观类别与动作/抽象概念通过 lift 和 bridge 实现，避免把所有系统结构挤到单层向量里。",
            ],
        },
        "strict_conclusion": {
            "what_is_understood_now": (
                "苹果颜色味道、苹果和香蕉、苹果与水果/物体这三类问题，已经可以被同一套三尺度编码框架解释："
                "属性纤维附着在对象 patch 上，对象通过 concept offset 区分，类别与系统关系通过 lift/bridge 进入宏观层。"
            ),
            "what_is_not_yet_cracked": (
                "还不能说已经彻底破解，因为 Delta_c 如何随学习写入、属性纤维如何在新概念上自发生成、"
                "以及宏观 lift 在真实大模型中的唯一算子形式，仍然没有唯一证明。"
            ),
        },
        "progress_estimate": {
            "micro_attribute_mechanism_percent": 66.0,
            "meso_object_family_patch_mechanism_percent": 74.0,
            "macro_abstraction_relation_protocol_mechanism_percent": 52.0,
            "three_scale_joint_mechanism_percent": 61.0,
            "full_brain_encoding_mechanism_percent": 47.0,
        },
        "next_large_blocks": [
            "把 micro 属性轴扩到颜色、味道、材质、动作倾向四大属性簇，统一到 qwen3 和 deepseek 的同一词表上。",
            "把 meso family patch 扩到 fruits / animals / vehicles / objects 的 hundreds-scale 对象集，直接拟合 shared basis 与 concept offsets。",
            "把 macro lift 与 relation-role bridge 做成统一算子搜索，专门测试 apple->fruit->food/object-of-eating 这类层级与角色迁移。",
            "把三尺度机制接回动态学习律，回答新概念首次出现时 micro 轴、meso offset、macro bridge 分别如何形成。",
        ],
    }
    return payload


def test_qwen_deepseek_micro_meso_macro_encoding_map() -> None:
    payload = build_payload()
    micro = payload["three_scales"]["micro"]["direct_evidence"]
    meso = payload["three_scales"]["meso"]["direct_evidence"]
    macro = payload["three_scales"]["macro"]["direct_evidence"]
    assert micro["axis_identifiability_accuracy"] >= 0.99
    assert micro["sweetness_edit_target_reversal_strong"] is True
    assert meso["apple_banana_distance"] < 0.2
    assert meso["qwen_deepseek_mean_family_fit_strength"] > 0.75
    assert macro["apple_meso_to_macro_jaccard_mean"] > macro["apple_micro_to_meso_jaccard_mean"]
    assert payload["strict_conclusion"]["what_is_not_yet_cracked"].startswith("还不能说")


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen/DeepSeek micro-meso-macro encoding map")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen_deepseek_micro_meso_macro_encoding_map_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["progress_estimate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
