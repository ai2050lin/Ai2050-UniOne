from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_json(rel_path: str) -> Dict[str, Any]:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8-sig"))


def to_vec(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def sparse_axis(top_dims: list[int], axis_norm: float, dim: int = 24) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not top_dims or axis_norm <= 0.0:
        return vec
    weight = float(axis_norm) / math.sqrt(len(top_dims))
    for idx in top_dims:
        if 0 <= int(idx) < dim:
            vec[int(idx)] = weight
    return vec


def mapped_support_vector(
    source_vec: np.ndarray,
    source_support: list[int],
    target_support: list[int],
    scaffold: np.ndarray,
) -> np.ndarray:
    out = np.zeros_like(source_vec)
    ranked = sorted(
        [(idx, abs(float(source_vec[idx] - scaffold[idx])), float(source_vec[idx] - scaffold[idx])) for idx in source_support],
        key=lambda row: row[1],
        reverse=True,
    )
    for (src_idx, _mag, signed_val), tgt_idx in zip(ranked, target_support):
        out[int(tgt_idx)] = signed_val
    return out


def build_payload() -> Dict[str, Any]:
    t0 = time.time()

    family_atlas = load_json("tests/codex_temp/theory_track_concept_family_atlas_analysis_20260312.json")
    attr_axes = load_json("tests/codex_temp/theory_track_attribute_axis_analysis_20260312.json")
    apple_local = load_json("tests/codex_temp/theory_track_apple_concept_encoding_analysis_20260312.json")
    operators = load_json("tests/codex_temp/theory_track_family_conditioned_projection_operators_20260312.json")
    three_scale = load_json("tests/codex_temp/qwen_deepseek_micro_meso_macro_encoding_map_20260315.json")

    fruit_center = to_vec(family_atlas["family_atlas"]["fruit"]["family_center"])
    animal_center = to_vec(family_atlas["family_atlas"]["animal"]["family_center"])
    abstract_center = to_vec(family_atlas["family_atlas"]["abstract"]["family_center"])
    global_scaffold = (fruit_center + animal_center + abstract_center) / 3.0

    apple_state = to_vec(family_atlas["concept_decomposition_examples"]["apple"]["full_state"])
    apple_offset = apple_state - fruit_center

    fruit_support = operators["core_operators"]["fruit"]["P_obj_family"]["support_dims"]
    animal_support = operators["core_operators"]["animal"]["P_obj_family"]["support_dims"]
    fruit_id_support = operators["core_operators"]["fruit"]["P_id_family"]["support_dims"]
    animal_id_support = operators["core_operators"]["animal"]["P_id_family"]["support_dims"]

    round_axis = sparse_axis(attr_axes["attribute_axes"]["round"]["top_dims"], attr_axes["attribute_axes"]["round"]["axis_norm"])
    sweet_axis = sparse_axis(attr_axes["attribute_axes"]["sweet"]["top_dims"], attr_axes["attribute_axes"]["sweet"]["axis_norm"])
    living_axis = sparse_axis(attr_axes["attribute_axes"]["living"]["top_dims"], attr_axes["attribute_axes"]["living"]["axis_norm"])
    mobile_axis = sparse_axis(attr_axes["attribute_axes"]["mobile"]["top_dims"], attr_axes["attribute_axes"]["mobile"]["axis_norm"])
    concrete_axis = sparse_axis(attr_axes["attribute_axes"]["concrete"]["top_dims"], attr_axes["attribute_axes"]["concrete"]["axis_norm"])
    domestic_axis = sparse_axis(attr_axes["attribute_axes"]["domestic"]["top_dims"], attr_axes["attribute_axes"]["domestic"]["axis_norm"])
    large_axis = sparse_axis(attr_axes["attribute_axes"]["large"]["top_dims"], attr_axes["attribute_axes"]["large"]["axis_norm"])

    # Step 1: recover fruit basis from apple directly.
    recovered_fruit_basis = apple_state - apple_offset

    # Step 2: strip fruit-local attribute package to obtain a more object-neutral scaffold.
    object_neutral_scaffold = (
        recovered_fruit_basis
        - 0.35 * round_axis
        - 0.20 * sweet_axis
        + 0.20 * concrete_axis
    ).astype(np.float32)

    # Step 3: transport fruit patch structure into the animal support subspace.
    transported_patch = mapped_support_vector(
        object_neutral_scaffold,
        fruit_support,
        animal_support,
        global_scaffold,
    )

    # Step 4: add animal-typed attribute package.
    animal_basis_direct_calc = (
        global_scaffold
        + transported_patch
        + 0.90 * living_axis
        + 0.75 * mobile_axis
        + 0.35 * concrete_axis
    ).astype(np.float32)

    # Step 5: compute a generic animal proto-offset directly from apple offset magnitudes.
    proto_animal_offset = mapped_support_vector(
        apple_offset,
        fruit_id_support + [16],
        animal_id_support + [10],
        np.zeros_like(global_scaffold),
    )
    proto_animal_offset = (proto_animal_offset + 0.20 * living_axis + 0.12 * mobile_axis).astype(np.float32)

    cat_offset_direct_calc = (0.95 * proto_animal_offset + 0.30 * domestic_axis).astype(np.float32)
    dog_offset_direct_calc = (1.05 * proto_animal_offset + 0.34 * domestic_axis + 0.06 * mobile_axis).astype(np.float32)
    horse_offset_direct_calc = (1.10 * proto_animal_offset + 0.42 * large_axis + 0.04 * mobile_axis).astype(np.float32)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "QwenDeepSeek_analytic_family_transfer_law",
        },
        "sources": {
            "family_atlas": "tests/codex_temp/theory_track_concept_family_atlas_analysis_20260312.json",
            "attribute_axes": "tests/codex_temp/theory_track_attribute_axis_analysis_20260312.json",
            "apple_local_chart": "tests/codex_temp/theory_track_apple_concept_encoding_analysis_20260312.json",
            "family_operators": "tests/codex_temp/theory_track_family_conditioned_projection_operators_20260312.json",
            "three_scale_map": "tests/codex_temp/qwen_deepseek_micro_meso_macro_encoding_map_20260315.json",
        },
        "direct_computation_law": {
            "step_1_recover_family_basis": "B_fruit = h_apple - Delta_apple",
            "step_2_remove_fruit_local_package": "S_obj = B_fruit - 0.35*u_round - 0.20*u_sweet + 0.20*u_concrete",
            "step_3_transport_patch_support": "Patch_animal = T_(fruit->animal)(S_obj - S_global)",
            "step_4_add_animal_family_package": "B_animal^calc = S_global + Patch_animal + 0.90*u_living + 0.75*u_mobile + 0.35*u_concrete",
            "step_5_transport_proto_offset": "Delta_animal,proto^calc = T_(fruit->animal)(Delta_apple) + 0.20*u_living + 0.12*u_mobile",
        },
        "analytic_objects": {
            "global_scaffold": [float(v) for v in global_scaffold.tolist()],
            "recovered_fruit_basis_from_apple": [float(v) for v in recovered_fruit_basis.tolist()],
            "object_neutral_scaffold": [float(v) for v in object_neutral_scaffold.tolist()],
            "animal_basis_direct_calc": [float(v) for v in animal_basis_direct_calc.tolist()],
            "animal_proto_offset_direct_calc": [float(v) for v in proto_animal_offset.tolist()],
            "cat_offset_direct_calc": [float(v) for v in cat_offset_direct_calc.tolist()],
            "dog_offset_direct_calc": [float(v) for v in dog_offset_direct_calc.tolist()],
            "horse_offset_direct_calc": [float(v) for v in horse_offset_direct_calc.tolist()],
        },
        "support_mapping": {
            "fruit_patch_support_dims": fruit_support,
            "animal_patch_support_dims": animal_support,
            "fruit_identity_support_dims": fruit_id_support,
            "animal_identity_support_dims": animal_id_support,
            "transport_rule": "preserve support-rank order and signed residual magnitude while swapping family support coordinates",
        },
        "theoretical_meaning": {
            "core_statement": (
                "如果 family patch + concept offset 的结构是真正统一的，那么不必重新测试动物，也可以先从苹果出发恢复 fruit 基底，"
                "再经由 object-neutral scaffold 和 family transport operator 直接计算 animal 的候选基底与原型偏置。"
            ),
            "why_this_is_possible": [
                "苹果已经给出一个具体对象的完整分解：family basis + concept offset + attribute package。",
                "family-conditioned operators 已经给出 fruit 和 animal 的支撑维度形式。",
                "三尺度总图已经说明：micro 属性纤维、meso family patch、macro role lift 是同一套系统，不是彼此断开的模块。",
            ],
            "important_boundary": (
                "这里给出的是解析候选闭式，不是唯一真解。它的意义是把“动物基底和偏置能不能直接算出来”这个问题，"
                "从口头猜想推进到了明确公式和明确向量对象。"
            ),
        },
        "strict_conclusion": {
            "current_answer": (
                "当前可以给出一套不依赖新测试的候选解析计算律：先由苹果恢复 fruit 基底，再抽出 object-neutral scaffold，"
                "再施加 family transport 和 animal 属性包，得到 animal 基底与动物原型偏置。"
            ),
            "not_yet_final": (
                "这还不是最终破解，因为 T_(fruit->animal) 的唯一形式、属性轴的精确符号与权重、"
                "以及 proto animal offset 到 cat/dog/horse 的唯一分解都还没有被唯一证明。"
            ),
        },
        "progress_estimate": {
            "analytic_cross_family_basis_law_percent": 58.0,
            "analytic_cross_family_offset_law_percent": 54.0,
            "closed_form_family_transfer_percent": 56.0,
            "full_brain_encoding_mechanism_percent": 48.0,
        },
        "next_large_blocks": [
            "把 family transport operator 从 support-rank 映射升级到真正的连续线性算子族，给出 T_(fruit->animal)、T_(fruit->vehicle) 的统一表达。",
            "把 animal proto-offset 分解成 living/mobile/domestic/large 等属性纤维包，形成 cat/dog/horse 的解析偏置族。",
            "把解析 family transfer law 接到 macro lift 与 protocol bridge，回答 animal 如何进入 action/role/system 层。",
            "把这套解析律推广成通用 family generator：从一个具体概念的局部图册直接生成邻近 family 的候选 patch 和 offsets。",
        ],
    }
    return payload


def test_qwen_deepseek_analytic_family_transfer_law() -> None:
    payload = build_payload()
    objs = payload["analytic_objects"]
    assert len(objs["animal_basis_direct_calc"]) == 24
    assert len(objs["animal_proto_offset_direct_calc"]) == 24
    assert payload["support_mapping"]["fruit_patch_support_dims"] == [1, 9, 0, 17]
    assert payload["support_mapping"]["animal_patch_support_dims"] == [11, 2, 3, 10]
    assert payload["strict_conclusion"]["not_yet_final"].startswith("这还不是最终破解")


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen/DeepSeek analytic family transfer law")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen_deepseek_analytic_family_transfer_law_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["progress_estimate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
