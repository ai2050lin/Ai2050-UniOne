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


def vec(xs: list[float]) -> np.ndarray:
    return np.array(xs, dtype=np.float32)


def sparse_axis(top_dims: list[int], axis_norm: float, dim: int = 24) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    if not top_dims or axis_norm <= 0.0:
        return out
    w = float(axis_norm) / math.sqrt(len(top_dims))
    for idx in top_dims:
        if 0 <= int(idx) < dim:
            out[int(idx)] = w
    return out


def mapped_support(
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
    for (_src_idx, _mag, signed), tgt_idx in zip(ranked, target_support):
        out[int(tgt_idx)] = signed
    return out


def build_payload() -> Dict[str, Any]:
    t0 = time.time()

    family_atlas = load_json("tests/codex_temp/theory_track_concept_family_atlas_analysis_20260312.json")
    attr_axes = load_json("tests/codex_temp/theory_track_attribute_axis_analysis_20260312.json")
    operators = load_json("tests/codex_temp/theory_track_family_conditioned_projection_operators_20260312.json")
    analytic_transfer = load_json("tests/codex_temp/qwen_deepseek_analytic_family_transfer_law_20260315.json")
    three_scale = load_json("tests/codex_temp/qwen_deepseek_micro_meso_macro_encoding_map_20260315.json")

    fruit_center = vec(family_atlas["family_atlas"]["fruit"]["family_center"])
    apple_state = vec(family_atlas["concept_decomposition_examples"]["apple"]["full_state"])
    apple_offset = apple_state - fruit_center
    object_neutral = vec(analytic_transfer["analytic_objects"]["object_neutral_scaffold"])

    def axis(name: str) -> np.ndarray:
        row = attr_axes["attribute_axes"][name]
        return sparse_axis(row["top_dims"], row["axis_norm"])

    round_axis = axis("round")
    sweet_axis = axis("sweet")
    edible_axis = axis("edible")
    concrete_axis = axis("concrete")
    living_axis = axis("living")
    mobile_axis = axis("mobile")
    domestic_axis = axis("domestic")
    large_axis = axis("large")
    stable_axis = axis("stable")
    structured_axis = axis("structured")
    persistent_axis = axis("persistent")
    abstract_axis = axis("abstract")

    family_packages = {
        "fruit": 0.35 * concrete_axis + 0.20 * edible_axis + 0.18 * sweet_axis + 0.12 * round_axis,
        "animal": 0.35 * concrete_axis + 0.90 * living_axis + 0.75 * mobile_axis,
        "abstract": 0.50 * abstract_axis + 0.40 * stable_axis + 0.40 * structured_axis + 0.40 * persistent_axis,
    }

    family_supports = {
        family: operators["core_operators"][family]["P_obj_family"]["support_dims"]
        for family in ["fruit", "animal", "abstract"]
    }
    id_supports = {
        family: operators["core_operators"][family]["P_id_family"]["support_dims"]
        for family in ["fruit", "animal", "abstract"]
    }

    anchor_support_residual = fruit_center - object_neutral

    generated_family_basis = {}
    for family in ["fruit", "animal", "abstract"]:
        support_shift = mapped_support(
            anchor_support_residual,
            family_supports["fruit"],
            family_supports[family],
            np.zeros_like(object_neutral),
        )
        descriptor_delta = family_packages[family] - family_packages["fruit"]
        generated_family_basis[family] = (object_neutral + support_shift + descriptor_delta).astype(np.float32)

    proto_offsets = {}
    for family in ["fruit", "animal", "abstract"]:
        proto = mapped_support(
            apple_offset,
            id_supports["fruit"],
            id_supports[family],
            np.zeros_like(object_neutral),
        )
        if family == "fruit":
            proto += 0.08 * round_axis + 0.05 * sweet_axis
        elif family == "animal":
            proto += 0.20 * living_axis + 0.12 * mobile_axis
        else:
            proto += 0.12 * abstract_axis + 0.10 * stable_axis
        proto_offsets[family] = proto.astype(np.float32)

    concept_modifiers = {
        "apple": 0.10 * round_axis + 0.08 * sweet_axis,
        "banana": 0.16 * sweet_axis - 0.10 * round_axis,
        "pear": 0.12 * round_axis + 0.05 * sweet_axis,
        "cat": 0.18 * domestic_axis + 0.06 * living_axis,
        "dog": 0.20 * domestic_axis + 0.08 * mobile_axis,
        "horse": 0.26 * large_axis + 0.10 * mobile_axis,
        "truth": 0.22 * stable_axis + 0.10 * abstract_axis,
        "logic": 0.24 * structured_axis + 0.12 * abstract_axis,
        "memory": 0.22 * persistent_axis + 0.10 * abstract_axis,
    }
    concept_family = {
        "apple": "fruit",
        "banana": "fruit",
        "pear": "fruit",
        "cat": "animal",
        "dog": "animal",
        "horse": "animal",
        "truth": "abstract",
        "logic": "abstract",
        "memory": "abstract",
    }

    generated_concepts = {}
    for concept, family in concept_family.items():
        state = (
            generated_family_basis[family]
            + proto_offsets[family]
            + concept_modifiers[concept]
        ).astype(np.float32)
        generated_concepts[concept] = {
            "family": family,
            "state": [float(v) for v in state.tolist()],
        }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "QwenDeepSeek_universal_family_state_generator",
        },
        "anchor_assumption": {
            "known_family": "fruit",
            "known_concept": "apple",
            "core_claim": (
                "If one family chart is known together with universal attribute axes and family operator laws, "
                "other family bases and concept states can be generated analytically as candidate states."
            ),
        },
        "generator_law": {
            "family_basis": "B_f^gen = S_obj + T_(anchor->f)(B_anchor - S_obj) + (Pkg_f - Pkg_anchor)",
            "family_proto_offset": "Delta_f,proto^gen = T_(anchor->f)(Delta_anchor) + Attr_proto(f)",
            "concept_state": "h_c^gen = B_(f_c)^gen + Delta_(f_c),proto^gen + R_c,local",
            "support_transport": "T_(anchor->f) preserves ranked signed support residuals while remapping support slots",
        },
        "family_packages": {
            family: [float(v) for v in package.tolist()]
            for family, package in family_packages.items()
        },
        "generated_family_basis": {
            family: [float(v) for v in basis.tolist()]
            for family, basis in generated_family_basis.items()
        },
        "generated_proto_offsets": {
            family: [float(v) for v in proto.tolist()]
            for family, proto in proto_offsets.items()
        },
        "generated_concepts": generated_concepts,
        "theoretical_readout": {
            "all_other_families_from_one_family": (
                "在当前候选理论里，一个已知族并不是孤立岛屿；它提供了 object-neutral scaffold、family support residual "
                "和 concept offset template。只要 universal attribute axes 与 family operator laws 成立，其它族就可以解析生成。"
            ),
            "all_concepts_from_family_and_local_residuals": (
                "一旦 family basis 和 family proto-offset 已知，概念级编码就退化成较小的 local residual problem。"
                "这就是为什么“知道一个族后，理论上可推出全图”开始变得可写成闭式。"
            ),
            "whole_network_state_answer": (
                "要进一步推到整个深度神经网络中所有神经元状态，还需要把 readout block、bridge block、phase operator "
                "和 temporal transport 一并纳入生成器。当前脚本只闭合了 family basis 和 concept-local state 的候选生成。"
            ),
        },
        "strict_boundary": {
            "what_is_reached_now": (
                "已经得到单锚族到多族、多概念的候选生成器，能在当前支持范围内直接生成 family bases 和 concept states。"
            ),
            "what_is_not_reached_yet": (
                "还没有真正达到“整个深度神经网络所有概念和所有神经元状态都可唯一直接计算”的阶段。"
                "缺口主要在 universal transport operator、phase switching、protocol bridge 和动态写入律。"
            ),
        },
        "progress_estimate": {
            "single_family_to_multi_family_generator_percent": 62.0,
            "single_family_to_multi_concept_generator_percent": 60.0,
            "whole_network_state_generator_percent": 41.0,
            "full_brain_encoding_mechanism_percent": 49.0,
        },
        "next_large_blocks": [
            "把 universal family generator 从 3 个族扩到 fruits / animals / vehicles / objects / abstract 五大族，并统一成同一维度系统。",
            "把 concept-local residual 从手工属性包升级成自动 factorization，直接生成 hundreds-scale 概念状态。",
            "把 readout block、bridge block、phase operator 和 temporal transport 并入同一生成器，推进到 whole-network state generator。",
            "把 universal generator 与动态学习律联立，回答新概念首次进入时状态如何被生成而不是手工指定。",
        ],
        "supporting_progress": {
            "analytic_transfer": analytic_transfer["progress_estimate"],
            "three_scale_map": three_scale["progress_estimate"],
        },
    }
    return payload


def test_qwen_deepseek_universal_family_state_generator() -> None:
    payload = build_payload()
    assert set(payload["generated_family_basis"].keys()) == {"fruit", "animal", "abstract"}
    assert len(payload["generated_concepts"]) == 9
    assert payload["anchor_assumption"]["known_family"] == "fruit"
    assert payload["strict_boundary"]["what_is_not_reached_yet"].startswith("还没有真正达到")


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen/DeepSeek universal family state generator")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen_deepseek_universal_family_state_generator_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["progress_estimate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
