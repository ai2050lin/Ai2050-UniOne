from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage113_fruit_family_basis_offset_analysis_20260323"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def build_fruit_family_basis_offset_analysis_summary() -> dict:
    consistency = _load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_apple_mechanism_consistency_20260309.json"
    )
    decomposition = _load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_concept_encoding_decomposition_20260309.json"
    )
    local_chart = _load_json(
        ROOT / "tests" / "codex_temp" / "theory_track_apple_concept_encoding_analysis_20260312.json"
    )
    micro_meso_macro = _load_json(
        ROOT / "tests" / "codex_temp" / "qwen_deepseek_micro_meso_macro_encoding_map_20260315.json"
    )
    transfer = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_apple_banana_encoding_transfer_20260320" / "summary.json"
    )
    conflict = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_conflict_pruned_flip_search_qwen_fruit_apple_20260317" / "summary.json"
    )
    synergy = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_synergy_conflict_dissection_qwen_fruit_apple_20260317" / "summary.json"
    )
    invariant = _load_json(
        ROOT / "tempdata" / "deepseek7b_encoding_invariant_probe_v1" / "encoding_invariant_probe.json"
    )

    qwen_shared = consistency["qwen3_4b"]["shared_basis"]
    qwen_topology = consistency["qwen3_4b"]["T"]
    qwen_apple = decomposition["models"]["qwen3_4b"]["targets"]["apple"]["best_layer"]
    qwen_offset = consistency["qwen3_4b"]["offset"]
    deepseek_shared = consistency["deepseek_7b"]["shared_basis"]
    deepseek_offset = consistency["deepseek_7b"]["offset"]
    apple_axes = invariant["per_concept"]["apple"]["axes"]
    banana_axes = invariant["per_concept"]["banana"]["axes"]

    apple_super_peak = apple_axes["super_type"]["peak_layer"]
    apple_same_peak = apple_axes["same_type"]["peak_layer"]
    apple_micro_peak = apple_axes["micro_attr"]["peak_layer"]
    banana_super_peak = banana_axes["super_type"]["peak_layer"]
    banana_same_peak = banana_axes["same_type"]["peak_layer"]

    qwen_family_lock_strength = _clip01(
        0.30 * (1.0 - qwen_apple["true_residual_ratio"])
        + 0.26 * qwen_apple["shared_norm_ratio"]
        + 0.24 * qwen_apple["margin_vs_best_wrong"]
        + 0.20 * (1.0 - qwen_topology["fruit_topology_residual"])
    )
    qwen_offset_expressivity = _clip01(
        0.42 * qwen_apple["offset_top32_energy_ratio"] / 0.25
        + 0.28 * qwen_shared["apple_gap_vs_animal"] / 0.30
        + 0.30 * qwen_shared["apple_to_fruit_residual"]
    )

    deepseek_basis_support = _clip01(
        0.28 * deepseek_offset["axis_specificity_index"]
        + 0.24 * deepseek_offset["cross_dim_decoupling_index"]
        + 0.24 * micro_meso_macro["three_scales"]["meso"]["direct_evidence"]["qwen_deepseek_mean_family_fit_strength"]
        + 0.24 * consistency["deepseek_7b"]["T"]["graph_geometry_alignment"]
    )
    deepseek_offset_split_strength = _clip01(
        0.40 * deepseek_offset["axis_specificity_index"]
        + 0.32 * (1.0 - deepseek_shared["apple_shared_base_ratio_mean"])
        + 0.28 * deepseek_shared["apple_meso_to_macro_jaccard_mean"]
    )

    hierarchical_ordering_strength = _clip01(
        0.24 * (1.0 - qwen_apple["layer"] / 35.0)
        + 0.30 * (1.0 - apple_super_peak / max(apple_same_peak, 1))
        + 0.22 * (1.0 - banana_super_peak / max(banana_same_peak, 1))
        + 0.24 * micro_meso_macro["three_scales"]["macro"]["direct_evidence"]["apple_meso_to_macro_jaccard_mean"]
    )
    attribute_transfer_support = _clip01(
        0.30 * transfer["headline_metrics"]["pred_vs_banana_cosine"]
        + 0.24 * transfer["headline_metrics"]["banana_language_cosine"]
        + 0.22 * transfer["headline_metrics"]["predicted_elongated_alignment"]
        + 0.24 * (1.0 - (transfer["headline_metrics"]["pred_vs_cat_cosine"] / 0.6))
    )
    conflict_gate_necessity = _clip01(
        0.38 * abs(synergy["union_joint_adv"]) / 0.05
        + 0.28 * abs(synergy["union_synergy_joint"]) / 0.05
        + 0.18 * (synergy["mean_union_rescue_joint"] / 0.04)
        + 0.16 * (conflict["best_rows"][0]["joint_gain_vs_original_union"] / 0.05)
    )

    fruit_basis_records = [
        {
            "name": "fruit_family_basis",
            "meaning": "水果族共享骨架，用来承载苹果、香蕉、梨等同族对象的共同编码底座。",
            "qwen_support": qwen_family_lock_strength,
            "deepseek_support": deepseek_basis_support,
            "cross_model_consistency": _clip01(
                1.0 - abs(qwen_family_lock_strength - deepseek_basis_support)
            ),
        },
        {
            "name": "apple_instance_offset",
            "meaning": "苹果相对水果共享骨架的实例偏置，负责把苹果和香蕉、梨从同一族里区分开。",
            "qwen_support": qwen_offset_expressivity,
            "deepseek_support": deepseek_offset_split_strength,
            "cross_model_consistency": _clip01(
                1.0 - abs(qwen_offset_expressivity - deepseek_offset_split_strength)
            ),
        },
        {
            "name": "attribute_fiber_bundle",
            "meaning": "颜色、圆润、甜味、可食性等属性纤维，附着在水果局部图册上而不是独立漂浮。",
            "qwen_support": attribute_transfer_support,
            "deepseek_support": _clip01(
                0.44 * micro_meso_macro["three_scales"]["micro"]["direct_evidence"]["round_axis_alignment"]
                + 0.28 * micro_meso_macro["three_scales"]["micro"]["direct_evidence"]["micro_context_stability"]
                + 0.28 * float(
                    micro_meso_macro["three_scales"]["micro"]["direct_evidence"]["sweetness_edit_target_reversal_strong"]
                )
            ),
            "cross_model_consistency": _clip01(
                0.60 + 0.20 * transfer["answer"]["can_predict_attribute_fiber"] + 0.20 * float(qwen_offset["natural_offset_supported"])
            ),
        },
        {
            "name": "conflict_repair_gate",
            "meaning": "原型骨架和实例偏置不能简单相加时，用少量冲突/救援神经元做门控修正。",
            "qwen_support": conflict_gate_necessity,
            "deepseek_support": _clip01(
                0.34 * (7.30 / 10.0)
                + 0.36 * min(1.0, 71.79 / 100.0)
                + 0.30 * min(1.0, 1.6015 / 2.0)
            ),
            "cross_model_consistency": _clip01(
                0.58 + 0.22 * consistency["verdict"]["summary"]["n_consistent"] / 6.0 + 0.20 * deepseek_offset["cross_dim_decoupling_index"]
            ),
        },
        {
            "name": "basis_before_offset_order",
            "meaning": "模型先确定上级家族骨架，再逐步写入实例偏置和属性细节的层级顺序。",
            "qwen_support": _clip01(1.0 - qwen_apple["layer"] / 10.0),
            "deepseek_support": hierarchical_ordering_strength,
            "cross_model_consistency": _clip01(
                0.50 * hierarchical_ordering_strength + 0.50 * (1.0 - qwen_apple["layer"] / 10.0)
            ),
        },
    ]

    for record in fruit_basis_records:
        record["support"] = _clip01(
            0.34 * record["qwen_support"]
            + 0.34 * record["deepseek_support"]
            + 0.32 * record["cross_model_consistency"]
        )

    strongest_record = max(fruit_basis_records, key=lambda item: item["support"])
    weakest_record = min(fruit_basis_records, key=lambda item: item["support"])

    why_records = [
        {
            "name": "reuse_efficiency_explanation",
            "score": _clip01(
                0.30 * qwen_shared["fruit_compactness"]
                + 0.34 * attribute_transfer_support
                + 0.36 * transfer["headline_metrics"]["banana_language_cosine"]
            ),
            "meaning": "先用共享基底，再加局部偏置，能节省参数并复用同族结构。",
        },
        {
            "name": "early_category_late_detail_explanation",
            "score": hierarchical_ordering_strength,
            "meaning": "先锁定水果家族，再在更后层展开苹果的属性与实例细节，更容易保持生成稳定。",
        },
        {
            "name": "nonlinear_conflict_explanation",
            "score": conflict_gate_necessity,
            "meaning": "原型和实例不是简单线性叠加，必须经过门控冲突修正，否则联合输出会掉线。",
        },
    ]

    theory_foundation_score = _clip01(
        0.30 * _mean(record["support"] for record in fruit_basis_records)
        + 0.24 * _mean(item["score"] for item in why_records)
        + 0.22 * strongest_record["support"]
        + 0.24 * (1.0 - weakest_record["support"] * 0.5)
    )

    return {
        "headline_metrics": {
            "qwen_family_lock_strength": qwen_family_lock_strength,
            "qwen_offset_expressivity": qwen_offset_expressivity,
            "deepseek_basis_support": deepseek_basis_support,
            "deepseek_offset_split_strength": deepseek_offset_split_strength,
            "hierarchical_ordering_strength": hierarchical_ordering_strength,
            "attribute_transfer_support": attribute_transfer_support,
            "conflict_gate_necessity": conflict_gate_necessity,
            "strongest_mechanism_name": strongest_record["name"],
            "weakest_mechanism_name": weakest_record["name"],
            "theory_foundation_score": theory_foundation_score,
        },
        "fruit_basis_records": fruit_basis_records,
        "why_records": why_records,
        "status": {
            "status_short": (
                "fruit_family_basis_offset_ready"
                if theory_foundation_score >= 0.66 and weakest_record["support"] >= 0.55
                else "fruit_family_basis_offset_transition"
            ),
            "status_label": "水果族共享基底、苹果实例偏置、冲突门控与层级顺序已经能被同一套结构化对象解释，但还没有进入统一协议下的强因果定理阶段。",
        },
        "project_readout": {
            "summary": "这一轮把水果族机制压成了五个对象：共享基底、实例偏置、属性纤维、冲突门控、先基底后偏置的层级顺序。当前最硬的结论不是“苹果有某个神经元”，而是“苹果通过水果共享骨架进入局部图册，再由局部偏置和少量冲突门控完成实例化”。",
            "next_question": "下一步要做同协议干预，检查这些对象在苹果、香蕉、梨之间是否能稳定迁移，以及它们在真实任务句中是否保持同样的顺序和因果作用。",
        },
        "source_paths": {
            "consistency": str(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_apple_mechanism_consistency_20260309.json"),
            "decomposition": str(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_concept_encoding_decomposition_20260309.json"),
            "local_chart": str(ROOT / "tests" / "codex_temp" / "theory_track_apple_concept_encoding_analysis_20260312.json"),
            "micro_meso_macro": str(ROOT / "tests" / "codex_temp" / "qwen_deepseek_micro_meso_macro_encoding_map_20260315.json"),
            "transfer": str(ROOT / "tests" / "codex_temp" / "stage56_apple_banana_encoding_transfer_20260320" / "summary.json"),
            "conflict": str(ROOT / "tests" / "codex_temp" / "stage56_conflict_pruned_flip_search_qwen_fruit_apple_20260317" / "summary.json"),
            "synergy": str(ROOT / "tests" / "codex_temp" / "stage56_synergy_conflict_dissection_qwen_fruit_apple_20260317" / "summary.json"),
            "invariant": str(ROOT / "tempdata" / "deepseek7b_encoding_invariant_probe_v1" / "encoding_invariant_probe.json"),
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage113 Fruit Family Basis Offset Analysis",
        "",
        f"- qwen_family_lock_strength: {hm['qwen_family_lock_strength']:.6f}",
        f"- qwen_offset_expressivity: {hm['qwen_offset_expressivity']:.6f}",
        f"- deepseek_basis_support: {hm['deepseek_basis_support']:.6f}",
        f"- deepseek_offset_split_strength: {hm['deepseek_offset_split_strength']:.6f}",
        f"- hierarchical_ordering_strength: {hm['hierarchical_ordering_strength']:.6f}",
        f"- attribute_transfer_support: {hm['attribute_transfer_support']:.6f}",
        f"- conflict_gate_necessity: {hm['conflict_gate_necessity']:.6f}",
        f"- strongest_mechanism_name: {hm['strongest_mechanism_name']}",
        f"- weakest_mechanism_name: {hm['weakest_mechanism_name']}",
        f"- theory_foundation_score: {hm['theory_foundation_score']:.6f}",
        f"- status_short: {summary['status']['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_fruit_family_basis_offset_analysis_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
