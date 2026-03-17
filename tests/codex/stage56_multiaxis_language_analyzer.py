from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def layer_band(rows: Sequence[Dict[str, object]]) -> str:
    layers = [int(row["layer"]) for row in rows if "layer" in row]
    if not layers:
        return "unknown"
    mean_layer = sum(layers) / len(layers)
    if mean_layer < 12:
        return "early"
    if mean_layer < 24:
        return "middle"
    return "late"


def top_categories(rows: Sequence[Dict[str, object]], limit: int, positive_only: bool) -> List[str]:
    filtered = []
    for row in rows:
        if positive_only and int(row.get("strict_positive_pair_count", 0)) <= 0:
            continue
        filtered.append(row)
    ordered = sorted(
        filtered,
        key=lambda row: (
            int(row.get("strict_positive_pair_count", 0)),
            safe_float(row.get("mean_union_synergy_joint", 0.0)),
            safe_float(row.get("mean_union_joint_adv", 0.0)),
        ),
        reverse=True,
    )
    return [str(row["category"]) for row in ordered[:limit]]


def bottom_categories(rows: Sequence[Dict[str, object]], limit: int) -> List[str]:
    ordered = sorted(
        rows,
        key=lambda row: (
            safe_float(row.get("mean_union_synergy_joint", 0.0)),
            safe_float(row.get("mean_union_joint_adv", 0.0)),
        ),
    )
    return [str(row["category"]) for row in ordered[:limit]]


def aggregate_category_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        category = str(row.get("category", "unknown"))
        grouped.setdefault(category, []).append(row)

    merged: List[Dict[str, object]] = []
    for category, category_rows in grouped.items():
        pair_count = sum(int(row.get("pair_count", 0)) for row in category_rows)
        strict_positive_pair_count = sum(int(row.get("strict_positive_pair_count", 0)) for row in category_rows)
        mean_union_joint_adv = sum(safe_float(row.get("mean_union_joint_adv", 0.0)) for row in category_rows) / len(category_rows)
        mean_union_synergy_joint = sum(safe_float(row.get("mean_union_synergy_joint", 0.0)) for row in category_rows) / len(category_rows)
        merged.append(
            {
                "category": category,
                "pair_count": pair_count,
                "strict_positive_pair_count": strict_positive_pair_count,
                "strict_positive_pair_ratio": (strict_positive_pair_count / pair_count) if pair_count else 0.0,
                "mean_union_joint_adv": mean_union_joint_adv,
                "mean_union_synergy_joint": mean_union_synergy_joint,
            }
        )
    return merged


def classify_hierarchy(micro_to_meso: float, meso_to_macro: float, anchor_ratio: float) -> str:
    if meso_to_macro > micro_to_meso and anchor_ratio > 0.0:
        return "macro_bridge_dominant"
    if micro_to_meso > 0.0 and meso_to_macro > 0.0:
        return "balanced_progressive"
    if anchor_ratio <= 0.0:
        return "unanchored"
    return "micro_heavy_or_flat"


def classify_generation(signal: float, decoupling: float, specificity: float) -> str:
    if signal > 0.5 and decoupling > 0.5 and specificity > 0.5:
        return "parallel_decoupled_axes"
    if signal > 0.3 and decoupling > 0.25:
        return "partially_decoupled_axes"
    return "entangled_or_weak_axes"


def build_concept_axis(metrics: Dict[str, object]) -> Dict[str, object]:
    micro_to_meso = safe_float(metrics.get("apple_micro_to_meso_jaccard_mean"))
    meso_to_macro = safe_float(metrics.get("apple_meso_to_macro_jaccard_mean"))
    anchor_ratio = safe_float(metrics.get("apple_shared_base_ratio_mean"))
    hierarchy_gain = meso_to_macro - micro_to_meso
    hierarchy_type = classify_hierarchy(micro_to_meso, meso_to_macro, anchor_ratio)
    return {
        "micro_to_meso_jaccard_mean": micro_to_meso,
        "meso_to_macro_jaccard_mean": meso_to_macro,
        "shared_base_ratio_mean": anchor_ratio,
        "hierarchy_gain": hierarchy_gain,
        "hierarchy_type": hierarchy_type,
        "interpretation": [
            "如果 meso_to_macro 明显大于 micro_to_meso，说明实体层到超系统层的桥接强于属性层到实体层的直接闭合。",
            "shared_base_ratio 为正，说明 apple 仍然挂在 fruit 家族基底上，而不是完全漂浮的孤立点。",
        ],
    }


def build_generation_axis(metrics: Dict[str, object]) -> Dict[str, object]:
    signal = safe_float(metrics.get("style_logic_syntax_signal"))
    decoupling = safe_float(metrics.get("cross_dim_decoupling_index"))
    specificity = safe_float(metrics.get("axis_specificity_index"))
    triplet_sep = safe_float(metrics.get("triplet_separability_index"))
    generation_type = classify_generation(signal, decoupling, specificity)
    return {
        "style_logic_syntax_signal": signal,
        "cross_dim_decoupling_index": decoupling,
        "axis_specificity_index": specificity,
        "triplet_separability_index": triplet_sep,
        "generation_type": generation_type,
        "interpretation": [
            "style/logic/syntax 不是完全共线塌缩，而是并行轴上的部分解耦调制。",
            "axis_specificity 和 triplet_separability 同时为正，支持局部线性关系轴与概念轴共存。",
        ],
    }


def build_cross_model_section(
    discovery_summary: Dict[str, object],
    per_model_rows: Sequence[Dict[str, object]],
    per_category_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    aggregated_rows = aggregate_category_rows(per_category_rows)
    strongest = top_categories(aggregated_rows, limit=5, positive_only=True)
    weakest = bottom_categories(aggregated_rows, limit=4)
    model_rows = {}
    for row in per_model_rows:
        model_tag = str(row["model_tag"])
        model_rows[model_tag] = {
            "prototype_layer_band": layer_band(row.get("top_prototype_layers", [])),
            "instance_layer_band": layer_band(row.get("top_instance_layers", [])),
            "strict_positive_pair_ratio": safe_float(row.get("strict_positive_pair_ratio", 0.0)),
        }
    return {
        "strict_positive_pair_ratio": safe_float(discovery_summary.get("strict_positive_pair_ratio", 0.0)),
        "margin_zero_pair_ratio": safe_float(discovery_summary.get("margin_zero_pair_ratio", 0.0)),
        "strongest_categories": strongest,
        "weakest_categories": weakest,
        "model_layer_bands": model_rows,
        "interpretation": [
            "跨模型正协同类别存在，但稳定强共现仍集中在少数类别，说明规律存在但还没有形成全域闭合。",
            "不同模型层段分布不同，支持同一数学规律可被不同层实现，而不是逐层同构。",
        ],
    }


def find_category_rows(per_category_rows: Sequence[Dict[str, object]], category: str) -> List[Dict[str, object]]:
    return [row for row in per_category_rows if str(row.get("category")) == category]


def summarize_category(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {
            "pair_count": 0,
            "strict_positive_pair_count": 0,
            "strict_positive_pair_ratio": 0.0,
            "mean_union_joint_adv": 0.0,
            "mean_union_synergy_joint": 0.0,
        }
    pair_count = sum(int(row.get("pair_count", 0)) for row in rows)
    strict_positive = sum(int(row.get("strict_positive_pair_count", 0)) for row in rows)
    mean_joint = sum(safe_float(row.get("mean_union_joint_adv", 0.0)) for row in rows) / len(rows)
    mean_synergy = sum(safe_float(row.get("mean_union_synergy_joint", 0.0)) for row in rows) / len(rows)
    return {
        "pair_count": pair_count,
        "strict_positive_pair_count": strict_positive,
        "strict_positive_pair_ratio": (strict_positive / pair_count) if pair_count else 0.0,
        "mean_union_joint_adv": mean_joint,
        "mean_union_synergy_joint": mean_synergy,
    }


def build_joint_law(
    concept_axis: Dict[str, object],
    generation_axis: Dict[str, object],
    cross_model: Dict[str, object],
    per_category_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    fruit_summary = summarize_category(find_category_rows(per_category_rows, "fruit"))
    tech_summary = summarize_category(find_category_rows(per_category_rows, "tech"))
    animal_summary = summarize_category(find_category_rows(per_category_rows, "animal"))

    return {
        "core_equation": "h_t = B_family + Delta_micro + Delta_meso + Delta_macro + G_style + G_logic + G_syntax + R_relation",
        "coding_rules": [
            "具体概念先落在 family basis 上，再由局部 offset 决定 apple 与 banana、pear 的实例差分。",
            "Micro/Meso/Macro 更像概念内容的层级展开，不是简单的词性分类。",
            "Style/Logic/Syntax 更像生成时的并行控制轴，它们调制同一概念底座的读出方式。",
            "词嵌入类比关系说明局部线性结构存在，但真实生成还叠加了上下文约束与层级门控。",
        ],
        "apple_case": {
            "family_anchor": "apple 当前最合理地被解释为 fruit 家族基底上的局部实例 offset。",
            "micro_status": "apple 的 micro->meso 重叠仍然偏低，说明属性纤维没有完全压缩成稳定实体闭包。",
            "meso_status": "apple 的 meso 层相对稳定，具备实体家族锚点。",
            "macro_status": "apple 的 meso->macro 强于 micro->meso，说明对象一旦成形，就更容易接入动作、抽象联想和故事超系统。",
            "generation_status": "apple 在生成时不会只调用一个概念轴，而是同时接受 style/logic/syntax 的并行调制。",
        },
        "category_support": {
            "fruit": fruit_summary,
            "tech": tech_summary,
            "animal": animal_summary,
        },
        "strong_claim": {
            "statement": "深度神经网络中的语言编码更像“分层概念底座 + 并行生成控制轴”的组合系统，而不是单一连续向量场。",
            "supported": bool(
                safe_float(concept_axis["hierarchy_gain"]) > 0.0
                and safe_float(generation_axis["style_logic_syntax_signal"]) > 0.5
                and safe_float(cross_model["strict_positive_pair_ratio"]) > 0.2
            ),
        },
        "hard_limits": [
            "fruit 类目前只有边缘正协同，说明 apple 所在家族仍未达到严格强闭合。",
            "原型通道仍依赖 proxy，尚不能宣称真实类别词本体已经完全锁定。",
            "跨模型类别共现不等于跨模型同机制共现，层段差异仍然很大。",
        ],
        "next_experiment_blocks": [
            "做 apple/banana/pear 的真实类别词闭合强化块，把 fruit 从边缘正向推到稳定正协同。",
            "做 concept-axis 与 generation-axis 的交叉干预块，检查改 style 是否改变逻辑轴和语法轴的选路。",
            "做强类 tech 与弱类 fruit/animal 的对照块，判断失败是原型弱、实例弱还是联合冲突弱。",
        ],
    }


def build_payload(
    apple_dossier: Dict[str, object],
    discovery_summary: Dict[str, object],
    per_model_rows: Sequence[Dict[str, object]],
    per_category_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    metrics = dict(apple_dossier.get("metrics", {}))
    concept_axis = build_concept_axis(metrics)
    generation_axis = build_generation_axis(metrics)
    cross_model = build_cross_model_section(discovery_summary, per_model_rows, per_category_rows)
    joint_law = build_joint_law(concept_axis, generation_axis, cross_model, per_category_rows)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage56_multiaxis_language_analyzer_v1",
        "title": "多轴语言结构分析块",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "concept_axis": concept_axis,
        "generation_axis": generation_axis,
        "cross_model_support": cross_model,
        "joint_law": joint_law,
    }


def build_markdown(payload: Dict[str, object]) -> str:
    concept_axis = payload["concept_axis"]
    generation_axis = payload["generation_axis"]
    cross_model = payload["cross_model_support"]
    joint_law = payload["joint_law"]

    lines = [
        "# 多轴语言结构分析块",
        "",
        "## 核心结论",
        f"- 概念轴类型: {concept_axis['hierarchy_type']}",
        f"- 生成轴类型: {generation_axis['generation_type']}",
        f"- 跨模型严格正协同比例: {safe_float(cross_model['strict_positive_pair_ratio']):.6f}",
        f"- 最强类别: {', '.join(cross_model['strongest_categories'])}",
        f"- 最弱类别: {', '.join(cross_model['weakest_categories'])}",
        "",
        "## Apple 结构",
        f"- micro->meso: {safe_float(concept_axis['micro_to_meso_jaccard_mean']):.6f}",
        f"- meso->macro: {safe_float(concept_axis['meso_to_macro_jaccard_mean']):.6f}",
        f"- shared_base_ratio: {safe_float(concept_axis['shared_base_ratio_mean']):.6f}",
        f"- hierarchy_gain: {safe_float(concept_axis['hierarchy_gain']):.6f}",
        "",
        "## 生成控制轴",
        f"- style_logic_syntax_signal: {safe_float(generation_axis['style_logic_syntax_signal']):.6f}",
        f"- cross_dim_decoupling_index: {safe_float(generation_axis['cross_dim_decoupling_index']):.6f}",
        f"- axis_specificity_index: {safe_float(generation_axis['axis_specificity_index']):.6f}",
        f"- triplet_separability_index: {safe_float(generation_axis['triplet_separability_index']):.6f}",
        "",
        "## 统一编码定律",
        f"- 公式: `{joint_law['core_equation']}`",
    ]
    lines.extend([f"- {rule}" for rule in joint_law["coding_rules"]])
    lines.extend(["", "## 硬伤"])
    lines.extend([f"- {line}" for line in joint_law["hard_limits"]])
    lines.extend(["", "## 下一步大块"])
    lines.extend([f"- {line}" for line in joint_law["next_experiment_blocks"]])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a multiaxis language analyzer from apple and stage56 outputs")
    parser.add_argument(
        "--apple-dossier-json",
        default="tempdata/deepseek7b_apple_encoding_law_dossier_20260306_223055/apple_multiaxis_encoding_law_dossier.json",
    )
    parser.add_argument(
        "--discovery-summary-json",
        default="tempdata/stage56_large_scale_discovery_multimodel_20260317_2105/discovery_summary.json",
    )
    parser.add_argument(
        "--discovery-per-model-jsonl",
        default="tempdata/stage56_large_scale_discovery_multimodel_20260317_2105/discovery_per_model.jsonl",
    )
    parser.add_argument(
        "--discovery-per-category-jsonl",
        default="tempdata/stage56_large_scale_discovery_multimodel_20260317_2105/discovery_per_category.jsonl",
    )
    parser.add_argument("--output-dir", default="tests/codex_temp/stage56_multiaxis_language_analyzer_20260317")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apple_dossier = read_json(Path(args.apple_dossier_json))
    discovery_summary = read_json(Path(args.discovery_summary_json))
    per_model_rows = read_jsonl(Path(args.discovery_per_model_jsonl))
    per_category_rows = read_jsonl(Path(args.discovery_per_category_jsonl))

    payload = build_payload(apple_dossier, discovery_summary, per_model_rows, per_category_rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "multiaxis_language_analysis.json"
    report_path = out_dir / "MULTIAXIS_LANGUAGE_ANALYSIS_REPORT.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_markdown(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "json": str(json_path),
                "report": str(report_path),
                "headline": {
                    "concept_axis": payload["concept_axis"]["hierarchy_type"],
                    "generation_axis": payload["generation_axis"]["generation_type"],
                    "strong_claim_supported": payload["joint_law"]["strong_claim"]["supported"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
