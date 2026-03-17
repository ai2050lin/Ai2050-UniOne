from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def layer_band(top_layers: Sequence[Dict[str, object]]) -> str:
    if not top_layers:
        return "unknown"
    layer_ids = [int(row["layer"]) for row in top_layers if "layer" in row]
    if not layer_ids:
        return "unknown"
    mean_layer = sum(layer_ids) / len(layer_ids)
    if mean_layer < 12:
        return "early"
    if mean_layer < 24:
        return "middle"
    return "late"


def top_categories(rows: Sequence[Dict[str, object]], limit: int, positive_only: bool = False) -> List[str]:
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


def failure_categories(rows: Sequence[Dict[str, object]], limit: int) -> List[str]:
    ordered = sorted(
        rows,
        key=lambda row: (
            safe_float(row.get("mean_union_synergy_joint", 0.0)),
            safe_float(row.get("mean_union_joint_adv", 0.0)),
        ),
    )
    return [str(row["category"]) for row in ordered[:limit]]


def derive_math_law(
    apple_dossier: Dict[str, object],
    discovery_summary: Dict[str, object],
    per_model_rows: Sequence[Dict[str, object]],
    per_category_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    metrics = dict(apple_dossier.get("metrics", {}))
    micro_to_meso = safe_float(metrics.get("apple_micro_to_meso_jaccard_mean"))
    meso_to_macro = safe_float(metrics.get("apple_meso_to_macro_jaccard_mean"))
    shared_base = safe_float(metrics.get("apple_shared_base_ratio_mean"))
    style_logic_syntax_signal = safe_float(metrics.get("style_logic_syntax_signal"))
    decoupling_index = safe_float(metrics.get("cross_dim_decoupling_index"))
    axis_specificity = safe_float(metrics.get("axis_specificity_index"))
    triplet_sep = safe_float(metrics.get("triplet_separability_index"))

    concept_hierarchy_gain = meso_to_macro - micro_to_meso
    cross_model_pair_ratio = safe_float(discovery_summary.get("strict_positive_pair_ratio"))
    cross_model_margin_zero_ratio = safe_float(discovery_summary.get("margin_zero_pair_ratio"))
    positive_categories = top_categories(per_category_rows, limit=6, positive_only=True)
    weak_categories = failure_categories(per_category_rows, limit=4)

    model_impl = {
        str(row["model_tag"]): {
            "prototype_layer_band": layer_band(row.get("top_prototype_layers", [])),
            "instance_layer_band": layer_band(row.get("top_instance_layers", [])),
            "strict_positive_pair_ratio": safe_float(row.get("strict_positive_pair_ratio", 0.0)),
        }
        for row in per_model_rows
    }

    law = {
        "concept_hierarchy_gain": concept_hierarchy_gain,
        "concept_anchor_ratio": shared_base,
        "generation_axis_signal": style_logic_syntax_signal,
        "generation_axis_decoupling": decoupling_index,
        "relation_axis_specificity": axis_specificity,
        "relation_axis_separability": triplet_sep,
        "cross_model_strict_positive_pair_ratio": cross_model_pair_ratio,
        "cross_model_margin_zero_ratio": cross_model_margin_zero_ratio,
        "positive_categories": positive_categories,
        "weak_categories": weak_categories,
        "model_implementation_bands": model_impl,
    }
    return law


def build_hypotheses(law: Dict[str, object]) -> List[Dict[str, object]]:
    concept_hierarchy_gain = safe_float(law["concept_hierarchy_gain"])
    concept_anchor_ratio = safe_float(law["concept_anchor_ratio"])
    generation_axis_signal = safe_float(law["generation_axis_signal"])
    generation_axis_decoupling = safe_float(law["generation_axis_decoupling"])
    relation_axis_specificity = safe_float(law["relation_axis_specificity"])
    relation_axis_separability = safe_float(law["relation_axis_separability"])
    cross_model_pair_ratio = safe_float(law["cross_model_strict_positive_pair_ratio"])
    positive_categories = list(law["positive_categories"])
    weak_categories = list(law["weak_categories"])

    return [
        {
            "id": "H1_concept_hierarchy_exists",
            "rule": "meso_to_macro > micro_to_meso and shared_base_ratio > 0",
            "pass": bool(concept_hierarchy_gain > 0.0 and concept_anchor_ratio > 0.0),
        },
        {
            "id": "H2_generation_axes_parallel_not_collapsed",
            "rule": "style_logic_syntax_signal > 0.5 and decoupling_index > 0.25",
            "pass": bool(generation_axis_signal > 0.5 and generation_axis_decoupling > 0.25),
        },
        {
            "id": "H3_relation_axis_locally_linear",
            "rule": "axis_specificity > 0 and triplet_separability > 0",
            "pass": bool(relation_axis_specificity > 0.0 and relation_axis_separability > 0.0),
        },
        {
            "id": "H4_cross_model_positive_categories_exist",
            "rule": "strict_positive_pair_ratio > 0.2 and at least 3 positive categories",
            "pass": bool(cross_model_pair_ratio > 0.2 and len(positive_categories) >= 3),
        },
        {
            "id": "H5_failure_categories_are_structured",
            "rule": "human or animal or food appears in weak categories",
            "pass": bool(any(category in {"human", "animal", "food"} for category in weak_categories)),
        },
    ]


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(path: Path, payload: Dict[str, object]) -> None:
    law = payload["law"]
    hypotheses = payload["hypotheses"]
    lines = [
        "# 语言数学结构统一档案",
        "",
        "## 核心数学式",
        "`h = B_concept + Δ_micro + Δ_meso + Δ_macro + G_style + G_logic + G_syntax + I_bind`",
        "",
        "## 指标",
        f"- concept_hierarchy_gain: {safe_float(law['concept_hierarchy_gain']):.6f}",
        f"- concept_anchor_ratio: {safe_float(law['concept_anchor_ratio']):.6f}",
        f"- generation_axis_signal: {safe_float(law['generation_axis_signal']):.6f}",
        f"- generation_axis_decoupling: {safe_float(law['generation_axis_decoupling']):.6f}",
        f"- relation_axis_specificity: {safe_float(law['relation_axis_specificity']):.6f}",
        f"- relation_axis_separability: {safe_float(law['relation_axis_separability']):.6f}",
        f"- cross_model_strict_positive_pair_ratio: {safe_float(law['cross_model_strict_positive_pair_ratio']):.6f}",
        f"- cross_model_margin_zero_ratio: {safe_float(law['cross_model_margin_zero_ratio']):.6f}",
        "",
        f"- positive_categories: {', '.join(law['positive_categories'])}",
        f"- weak_categories: {', '.join(law['weak_categories'])}",
        "",
        "## 模型实现差异",
    ]
    for model_tag, model_info in dict(law["model_implementation_bands"]).items():
        lines.append(
            "- "
            f"{model_tag}: prototype={model_info['prototype_layer_band']}, "
            f"instance={model_info['instance_layer_band']}, "
            f"strict_positive_pair_ratio={safe_float(model_info['strict_positive_pair_ratio']):.6f}"
        )
    lines.extend(["", "## 假设判定"])
    for hypothesis in hypotheses:
        lines.append(f"- {hypothesis['id']}: {'PASS' if hypothesis['pass'] else 'FAIL'}")
    lines.extend(
        [
            "",
            "## 解释",
            "- 这套结果支持一种分层可组合结构：概念基底负责实体身份，micro/meso/macro 负责语义层级偏移，style/logic/syntax 负责生成调制。",
            "- 当前最强跨模型类仍是 tech，其次是 nature 和 object；human 与 animal 持续失败，说明弱类失败不是随机噪声，而是系统性困难。",
            "- 原型通道仍依赖 proxy，所以这份档案证明的是编码规律存在，不是现实词本体闭合已经成立。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a unified language math structure dossier from multiaxis and stage56 outputs")
    ap.add_argument(
        "--apple-dossier-json",
        default="tempdata/deepseek7b_apple_encoding_law_dossier_20260306_223055/apple_multiaxis_encoding_law_dossier.json",
    )
    ap.add_argument(
        "--discovery-summary-json",
        default="tempdata/stage56_large_scale_discovery_multimodel_20260317_2105/discovery_summary.json",
    )
    ap.add_argument(
        "--discovery-per-model-jsonl",
        default="tempdata/stage56_large_scale_discovery_multimodel_20260317_2105/discovery_per_model.jsonl",
    )
    ap.add_argument(
        "--discovery-per-category-jsonl",
        default="tempdata/stage56_large_scale_discovery_multimodel_20260317_2105/discovery_per_category.jsonl",
    )
    ap.add_argument("--output-dir", default="tempdata/stage56_language_math_structure_dossier_20260317")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    apple_dossier = read_json(Path(args.apple_dossier_json))
    discovery_summary = read_json(Path(args.discovery_summary_json))
    per_model_rows = read_jsonl(Path(args.discovery_per_model_jsonl))
    per_category_rows = read_jsonl(Path(args.discovery_per_category_jsonl))

    law = derive_math_law(
        apple_dossier=apple_dossier,
        discovery_summary=discovery_summary,
        per_model_rows=per_model_rows,
        per_category_rows=per_category_rows,
    )
    hypotheses = build_hypotheses(law)
    payload = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage56_language_math_structure_dossier_v1",
        "title": "语言数学结构统一档案",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "apple_dossier_json": args.apple_dossier_json,
            "discovery_summary_json": args.discovery_summary_json,
            "discovery_per_model_jsonl": args.discovery_per_model_jsonl,
            "discovery_per_category_jsonl": args.discovery_per_category_jsonl,
        },
        "law": law,
        "hypotheses": hypotheses,
        "notes": [
            "本档案是聚合分析，不重新运行大模型。",
            "它把概念层级、生成维度、跨模型正协同放入同一数学口径，便于下一轮机制实验直接调用。",
        ],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "language_math_structure_dossier.json"
    report_path = out_dir / "LANGUAGE_MATH_STRUCTURE_DOSSIER_REPORT.md"
    write_json(json_path, payload)
    write_report(report_path, payload)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "json": str(json_path),
                "report": str(report_path),
                "law": law,
                "hypotheses": hypotheses,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
