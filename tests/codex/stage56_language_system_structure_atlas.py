from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]


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


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def aggregate_category_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        category = str(row.get("category", "unknown"))
        grouped.setdefault(category, []).append(row)

    merged: List[Dict[str, object]] = []
    for category, bucket in sorted(grouped.items()):
        pair_count = sum(int(row.get("pair_count", 0)) for row in bucket)
        strict_positive_pair_count = sum(int(row.get("strict_positive_pair_count", 0)) for row in bucket)
        merged.append(
            {
                "category": category,
                "model_count": len(bucket),
                "pair_count": pair_count,
                "strict_positive_pair_count": strict_positive_pair_count,
                "strict_positive_pair_ratio": (strict_positive_pair_count / pair_count) if pair_count else 0.0,
                "mean_union_joint_adv": average([safe_float(row.get("mean_union_joint_adv", 0.0)) for row in bucket]),
                "mean_union_synergy_joint": average([safe_float(row.get("mean_union_synergy_joint", 0.0)) for row in bucket]),
            }
        )
    return merged


def find_category(aggregated_rows: Sequence[Dict[str, object]], category: str) -> Dict[str, object]:
    for row in aggregated_rows:
        if str(row["category"]) == category:
            return row
    return {
        "category": category,
        "model_count": 0,
        "pair_count": 0,
        "strict_positive_pair_count": 0,
        "strict_positive_pair_ratio": 0.0,
        "mean_union_joint_adv": 0.0,
        "mean_union_synergy_joint": 0.0,
    }


def layer_count(role_row: Dict[str, object]) -> int:
    return len(dict(role_row.get("layer_distribution", {})))


def build_modifier_structure(micro_json: Dict[str, object]) -> Dict[str, object]:
    apple = dict(micro_json.get("concepts", {})).get("apple", {})
    role_subsets = dict(apple.get("role_subsets", {}))
    entity = dict(role_subsets.get("entity", {}))
    size = dict(role_subsets.get("size", {}))
    weight = dict(role_subsets.get("weight", {}))
    fruit = dict(role_subsets.get("fruit", {}))

    adjective_roles = {
        "size": {
            "subset_size": int(size.get("size", 0)),
            "layer_count": layer_count(size),
            "drop_target": safe_float(size.get("drop_target", 0.0)),
        },
        "weight": {
            "subset_size": int(weight.get("size", 0)),
            "layer_count": layer_count(weight),
            "drop_target": safe_float(weight.get("drop_target", 0.0)),
        },
    }
    noun_anchor_roles = {
        "entity": {
            "subset_size": int(entity.get("size", 0)),
            "layer_count": layer_count(entity),
            "drop_target": safe_float(entity.get("drop_target", 0.0)),
        },
        "fruit": {
            "subset_size": int(fruit.get("size", 0)),
            "layer_count": layer_count(fruit),
            "drop_target": safe_float(fruit.get("drop_target", 0.0)),
        },
    }

    adjective_layer_spread = average([row["layer_count"] for row in adjective_roles.values()])
    noun_layer_spread = average([row["layer_count"] for row in noun_anchor_roles.values()])

    return {
        "evidence_level": "direct",
        "claim": "形容词型微观属性更像附着在实体锚点周围的修饰纤维，而不是独立实体锚点。",
        "adjective_roles": adjective_roles,
        "noun_anchor_roles": noun_anchor_roles,
        "adjective_layer_spread_mean": adjective_layer_spread,
        "noun_anchor_layer_spread_mean": noun_layer_spread,
        "interpretation": [
            "apple 的 size 和 weight 子属性分散在更多层，说明修饰信息是跨层展开的纤维。",
            "entity 与 fruit 角色更紧，说明名词本体更像锚点，而不是纯修饰片段。",
        ],
    }


def build_concept_anchor_structure(
    apple_dossier: Dict[str, object],
    concept_family: Dict[str, object],
    aggregated_categories: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    metrics = dict(apple_dossier.get("metrics", {}))
    family_metrics = dict(concept_family.get("metrics", {}))
    fruit_row = find_category(aggregated_categories, "fruit")
    object_row = find_category(aggregated_categories, "object")
    animal_row = find_category(aggregated_categories, "animal")

    return {
        "evidence_level": "direct",
        "claim": "普通名词和具体概念更像家族锚点上的局部实例偏移，而不是孤立向量点。",
        "metrics": {
            "apple_micro_to_meso_jaccard_mean": safe_float(metrics.get("apple_micro_to_meso_jaccard_mean")),
            "apple_meso_to_macro_jaccard_mean": safe_float(metrics.get("apple_meso_to_macro_jaccard_mean")),
            "apple_shared_base_ratio_mean": safe_float(metrics.get("apple_shared_base_ratio_mean")),
            "apple_vs_cat_shared_base_gap_mean": safe_float(family_metrics.get("apple_vs_cat_shared_base_gap_mean")),
        },
        "category_support": {
            "fruit": fruit_row,
            "object": object_row,
            "animal": animal_row,
        },
        "interpretation": [
            "apple 的 meso->macro 明显强于 micro->meso，说明中观实体锚点比微观属性更稳定。",
            "fruit 和 object 在当前 stage6 里仍比 animal 更接近正向联合优势，支持“实体锚点”读法。",
        ],
    }


def build_relation_axis_structure(triplet_json: Dict[str, object]) -> Dict[str, object]:
    metrics = dict(triplet_json.get("metrics", {}))
    return {
        "evidence_level": "direct",
        "claim": "king / queen 这类关系不是全局线性空间，而是局部关系轴上的可迁移偏移。",
        "metrics": {
            "king_queen_jaccard": safe_float(metrics.get("king_queen_jaccard")),
            "apple_king_jaccard": safe_float(metrics.get("apple_king_jaccard")),
            "axis_specificity_index": safe_float(metrics.get("axis_specificity_index")),
            "triplet_separability_index": safe_float(metrics.get("triplet_separability_index")),
            "king_axis_projection_abs": safe_float(metrics.get("king_axis_projection_abs")),
            "queen_axis_projection_abs": safe_float(metrics.get("queen_axis_projection_abs")),
            "apple_axis_projection_abs": safe_float(metrics.get("apple_axis_projection_abs")),
        },
        "interpretation": [
            "更常见、也更合理的局部类比写法是 king - man + woman ≈ queen。",
            "当前证据支持局部关系轴存在，但不支持所有词都落在同一个全局线性代数里。",
        ],
    }


def compute_multidim_signal_fallback(multidim_json: Dict[str, object]) -> Dict[str, float]:
    cross_dimension = dict(multidim_json.get("cross_dimension", {}))
    specificity = dict(multidim_json.get("specificity", {}))
    layer_corrs = [safe_float(row.get("layer_profile_corr", 0.0)) for row in cross_dimension.values()]
    neuron_jaccards = [safe_float(row.get("top_neuron_jaccard", 0.0)) for row in cross_dimension.values()]
    specificity_margins = [safe_float(row.get("specificity_margin", 0.0)) for row in specificity.values()]
    return {
        "style_logic_syntax_signal": average(specificity_margins),
        "cross_dim_decoupling_index": 1.0 - average(neuron_jaccards),
        "cross_dim_mean_layer_profile_corr": average(layer_corrs),
    }


def build_generation_control_structure(
    apple_dossier: Dict[str, object],
    multidim_json: Dict[str, object],
    gate_category_link: Dict[str, object],
    gate_pair_link: Dict[str, object],
) -> Dict[str, object]:
    dimensions = dict(multidim_json.get("dimensions", {}))
    apple_metrics = dict(apple_dossier.get("metrics", {}))
    fallback = compute_multidim_signal_fallback(multidim_json)
    pair_stats = []
    for axis_name in ("style", "logic", "syntax"):
        axis_row = dict(dimensions.get(axis_name, {}))
        pair_stats.append(
            {
                "axis": axis_name,
                "mean_pair_delta_l2": safe_float(axis_row.get("mean_pair_delta_l2")),
                "pair_delta_cosine_mean": safe_float(axis_row.get("pair_delta_cosine_mean")),
            }
        )

    category_axis = dict(gate_category_link.get("axis_target_stats", {}))
    pair_axis = dict(gate_pair_link.get("axis_target_stats", {}))

    return {
        "evidence_level": "direct",
        "claim": "style / logic / syntax 不是概念本体，而是并行控制轴；它们决定概念在生成时怎样被读出、约束和闭包。",
        "multidim_probe": {
            "style_logic_syntax_signal": safe_float(
                apple_metrics.get("style_logic_syntax_signal", fallback["style_logic_syntax_signal"])
            ),
            "cross_dim_decoupling_index": safe_float(
                apple_metrics.get("cross_dim_decoupling_index", fallback["cross_dim_decoupling_index"])
            ),
            "cross_dim_mean_layer_profile_corr": safe_float(
                apple_metrics.get("cross_dim_mean_layer_profile_corr", fallback["cross_dim_mean_layer_profile_corr"])
            ),
            "pair_stats": pair_stats,
        },
        "closure_link": {
            "category_level_top": gate_category_link.get("top_findings", {}),
            "pair_level_top": gate_pair_link.get("top_findings", {}),
            "logic_P_pair_corr": safe_float(
                pair_axis.get("logic", {}).get("prototype_field_proxy", {}).get("targets", {}).get("union_synergy_joint", {}).get("pearson_corr")
            ),
            "logic_B_pair_corr": safe_float(
                pair_axis.get("logic", {}).get("bridge_field_proxy", {}).get("targets", {}).get("union_synergy_joint", {}).get("pearson_corr")
            ),
            "syntax_X_pair_corr": safe_float(
                pair_axis.get("syntax", {}).get("conflict_field_proxy", {}).get("targets", {}).get("union_synergy_joint", {}).get("pearson_corr")
            ),
            "logic_P_category_corr": safe_float(
                category_axis.get("logic", {}).get("prototype_field_proxy", {}).get("targets", {}).get("union_synergy_joint", {}).get("pearson_corr")
            ),
            "logic_B_category_corr": safe_float(
                category_axis.get("logic", {}).get("bridge_field_proxy", {}).get("targets", {}).get("union_synergy_joint", {}).get("pearson_corr")
            ),
            "syntax_X_category_corr": safe_float(
                category_axis.get("syntax", {}).get("conflict_field_proxy", {}).get("targets", {}).get("union_synergy_joint", {}).get("pearson_corr")
            ),
        },
        "interpretation": [
            "logic 更像原型骨架强化器，syntax 更像高约束冲突整形器。",
            "style 在扩到 12 类后变弱，说明它更像弱调制轴，而不是当前闭包主导轴。",
        ],
    }


def build_pos_hypotheses(aggregated_categories: Sequence[Dict[str, object]]) -> Dict[str, object]:
    abstract_row = find_category(aggregated_categories, "abstract")
    action_row = find_category(aggregated_categories, "action")
    tech_row = find_category(aggregated_categories, "tech")
    human_row = find_category(aggregated_categories, "human")

    noun_like = {
        "claim": "普通名词更像锚点加实例偏移。",
        "evidence_level": "direct",
        "support_categories": ["fruit", "object", "animal"],
    }
    adjective_like = {
        "claim": "形容词更像挂在锚点上的修饰纤维。",
        "evidence_level": "direct",
        "support_roles": ["size", "weight"],
    }
    verb_like = {
        "claim": "动词或动作词更像后继传输或协议操作子，而不是静态实体锚点。",
        "evidence_level": "direct",
        "action_summary": action_row,
    }
    abstract_noun_like = {
        "claim": "抽象名词更像协议束或关系束，不像纯实体锚点。",
        "evidence_level": "direct",
        "abstract_summary": abstract_row,
        "tech_summary": tech_row,
        "human_summary": human_row,
    }
    adverb_like = {
        "claim": "副词更可能是二阶调制子：它主要改写动作路径、强度和读出方式，而不是提供新的实体锚点。",
        "evidence_level": "inference_only",
        "reason": "当前仓库缺少副词单独因果探针，但从 style / logic / syntax 与 action 的分离性看，副词更像作用在操作路径上。",
    }

    return {
        "noun": noun_like,
        "adjective": adjective_like,
        "verb": verb_like,
        "abstract_noun": abstract_noun_like,
        "adverb": adverb_like,
    }


def build_why_it_emerges() -> List[Dict[str, object]]:
    return [
        {
            "mechanism": "共享压缩",
            "claim": "大量词项共享家族统计，最省参数的办法不是给每个词单独建模，而是先学家族基底，再学实例偏移。",
            "evidence_level": "inference_with_support",
        },
        {
            "mechanism": "修饰复用",
            "claim": "形容词之所以长成纤维结构，是因为红、甜、重这类修饰要能在很多名词上重复挂接，最自然的实现就是可迁移修饰纤维。",
            "evidence_level": "inference_with_support",
        },
        {
            "mechanism": "关系局部线性",
            "claim": "king / queen 这类局部类比之所以出现，是因为模型在很多上下文里重复看到稳定角色变换，最终形成局部关系偏移轴。",
            "evidence_level": "inference_with_support",
        },
        {
            "mechanism": "生成多约束",
            "claim": "style / logic / syntax 必须同时满足，所以网络会长出并行控制轴，而不是把所有生成约束压成一条单轴。",
            "evidence_level": "direct_plus_inference",
        },
        {
            "mechanism": "闭包压力",
            "claim": "prototype / instance / union 之所以会形成 stage6 闭包结构，是因为语言系统不仅要存词义，还要在上下文里把词义拼成可执行读出。",
            "evidence_level": "direct_plus_inference",
        },
    ]


def build_payload(
    micro_json: Dict[str, object],
    apple_dossier: Dict[str, object],
    concept_family: Dict[str, object],
    triplet_json: Dict[str, object],
    multidim_json: Dict[str, object],
    discovery_summary: Dict[str, object],
    discovery_per_category_rows: Sequence[Dict[str, object]],
    gate_category_link: Dict[str, object],
    gate_pair_link: Dict[str, object],
) -> Dict[str, object]:
    aggregated_categories = aggregate_category_rows(discovery_per_category_rows)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage56_language_system_structure_atlas_v1",
        "title": "语言系统结构总汇图",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "core_equation": "h_language = B_anchor + F_modifier + A_relation + G_control + C_closure",
        "system_claim": "语言系统更像分层锚点、修饰纤维、关系轴、控制轴和闭包读出的组合结构，而不是单一连续词向量场。",
        "structures": {
            "concept_anchor": build_concept_anchor_structure(apple_dossier, concept_family, aggregated_categories),
            "modifier_fiber": build_modifier_structure(micro_json),
            "relation_axis": build_relation_axis_structure(triplet_json),
            "generation_control": build_generation_control_structure(
                apple_dossier,
                multidim_json,
                gate_category_link,
                gate_pair_link,
            ),
        },
        "parts_of_speech": build_pos_hypotheses(aggregated_categories),
        "category_landscape": {
            "aggregated_categories": aggregated_categories,
            "strict_positive_pair_ratio": safe_float(discovery_summary.get("strict_positive_pair_ratio")),
            "margin_zero_pair_ratio": safe_float(discovery_summary.get("margin_zero_pair_ratio")),
        },
        "emergence_reasons": build_why_it_emerges(),
        "hard_limits": [
            "副词目前没有单独因果探针，当前只能做结构推断，不能当成已证事实。",
            "relation axis 目前仍主要来自 king / queen 这类局部例子，还没有形成大规模关系图谱闭环。",
            "modifier fiber 目前主要来自 apple 微观因果图，还缺三模型复核。",
            "当前 system atlas 仍然是语言系统的编码骨架，不是最终严格闭式数学。",
        ],
    }


def build_markdown(payload: Dict[str, object]) -> str:
    structures = payload["structures"]
    pos = payload["parts_of_speech"]
    category_landscape = payload["category_landscape"]

    lines = [
        "# 语言系统结构总汇图",
        "",
        f"- 核心公式: `{payload['core_equation']}`",
        f"- 系统主张: {payload['system_claim']}",
        "",
        "## 已有证据支持的结构",
        f"- 名词/概念: {structures['concept_anchor']['claim']}",
        f"- 形容词: {structures['modifier_fiber']['claim']}",
        f"- 关系轴: {structures['relation_axis']['claim']}",
        f"- 生成控制: {structures['generation_control']['claim']}",
        "",
        "## 词类结构",
        f"- 名词: {pos['noun']['claim']}",
        f"- 形容词: {pos['adjective']['claim']}",
        f"- 动词: {pos['verb']['claim']}",
        f"- 抽象名词: {pos['abstract_noun']['claim']}",
        f"- 副词: {pos['adverb']['claim']}（推断）",
        "",
        "## 关键指标",
        f"- apple micro->meso: {safe_float(structures['concept_anchor']['metrics']['apple_micro_to_meso_jaccard_mean']):.6f}",
        f"- apple meso->macro: {safe_float(structures['concept_anchor']['metrics']['apple_meso_to_macro_jaccard_mean']):.6f}",
        f"- apple shared base ratio: {safe_float(structures['concept_anchor']['metrics']['apple_shared_base_ratio_mean']):.6f}",
        f"- king_queen_jaccard: {safe_float(structures['relation_axis']['metrics']['king_queen_jaccard']):.6f}",
        f"- axis_specificity_index: {safe_float(structures['relation_axis']['metrics']['axis_specificity_index']):.6f}",
        f"- logic->P pair corr: {safe_float(structures['generation_control']['closure_link']['logic_P_pair_corr']):.6f}",
        f"- logic->B pair corr: {safe_float(structures['generation_control']['closure_link']['logic_B_pair_corr']):.6f}",
        f"- syntax->X pair corr: {safe_float(structures['generation_control']['closure_link']['syntax_X_pair_corr']):.6f}",
        f"- 跨模型严格正协同比例: {safe_float(category_landscape['strict_positive_pair_ratio']):.6f}",
        "",
        "## 为什么会长出这种结构",
    ]
    for row in payload["emergence_reasons"]:
        lines.append(f"- {row['mechanism']}: {row['claim']}")
    lines.extend(["", "## 最严格的硬伤"])
    for line in payload["hard_limits"]:
        lines.append(f"- {line}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a clean language-system structure atlas from existing experiment outputs")
    ap.add_argument(
        "--micro-json",
        default=str(ROOT / "tempdata" / "deepseek7b_micro_causal_apple_banana_20260301_210442" / "micro_causal_encoding_graph_results.json"),
    )
    ap.add_argument(
        "--apple-dossier-json",
        default=str(ROOT / "tempdata" / "deepseek7b_apple_encoding_law_dossier_20260306_223055" / "apple_multiaxis_encoding_law_dossier.json"),
    )
    ap.add_argument(
        "--concept-family-json",
        default=str(ROOT / "tempdata" / "deepseek7b_concept_family_parallel_latest" / "concept_family_parallel_scale.json"),
    )
    ap.add_argument(
        "--triplet-json",
        default=str(ROOT / "tempdata" / "deepseek7b_triplet_probe_20260306_150637" / "apple_king_queen_triplet_probe.json"),
    )
    ap.add_argument(
        "--multidim-json",
        default=str(ROOT / "tempdata" / "deepseek7b_multidim_encoding_probe_v2_specific" / "multidim_encoding_probe.json"),
    )
    ap.add_argument(
        "--discovery-summary-json",
        default=str(ROOT / "tempdata" / "stage56_mass_term_large_seq_20260318_1540" / "discovery_summary.json"),
    )
    ap.add_argument(
        "--discovery-per-category-jsonl",
        default=str(ROOT / "tempdata" / "stage56_mass_term_large_seq_20260318_1540" / "discovery_per_category.jsonl"),
    )
    ap.add_argument(
        "--gate-category-link-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_generation_gate_stage6_link_all3_12cat_20260318_2120" / "summary.json"),
    )
    ap.add_argument(
        "--gate-pair-link-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_generation_gate_stage6_pair_link_all3_12cat_pairs_20260318_2120" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_language_system_structure_atlas_20260318_2135"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(
        micro_json=read_json(Path(args.micro_json)),
        apple_dossier=read_json(Path(args.apple_dossier_json)),
        concept_family=read_json(Path(args.concept_family_json)),
        triplet_json=read_json(Path(args.triplet_json)),
        multidim_json=read_json(Path(args.multidim_json)),
        discovery_summary=read_json(Path(args.discovery_summary_json)),
        discovery_per_category_rows=read_jsonl(Path(args.discovery_per_category_jsonl)),
        gate_category_link=read_json(Path(args.gate_category_link_json)),
        gate_pair_link=read_json(Path(args.gate_pair_link_json)),
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "language_system_structure_atlas.json", payload)
    write_markdown(out_dir / "LANGUAGE_SYSTEM_STRUCTURE_ATLAS_REPORT.md", build_markdown(payload))
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "json": str(out_dir / "language_system_structure_atlas.json"),
                "report": str(out_dir / "LANGUAGE_SYSTEM_STRUCTURE_ATLAS_REPORT.md"),
                "core_equation": payload["core_equation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
