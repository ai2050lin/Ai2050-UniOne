from __future__ import annotations

import argparse
import json
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


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def average(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def layer_band(row: Dict[str, object]) -> str:
    top_layers = list(row.get("top_prototype_layers", []))
    layer_ids = [int(item.get("layer", 0)) for item in top_layers]
    mean_layer = average(layer_ids)
    if mean_layer < 20:
        return "early"
    if mean_layer < 33:
        return "middle"
    return "late"


def top_and_bottom_categories(rows: Sequence[Dict[str, object]]) -> Dict[str, List[str]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            safe_float(row.get("strict_positive_pair_ratio")),
            safe_float(row.get("mean_union_joint_adv")),
            safe_float(row.get("mean_union_synergy_joint")),
        ),
        reverse=True,
    )
    return {"top": [str(row.get("category")) for row in ordered[:4]], "bottom": [str(row.get("category")) for row in ordered[-4:]]}


def aggregate_category_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("category", "")), []).append(row)
    out: List[Dict[str, object]] = []
    for category, bucket in sorted(grouped.items()):
        out.append(
            {
                "category": category,
                "strict_positive_pair_ratio": average([safe_float(row.get("strict_positive_pair_ratio")) for row in bucket]),
                "mean_union_joint_adv": average([safe_float(row.get("mean_union_joint_adv")) for row in bucket]),
                "mean_union_synergy_joint": average([safe_float(row.get("mean_union_synergy_joint")) for row in bucket]),
            }
        )
    return out


def build_model_private(rows: Sequence[Dict[str, object]], internal_map: Dict[str, object]) -> Dict[str, object]:
    per_model_internal = dict(internal_map.get("per_model", {}))
    out: Dict[str, object] = {}
    for row in rows:
        model_id = str(row.get("model_id", row.get("model_tag", "")))
        internal = dict(per_model_internal.get(model_id, {}))
        categories = list(row.get("strict_positive_categories", []))
        if "DeepSeek" in model_id:
            reading = "结构稳定但协同偏负型实现"
        elif "Qwen" in model_id:
            reading = "闭包友好型实现"
        else:
            reading = "类别异常放大型实现"
        out[model_id] = {
            "reading": reading,
            "strict_positive_pair_ratio": safe_float(row.get("strict_positive_pair_ratio")),
            "mean_union_joint_adv": safe_float(row.get("mean_union_joint_adv")),
            "mean_union_synergy_joint": safe_float(row.get("mean_union_synergy_joint")),
            "prototype_layer_band": layer_band(row),
            "strict_positive_categories": categories,
            "internal_axes": {
                axis: {
                    "dominant_hidden_layer": dict(axis_block).get("dominant_hidden_layer", "unknown"),
                    "dominant_mlp_layer": dict(axis_block).get("dominant_mlp_layer", "unknown"),
                    "dominant_attention_head": dict(axis_block).get("dominant_attention_head", "unknown"),
                }
                for axis, axis_block in dict(internal).get("per_axis", {}).items()
            },
        }
    return out


def build_common_laws(
    relation_summary: Dict[str, object],
    probe_summary: Dict[str, object],
    discovery_summary: Dict[str, object],
    pair_link_summary: Dict[str, object],
    bxm_summary: Dict[str, object],
) -> Dict[str, object]:
    top_consensus = [str(row.get("category")) for row in discovery_summary.get("top_consensus_categories", [])[:6]]
    pair_axes = dict(pair_link_summary.get("axis_target_stats", {}))
    logic_p_corr = safe_float(dict(dict(pair_axes.get("logic", {})).get("prototype_field_proxy", {})).get("targets", {}).get("union_synergy_joint", {}).get("pearson_corr"))
    syntax_x_corr = safe_float(dict(dict(pair_axes.get("syntax", {})).get("conflict_field_proxy", {})).get("targets", {}).get("union_synergy_joint", {}).get("pearson_corr"))
    logic_fragile_bridge_corr = safe_float(dict(dict(bxm_summary.get("per_axis", {})).get("logic", {})).get("fragile_bridge", {}).get("corr_to_union_synergy_joint"))
    return {
        "shared_closure_categories": top_consensus,
        "relation_system_split": {
            "local_linear_count": int(dict(relation_summary.get("counts_by_interpretation", {})).get("local_linear", 0)),
            "path_bundle_count": int(dict(relation_summary.get("counts_by_interpretation", {})).get("path_bundle", 0)),
        },
        "wordclass_mechanisms": dict(probe_summary.get("probes", {})),
        "control_laws": {
            "logic_prototype_to_synergy_corr": logic_p_corr,
            "syntax_conflict_to_synergy_corr": syntax_x_corr,
            "logic_fragile_bridge_to_synergy_corr": logic_fragile_bridge_corr,
        },
    }


def build_summary(
    relation_summary: Dict[str, object],
    probe_summary: Dict[str, object],
    discovery_summary: Dict[str, object],
    discovery_per_model_rows: Sequence[Dict[str, object]],
    discovery_per_category_rows: Sequence[Dict[str, object]],
    internal_map: Dict[str, object],
    pair_link_summary: Dict[str, object],
    bxm_summary: Dict[str, object],
) -> Dict[str, object]:
    aggregated_categories = aggregate_category_rows(discovery_per_category_rows)
    return {
        "record_type": "stage56_multimodel_language_unified_atlas_summary",
        "model_count": int(discovery_summary.get("model_count", 0)),
        "global_category_frontier": top_and_bottom_categories(aggregated_categories),
        "common_laws": build_common_laws(relation_summary, probe_summary, discovery_summary, pair_link_summary, bxm_summary),
        "model_private_implementations": build_model_private(discovery_per_model_rows, internal_map),
    }


def write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Stage56 三模型统一语言图谱块",
        "",
        f"- model_count: {int(summary['model_count'])}",
        f"- global_category_frontier: {summary['global_category_frontier']}",
        "",
        "## 共享规律",
    ]
    common = dict(summary["common_laws"])
    lines.append(f"- shared_closure_categories: {common['shared_closure_categories']}")
    lines.append(f"- relation_system_split: {common['relation_system_split']}")
    lines.append(f"- control_laws: {common['control_laws']}")
    lines.extend(["", "## 模型私有实现"])
    for model_id, block in dict(summary["model_private_implementations"]).items():
        lines.append(
            f"- {model_id}: reading={block['reading']}, strict_positive_pair_ratio={safe_float(block['strict_positive_pair_ratio']):.4f}, "
            f"prototype_layer_band={block['prototype_layer_band']}, strict_positive_categories={block['strict_positive_categories']}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Unify relation atlas, word-class probes, and three-model closure/gate results into one language atlas")
    ap.add_argument("--relation-summary-json", default=str(ROOT / "tests" / "codex_temp" / "stage56_large_relation_atlas_20260318" / "summary.json"))
    ap.add_argument("--wordclass-summary-json", default=str(ROOT / "tests" / "codex_temp" / "stage56_wordclass_causal_probe_20260318" / "summary.json"))
    ap.add_argument("--discovery-summary-json", default=str(ROOT / "tempdata" / "stage56_mass_term_large_seq_20260318_1540" / "discovery_summary.json"))
    ap.add_argument("--discovery-per-model-jsonl", default=str(ROOT / "tempdata" / "stage56_mass_term_large_seq_20260318_1540" / "discovery_per_model.jsonl"))
    ap.add_argument("--discovery-per-category-jsonl", default=str(ROOT / "tempdata" / "stage56_mass_term_large_seq_20260318_1540" / "discovery_per_category.jsonl"))
    ap.add_argument("--internal-map-summary-json", default=str(ROOT / "tests" / "codex_temp" / "stage56_generation_gate_internal_map_20260318_1338" / "summary.json"))
    ap.add_argument("--pair-link-summary-json", default=str(ROOT / "tests" / "codex_temp" / "stage56_generation_gate_stage6_pair_link_all3_12cat_pairs_20260318_2120" / "summary.json"))
    ap.add_argument("--bxm-summary-json", default=str(ROOT / "tests" / "codex_temp" / "stage56_bxm_rewrite_20260318_2222" / "summary.json"))
    ap.add_argument("--output-dir", default=str(ROOT / "tests" / "codex_temp" / "stage56_multimodel_language_unified_atlas_20260318"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        relation_summary=read_json(Path(args.relation_summary_json)),
        probe_summary=read_json(Path(args.wordclass_summary_json)),
        discovery_summary=read_json(Path(args.discovery_summary_json)),
        discovery_per_model_rows=read_jsonl(Path(args.discovery_per_model_jsonl)),
        discovery_per_category_rows=read_jsonl(Path(args.discovery_per_category_jsonl)),
        internal_map=read_json(Path(args.internal_map_summary_json)),
        pair_link_summary=read_json(Path(args.pair_link_summary_json)),
        bxm_summary=read_json(Path(args.bxm_summary_json)),
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "REPORT.md", summary)
    print(json.dumps({"output_dir": str(out_dir), "model_count": summary["model_count"], "shared_closure_categories": summary["common_laws"]["shared_closure_categories"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
