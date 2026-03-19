from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
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


def parse_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def role_instability_score(row: Dict[str, object]) -> float:
    proto = safe_float(row.get("proto_joint_adv"))
    inst = safe_float(row.get("instance_joint_adv"))
    synergy = safe_float(row.get("union_synergy_joint"))
    return max(inst - proto, 0.0) + max(-synergy, 0.0)


def protocol_role_pressure(row: Dict[str, object]) -> float:
    proto = safe_float(row.get("proto_joint_adv"))
    inst = safe_float(row.get("instance_joint_adv"))
    synergy = safe_float(row.get("union_synergy_joint"))
    strict_positive = bool(row.get("strict_positive_synergy"))
    return (0.25 * max(inst - proto, 0.0)) + max(-synergy, 0.0) + (0.0 if strict_positive else 1.0)


def overlap_ratio(row: Dict[str, object]) -> float:
    union_count = safe_float(row.get("union_neuron_count"))
    overlap_count = safe_float(row.get("overlap_neuron_count"))
    if union_count <= 0.0:
        return 0.0
    return overlap_count / union_count


def classify_category(
    mean_instance_proto_gap: float,
    mean_union_synergy_joint: float,
    strict_positive_ratio: float,
    mean_overlap_ratio: float,
) -> str:
    if mean_instance_proto_gap > 0.0 and mean_union_synergy_joint < 0.0 and strict_positive_ratio < 0.2:
        return "protocol_role_dominant"
    if mean_union_synergy_joint < 0.0 and strict_positive_ratio < 0.2 and mean_overlap_ratio < 0.4:
        return "mixed_protocol_anchor"
    if mean_instance_proto_gap > 0.0 and strict_positive_ratio < 0.5:
        return "mixed_protocol_anchor"
    return "anchor_like"


def aggregate_stage6_category(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            safe_float(row.get("union_joint_adv")),
            safe_float(row.get("instance_joint_adv")),
            safe_float(row.get("union_synergy_joint")),
        ),
        reverse=True,
    )
    mean_instance_proto_gap = average(
        [safe_float(row.get("instance_joint_adv")) - safe_float(row.get("proto_joint_adv")) for row in rows]
    )
    mean_union_synergy_joint = average([safe_float(row.get("union_synergy_joint")) for row in rows])
    strict_positive_ratio = average([1.0 if bool(row.get("strict_positive_synergy")) else 0.0 for row in rows])
    mean_overlap = average([overlap_ratio(row) for row in rows])
    return {
        "pair_count": len(rows),
        "strict_positive_pair_ratio": strict_positive_ratio,
        "mean_proto_joint_adv": average([safe_float(row.get("proto_joint_adv")) for row in rows]),
        "mean_instance_joint_adv": average([safe_float(row.get("instance_joint_adv")) for row in rows]),
        "mean_union_joint_adv": average([safe_float(row.get("union_joint_adv")) for row in rows]),
        "mean_union_synergy_joint": mean_union_synergy_joint,
        "mean_overlap_ratio": mean_overlap,
        "mean_instance_proto_gap": mean_instance_proto_gap,
        "mean_role_instability_score": average([role_instability_score(row) for row in rows]),
        "mean_protocol_role_pressure": average([protocol_role_pressure(row) for row in rows]),
        "encoding_class": classify_category(
            mean_instance_proto_gap=mean_instance_proto_gap,
            mean_union_synergy_joint=mean_union_synergy_joint,
            strict_positive_ratio=strict_positive_ratio,
            mean_overlap_ratio=mean_overlap,
        ),
        "top_pairs": [
            {
                "model_id": str(row.get("model_id", "")),
                "prototype_term": str(row.get("prototype_term", "")),
                "instance_term": str(row.get("instance_term", "")),
                "union_joint_adv": safe_float(row.get("union_joint_adv")),
                "union_synergy_joint": safe_float(row.get("union_synergy_joint")),
                "instance_proto_gap": safe_float(row.get("instance_joint_adv")) - safe_float(row.get("proto_joint_adv")),
            }
            for row in sorted_rows[:6]
        ],
    }


def build_stage6_rows(paths: Sequence[Path], categories: Sequence[str]) -> List[Dict[str, object]]:
    selected_categories = set(categories)
    rows: List[Dict[str, object]] = []
    for path in paths:
        model_dir = path.parent.parent.name
        for row in read_jsonl(path):
            category = str(row.get("category", ""))
            if selected_categories and category not in selected_categories:
                continue
            enriched = dict(row)
            enriched["model_id"] = model_dir
            rows.append(enriched)
    return rows


def build_discovery_map(rows: Sequence[Dict[str, object]], categories: Sequence[str]) -> Dict[str, object]:
    selected_categories = set(categories)
    out: Dict[str, object] = {}
    for row in rows:
        category = str(row.get("category", ""))
        if selected_categories and category not in selected_categories:
            continue
        model_id = str(row.get("model_tag", row.get("model_id", "")))
        out.setdefault(category, {})[model_id] = {
            "top_instance_term": str(row.get("top_instance_term", "")),
            "strict_positive_pair_ratio": safe_float(row.get("strict_positive_pair_ratio")),
            "mean_union_joint_adv": safe_float(row.get("mean_union_joint_adv")),
            "mean_union_synergy_joint": safe_float(row.get("mean_union_synergy_joint")),
        }
    return out


def build_summary(
    stage6_rows: Sequence[Dict[str, object]],
    discovery_rows: Sequence[Dict[str, object]],
    focus_categories: Sequence[str],
    anchor_categories: Sequence[str],
) -> Dict[str, object]:
    selected_categories = list(dict.fromkeys(list(focus_categories) + list(anchor_categories)))
    per_category = {}
    for category in selected_categories:
        category_rows = [row for row in stage6_rows if str(row.get("category")) == category]
        per_category[category] = aggregate_stage6_category(category_rows)

    discovery_map = build_discovery_map(discovery_rows, selected_categories)
    focus_scores = [safe_float(per_category[category]["mean_role_instability_score"]) for category in focus_categories if category in per_category]
    anchor_scores = [safe_float(per_category[category]["mean_role_instability_score"]) for category in anchor_categories if category in per_category]
    focus_pressures = [safe_float(per_category[category]["mean_protocol_role_pressure"]) for category in focus_categories if category in per_category]
    anchor_pressures = [safe_float(per_category[category]["mean_protocol_role_pressure"]) for category in anchor_categories if category in per_category]
    return {
        "record_type": "stage56_protocol_role_encoding_block_summary",
        "focus_categories": list(focus_categories),
        "anchor_categories": list(anchor_categories),
        "focus_mean_role_instability_score": average(focus_scores),
        "anchor_mean_role_instability_score": average(anchor_scores),
        "focus_mean_protocol_role_pressure": average(focus_pressures),
        "anchor_mean_protocol_role_pressure": average(anchor_pressures),
        "per_category": per_category,
        "discovery_support": discovery_map,
    }


def write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Stage56 Protocol Role Encoding Block",
        "",
        f"- focus_mean_role_instability_score: {safe_float(summary['focus_mean_role_instability_score']):.6f}",
        f"- anchor_mean_role_instability_score: {safe_float(summary['anchor_mean_role_instability_score']):.6f}",
        f"- focus_mean_protocol_role_pressure: {safe_float(summary['focus_mean_protocol_role_pressure']):.6f}",
        f"- anchor_mean_protocol_role_pressure: {safe_float(summary['anchor_mean_protocol_role_pressure']):.6f}",
        "",
        "## Per Category",
    ]
    for category, block in dict(summary["per_category"]).items():
        lines.append(
            f"- {category}: "
            f"class={block['encoding_class']}, "
            f"strict_positive={safe_float(block['strict_positive_pair_ratio']):.4f}, "
            f"instance_proto_gap={safe_float(block['mean_instance_proto_gap']):.4f}, "
            f"union_synergy={safe_float(block['mean_union_synergy_joint']):.4f}, "
            f"role_instability={safe_float(block['mean_role_instability_score']):.4f}, "
            f"protocol_pressure={safe_float(block['mean_protocol_role_pressure']):.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze protocol-role encoding in tech/human/action against anchor categories")
    ap.add_argument(
        "--stage6-result-files",
        nargs="+",
        default=[
            str(ROOT / "tempdata" / "stage56_mass_term_large_seq_20260318_1540" / "deepseek_7b" / "stage6_prototype_instance_decomposition" / "results.jsonl"),
            str(ROOT / "tempdata" / "stage56_mass_term_large_seq_20260318_1540" / "qwen3_4b" / "stage6_prototype_instance_decomposition" / "results.jsonl"),
            str(ROOT / "tempdata" / "stage56_mass_term_large_seq_20260318_1540" / "glm4_9b_chat_hf" / "stage6_prototype_instance_decomposition" / "results.jsonl"),
        ],
    )
    ap.add_argument(
        "--discovery-per-category-jsonl",
        default=str(ROOT / "tempdata" / "stage56_mass_term_large_seq_20260318_1540" / "discovery_per_category.jsonl"),
    )
    ap.add_argument("--focus-categories", default="tech,human,action")
    ap.add_argument("--anchor-categories", default="fruit,object")
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_protocol_role_encoding_block_20260318"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    focus_categories = parse_csv(args.focus_categories)
    anchor_categories = parse_csv(args.anchor_categories)
    selected_categories = focus_categories + anchor_categories
    stage6_rows = build_stage6_rows([Path(path) for path in args.stage6_result_files], selected_categories)
    discovery_rows = read_jsonl(Path(args.discovery_per_category_jsonl))
    summary = build_summary(stage6_rows, discovery_rows, focus_categories, anchor_categories)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "REPORT.md", summary)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "focus_categories": focus_categories,
                "focus_mean_role_instability_score": summary["focus_mean_role_instability_score"],
                "anchor_mean_role_instability_score": summary["anchor_mean_role_instability_score"],
                "focus_mean_protocol_role_pressure": summary["focus_mean_protocol_role_pressure"],
                "anchor_mean_protocol_role_pressure": summary["anchor_mean_protocol_role_pressure"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
