from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]

AXES = ("style", "logic", "syntax")
FIELD_PROXY_NAMES = (
    "prototype_field_proxy",
    "instance_field_proxy",
    "bridge_field_proxy",
    "conflict_field_proxy",
    "mismatch_field_proxy",
)
FIELD_SHORT_NAMES = {
    "prototype_field_proxy": "P",
    "instance_field_proxy": "I",
    "bridge_field_proxy": "B",
    "conflict_field_proxy": "X",
    "mismatch_field_proxy": "M",
}
TARGETS = (
    "mean_union_joint_adv",
    "mean_union_synergy_joint",
    "strict_positive_pair_ratio",
)


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_csv_arg(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def safe_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = average(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = average(xs)
    mean_y = average(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return float(num / (den_x * den_y))


def safe_cov(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = average(xs)
    mean_y = average(ys)
    return float(sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / len(xs))


def load_gate_rows(paths: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for raw_path in paths:
        path = Path(raw_path)
        gate_path = path / "cases.jsonl" if path.is_dir() else path
        rows.extend(read_jsonl(gate_path))
    return rows


def filter_gate_rows(
    rows: Sequence[Dict[str, object]],
    model_ids: Sequence[str],
    categories: Sequence[str],
) -> List[Dict[str, object]]:
    allowed_models = {item for item in model_ids if item}
    allowed_categories = {item for item in categories if item}
    selected: List[Dict[str, object]] = []
    for row in rows:
        model_id = str(row.get("model_id", ""))
        category = str(row.get("category", ""))
        if allowed_models and model_id not in allowed_models:
            continue
        if allowed_categories and category not in allowed_categories:
            continue
        selected.append(row)
    return selected


def aggregate_gate_by_model_category(rows: Sequence[Dict[str, object]]) -> Dict[Tuple[str, str], Dict[str, object]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["model_id"]), str(row["category"]))
        buckets[key].append(row)

    aggregated: Dict[Tuple[str, str], Dict[str, object]] = {}
    for key, bucket in buckets.items():
        model_id, category = key
        axis_payload: Dict[str, Dict[str, float]] = {}
        for axis in AXES:
            axis_payload[axis] = {
                field_name: average(
                    [
                        float(case["axis_gate_summary"]["axes"][axis]["deltas"][field_name])
                        for case in bucket
                    ]
                )
                for field_name in FIELD_PROXY_NAMES
            }
        aggregated[key] = {
            "model_id": model_id,
            "category": category,
            "case_count": len(bucket),
            "group_labels": sorted({str(case.get("group_label", "")) for case in bucket}),
            "axes": axis_payload,
        }
    return aggregated


def load_stage6_category_rows(path: str) -> Dict[Tuple[str, str], Dict[str, object]]:
    rows = read_jsonl(Path(path))
    out: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in rows:
        key = (str(row["model_id"]), str(row["category"]))
        out[key] = row
    return out


def join_gate_and_stage6(
    gate_rows: Dict[Tuple[str, str], Dict[str, object]],
    stage6_rows: Dict[Tuple[str, str], Dict[str, object]],
) -> List[Dict[str, object]]:
    joined: List[Dict[str, object]] = []
    for key in sorted(set(gate_rows) & set(stage6_rows)):
        gate_row = gate_rows[key]
        stage6_row = stage6_rows[key]
        joined.append(
            {
                "model_id": key[0],
                "category": key[1],
                "gate_case_count": int(gate_row["case_count"]),
                "group_labels": gate_row["group_labels"],
                "pair_count": int(stage6_row["pair_count"]),
                "strict_positive_pair_count": int(stage6_row["strict_positive_pair_count"]),
                "strict_positive_pair_ratio": float(stage6_row["strict_positive_pair_ratio"]),
                "mean_union_joint_adv": float(stage6_row["mean_union_joint_adv"]),
                "mean_union_synergy_joint": float(stage6_row["mean_union_synergy_joint"]),
                "mean_overlap_ratio": float(stage6_row["mean_overlap_ratio"]),
                "top_instance_term": str(stage6_row["top_instance_term"]),
                "top_row_is_strict_positive": bool(stage6_row["top_row_is_strict_positive"]),
                "axes": gate_row["axes"],
            }
        )
    return joined


def direction_label(value: float, threshold: float = 0.03) -> str:
    if value > threshold:
        return "positive"
    if value < -threshold:
        return "negative"
    return "neutral"


def association_label(corr: float, positive_gap: float, negative_gap: float) -> str:
    corr_dir = direction_label(corr, threshold=0.12)
    gap_dir = direction_label(positive_gap - negative_gap, threshold=0.01)
    if corr_dir == gap_dir and corr_dir != "neutral":
        return corr_dir
    if corr_dir != "neutral" and gap_dir == "neutral":
        return corr_dir
    if corr_dir == "neutral" and gap_dir != "neutral":
        return gap_dir
    if corr_dir == "neutral" and gap_dir == "neutral":
        return "neutral"
    return "mixed"


def build_axis_target_stats(joined_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    positives = [row for row in joined_rows if float(row["strict_positive_pair_ratio"]) > 0.0]
    negatives = [row for row in joined_rows if float(row["strict_positive_pair_ratio"]) == 0.0]

    for axis in AXES:
        axis_block: Dict[str, object] = {}
        for field_name in FIELD_PROXY_NAMES:
            field_values = [float(row["axes"][axis][field_name]) for row in joined_rows]
            pos_values = [float(row["axes"][axis][field_name]) for row in positives]
            neg_values = [float(row["axes"][axis][field_name]) for row in negatives]
            target_block: Dict[str, object] = {}
            for target_name in TARGETS:
                target_values = [float(row[target_name]) for row in joined_rows]
                corr = pearson_corr(field_values, target_values)
                cov = safe_cov(field_values, target_values)
                target_block[target_name] = {
                    "pearson_corr": corr,
                    "covariance": cov,
                    "corr_direction": direction_label(corr, threshold=0.12),
                    "cov_direction": direction_label(cov, threshold=0.0005),
                }

            positive_gap = average(pos_values) - average(field_values)
            negative_gap = average(neg_values) - average(field_values)
            axis_block[field_name] = {
                "field_short_name": FIELD_SHORT_NAMES[field_name],
                "mean_value": average(field_values),
                "std_value": safe_std(field_values),
                "positive_closure_mean": average(pos_values),
                "non_positive_closure_mean": average(neg_values),
                "positive_closure_gap": average(pos_values) - average(neg_values),
                "association_to_closure": association_label(
                    target_block["mean_union_synergy_joint"]["pearson_corr"],
                    average(pos_values),
                    average(neg_values),
                ),
                "targets": target_block,
            }
        out[axis] = axis_block
    return out


def build_top_findings(axis_target_stats: Dict[str, object]) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for axis in AXES:
        for field_name in FIELD_PROXY_NAMES:
            block = axis_target_stats[axis][field_name]
            target_synergy = block["targets"]["mean_union_synergy_joint"]
            rows.append(
                {
                    "axis": axis,
                    "field_name": field_name,
                    "field_short_name": FIELD_SHORT_NAMES[field_name],
                    "association_to_closure": block["association_to_closure"],
                    "corr_synergy": float(target_synergy["pearson_corr"]),
                    "corr_joint_adv": float(block["targets"]["mean_union_joint_adv"]["pearson_corr"]),
                    "corr_closure_ratio": float(block["targets"]["strict_positive_pair_ratio"]["pearson_corr"]),
                    "positive_closure_gap": float(block["positive_closure_gap"]),
                }
            )
    rows.sort(
        key=lambda row: (
            abs(row["corr_synergy"]),
            abs(row["corr_joint_adv"]),
            abs(row["positive_closure_gap"]),
        ),
        reverse=True,
    )
    return {
        "top_synergy_associations": rows[:10],
        "positive_rows": [row for row in rows if row["association_to_closure"] == "positive"][:10],
        "negative_rows": [row for row in rows if row["association_to_closure"] == "negative"][:10],
    }


def build_summary(joined_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    axis_target_stats = build_axis_target_stats(joined_rows)
    return {
        "record_type": "stage56_generation_gate_stage6_link_summary",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "joined_row_count": len(joined_rows),
        "model_count": len({str(row["model_id"]) for row in joined_rows}),
        "category_count": len({str(row["category"]) for row in joined_rows}),
        "models": sorted({str(row["model_id"]) for row in joined_rows}),
        "categories": sorted({str(row["category"]) for row in joined_rows}),
        "stage6_joined_means": {
            target_name: average([float(row[target_name]) for row in joined_rows])
            for target_name in TARGETS
        },
        "axis_target_stats": axis_target_stats,
        "top_findings": build_top_findings(axis_target_stats),
    }


def write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Stage56 Generation Gate Stage6 Link Report",
        "",
        f"- joined_row_count: {summary['joined_row_count']}",
        f"- model_count: {summary['model_count']}",
        f"- category_count: {summary['category_count']}",
        f"- mean_union_joint_adv: {summary['stage6_joined_means']['mean_union_joint_adv']:.6f}",
        f"- mean_union_synergy_joint: {summary['stage6_joined_means']['mean_union_synergy_joint']:.6f}",
        f"- mean_strict_positive_pair_ratio: {summary['stage6_joined_means']['strict_positive_pair_ratio']:.6f}",
        "",
        "## Top Synergy Associations",
    ]
    for row in summary["top_findings"]["top_synergy_associations"]:
        lines.append(
            f"- {row['axis']} / {row['field_short_name']}: "
            f"closure={row['association_to_closure']}, "
            f"corr_synergy={row['corr_synergy']:.4f}, "
            f"corr_joint_adv={row['corr_joint_adv']:.4f}, "
            f"corr_closure_ratio={row['corr_closure_ratio']:.4f}, "
            f"positive_gap={row['positive_closure_gap']:.6f}"
        )
    lines.append("")
    lines.append("## Per Axis")
    for axis in AXES:
        lines.append(f"- {axis}:")
        for field_name in FIELD_PROXY_NAMES:
            block = summary["axis_target_stats"][axis][field_name]
            synergy = block["targets"]["mean_union_synergy_joint"]
            joint_adv = block["targets"]["mean_union_joint_adv"]
            closure = block["targets"]["strict_positive_pair_ratio"]
            lines.append(
                "  - "
                f"{FIELD_SHORT_NAMES[field_name]}: "
                f"assoc={block['association_to_closure']}, "
                f"corr_synergy={synergy['pearson_corr']:.4f}, "
                f"corr_joint_adv={joint_adv['pearson_corr']:.4f}, "
                f"corr_closure_ratio={closure['pearson_corr']:.4f}, "
                f"gap={block['positive_closure_gap']:.6f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Link generation gate field deltas with stage6 category closure metrics")
    ap.add_argument("--gate-inputs", nargs="+", required=True)
    ap.add_argument("--stage6-category-file", required=True)
    ap.add_argument("--model-ids", default="")
    ap.add_argument("--categories", default="")
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_generation_gate_stage6_link_20260318"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    gate_rows = load_gate_rows(args.gate_inputs)
    gate_rows = filter_gate_rows(
        gate_rows,
        model_ids=parse_csv_arg(args.model_ids),
        categories=parse_csv_arg(args.categories),
    )
    gate_by_key = aggregate_gate_by_model_category(gate_rows)
    stage6_rows = load_stage6_category_rows(args.stage6_category_file)
    joined_rows = join_gate_and_stage6(gate_by_key, stage6_rows)
    summary = build_summary(joined_rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "joined_rows.jsonl", joined_rows)
    write_report(out_dir / "REPORT.md", summary)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "joined_row_count": len(joined_rows),
                "model_count": summary["model_count"],
                "category_count": summary["category_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
