from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from stage56_generation_gate_stage6_link import (  # noqa: E402
    AXES,
    FIELD_PROXY_NAMES,
    TARGETS,
    FIELD_SHORT_NAMES,
    average,
    association_label,
    build_top_findings,
    direction_label,
    load_gate_rows,
    parse_csv_arg,
    pearson_corr,
    read_jsonl,
    safe_cov,
    safe_std,
    write_json,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]


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


def stage6_result_paths_from_root(root: Path) -> List[Path]:
    paths: List[Path] = []
    for child in sorted(root.iterdir()):
        path = child / "stage6_prototype_instance_decomposition" / "results.jsonl"
        if path.exists():
            paths.append(path)
    return paths


def load_stage6_pair_rows(paths: Sequence[str], output_root: str = "") -> Dict[Tuple[str, str, str, str], Dict[str, object]]:
    result_paths = [Path(path) for path in paths]
    if output_root:
        result_paths.extend(stage6_result_paths_from_root(Path(output_root)))
    out: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    for path in result_paths:
        summary_path = path.parent / "summary.json"
        model_id_from_summary = ""
        if summary_path.exists():
            model_id_from_summary = str(json.loads(summary_path.read_text(encoding="utf-8")).get("model_id", ""))
        for row in read_jsonl(path):
            model_id = str(row.get("model_id", "")) or model_id_from_summary
            if not model_id:
                continue
            key = (
                model_id,
                str(row["category"]),
                str(row["prototype_term"]),
                str(row["instance_term"]),
            )
            out[key] = row
    return out


def join_gate_and_stage6_pairs(
    gate_rows: Sequence[Dict[str, object]],
    stage6_rows: Dict[Tuple[str, str, str, str], Dict[str, object]],
) -> List[Dict[str, object]]:
    joined: List[Dict[str, object]] = []
    for gate_row in gate_rows:
        key = (
            str(gate_row["model_id"]),
            str(gate_row["category"]),
            str(gate_row["prototype_term"]),
            str(gate_row["instance_term"]),
        )
        if key not in stage6_rows:
            continue
        stage6_row = stage6_rows[key]
        joined.append(
            {
                "model_id": key[0],
                "category": key[1],
                "prototype_term": key[2],
                "instance_term": key[3],
                "group_label": str(gate_row["group_label"]),
                "strict_positive_synergy": bool(stage6_row["strict_positive_synergy"]),
                "union_joint_adv": float(stage6_row["union_joint_adv"]),
                "union_synergy_joint": float(stage6_row["union_synergy_joint"]),
                "proto_joint_adv": float(stage6_row["proto_joint_adv"]),
                "instance_joint_adv": float(stage6_row["instance_joint_adv"]),
                "axes": {
                    axis: {
                        field_name: float(gate_row["axis_gate_summary"]["axes"][axis]["deltas"][field_name])
                        for field_name in FIELD_PROXY_NAMES
                    }
                    for axis in AXES
                },
            }
        )
    return joined


def build_axis_target_stats(joined_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    positives = [row for row in joined_rows if bool(row["strict_positive_synergy"])]
    negatives = [row for row in joined_rows if not bool(row["strict_positive_synergy"])]

    for axis in AXES:
        axis_block: Dict[str, object] = {}
        for field_name in FIELD_PROXY_NAMES:
            field_values = [float(row["axes"][axis][field_name]) for row in joined_rows]
            pos_values = [float(row["axes"][axis][field_name]) for row in positives]
            neg_values = [float(row["axes"][axis][field_name]) for row in negatives]
            targets = {
                "union_joint_adv": [float(row["union_joint_adv"]) for row in joined_rows],
                "union_synergy_joint": [float(row["union_synergy_joint"]) for row in joined_rows],
                "strict_positive_synergy_rate": [1.0 if bool(row["strict_positive_synergy"]) else 0.0 for row in joined_rows],
            }
            target_block = {}
            for target_name, target_values in targets.items():
                corr = pearson_corr(field_values, target_values)
                cov = safe_cov(field_values, target_values)
                target_block[target_name] = {
                    "pearson_corr": corr,
                    "covariance": cov,
                    "corr_direction": direction_label(corr, threshold=0.12),
                    "cov_direction": direction_label(cov, threshold=0.0005),
                }
            axis_block[field_name] = {
                "field_short_name": FIELD_SHORT_NAMES[field_name],
                "mean_value": average(field_values),
                "std_value": safe_std(field_values),
                "positive_pair_mean": average(pos_values),
                "non_positive_pair_mean": average(neg_values),
                "positive_pair_gap": average(pos_values) - average(neg_values),
                "association_to_pair_synergy": association_label(
                    target_block["union_synergy_joint"]["pearson_corr"],
                    average(pos_values),
                    average(neg_values),
                ),
                "targets": target_block,
            }
        out[axis] = axis_block
    return out


def build_summary(joined_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    axis_target_stats = build_axis_target_stats(joined_rows)
    temp_summary = {
        "axis_target_stats": {
            axis: {
                field_name: {
                    "association_to_closure": axis_target_stats[axis][field_name]["association_to_pair_synergy"],
                    "positive_closure_gap": axis_target_stats[axis][field_name]["positive_pair_gap"],
                    "targets": {
                        "mean_union_synergy_joint": axis_target_stats[axis][field_name]["targets"]["union_synergy_joint"],
                        "mean_union_joint_adv": axis_target_stats[axis][field_name]["targets"]["union_joint_adv"],
                        "strict_positive_pair_ratio": axis_target_stats[axis][field_name]["targets"]["strict_positive_synergy_rate"],
                    },
                }
                for field_name in FIELD_PROXY_NAMES
            }
            for axis in AXES
        }
    }
    return {
        "record_type": "stage56_generation_gate_stage6_pair_link_summary",
        "joined_row_count": len(joined_rows),
        "model_count": len({str(row["model_id"]) for row in joined_rows}),
        "category_count": len({str(row["category"]) for row in joined_rows}),
        "pair_positive_ratio": average([1.0 if bool(row["strict_positive_synergy"]) else 0.0 for row in joined_rows]),
        "mean_union_joint_adv": average([float(row["union_joint_adv"]) for row in joined_rows]),
        "mean_union_synergy_joint": average([float(row["union_synergy_joint"]) for row in joined_rows]),
        "axis_target_stats": axis_target_stats,
        "top_findings": build_top_findings(temp_summary["axis_target_stats"]),
    }


def write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Stage56 Generation Gate Stage6 Pair Link Report",
        "",
        f"- joined_row_count: {summary['joined_row_count']}",
        f"- model_count: {summary['model_count']}",
        f"- category_count: {summary['category_count']}",
        f"- pair_positive_ratio: {summary['pair_positive_ratio']:.6f}",
        f"- mean_union_joint_adv: {summary['mean_union_joint_adv']:.6f}",
        f"- mean_union_synergy_joint: {summary['mean_union_synergy_joint']:.6f}",
        "",
        "## Top Pair Associations",
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Link generation gate deltas with exact stage6 pair metrics")
    ap.add_argument("--gate-inputs", nargs="+", required=True)
    ap.add_argument("--stage6-result-files", nargs="*", default=[])
    ap.add_argument("--stage6-output-root", default="")
    ap.add_argument("--model-ids", default="")
    ap.add_argument("--categories", default="")
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_generation_gate_stage6_pair_link_20260318"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    gate_rows = filter_gate_rows(
        load_gate_rows(args.gate_inputs),
        model_ids=parse_csv_arg(args.model_ids),
        categories=parse_csv_arg(args.categories),
    )
    stage6_rows = load_stage6_pair_rows(args.stage6_result_files, output_root=args.stage6_output_root)
    joined_rows = join_gate_and_stage6_pairs(gate_rows, stage6_rows)
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
