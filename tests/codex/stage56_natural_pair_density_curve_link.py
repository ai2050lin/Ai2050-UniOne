from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from stage56_density_frontier_curve import infer_model_label, knee_mass_ratio, load_json, safe_float
from stage56_density_frontier_closure_link import mean, normalize_model_label, pearson, read_jsonl

AXES = ("style", "logic", "syntax")


def auc_gap(curve: Sequence[Dict[str, object]]) -> float:
    if len(curve) < 2:
        return 0.0
    area = 0.0
    for i in range(1, len(curve)):
        x0 = safe_float(curve[i - 1]["mass_ratio"])
        x1 = safe_float(curve[i]["mass_ratio"])
        y0 = x0 - safe_float(curve[i - 1]["compaction_ratio"])
        y1 = x1 - safe_float(curve[i]["compaction_ratio"])
        area += (x1 - x0) * (y0 + y1) * 0.5
    return float(area)


def first_mass_ratio_for_coverage(curve: Sequence[Dict[str, object]], threshold: float) -> float:
    for row in curve:
        if safe_float(row.get("layer_coverage_ratio")) >= threshold:
            return safe_float(row["mass_ratio"])
    return 1.0


def summarize_curve(curve: Sequence[Dict[str, object]]) -> Dict[str, float]:
    return {
        "mass10_compaction_ratio": safe_float(next((row["compaction_ratio"] for row in curve if safe_float(row["mass_ratio"]) == 0.10), 0.0)),
        "mass25_compaction_ratio": safe_float(next((row["compaction_ratio"] for row in curve if safe_float(row["mass_ratio"]) == 0.25), 0.0)),
        "mass50_compaction_ratio": safe_float(next((row["compaction_ratio"] for row in curve if safe_float(row["mass_ratio"]) == 0.50), 0.0)),
        "frontier_sharpness_auc": auc_gap(curve),
        "knee_mass_ratio": knee_mass_ratio(curve),
        "full_layer_coverage_mass_ratio": first_mass_ratio_for_coverage(curve, 1.0),
    }


def load_manifest(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for axis, rows in (payload.get("pairs") or {}).items():
        for row in rows or []:
            out[(str(axis), str(row.get("id", "")))] = {
                "term": str(row.get("term", "")),
                "category": str(row.get("category", "")),
            }
    return out


def build_term_curve_metrics(
    probe_json_paths: Sequence[Path],
    manifest: Dict[Tuple[str, str], Dict[str, str]],
) -> Dict[Tuple[str, str, str, str], Dict[str, float]]:
    out: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    for probe_path in probe_json_paths:
        probe = load_json(probe_path)
        model_label = infer_model_label(probe, probe_path)
        for axis, dim in (probe.get("dimensions") or {}).items():
            for pair_row in dim.get("pairs") or []:
                key_info = manifest.get((str(axis), str(pair_row.get("id", ""))))
                if not key_info:
                    continue
                curve = list(pair_row.get("frontier_curve") or [])
                if not curve:
                    continue
                out[(model_label, str(axis), str(key_info["category"]), str(key_info["term"]))] = {
                    "selected_neuron_count": safe_float(pair_row.get("selected_neuron_count")),
                    "delta_l2": safe_float(pair_row.get("delta_l2")),
                    "delta_mean_abs": safe_float(pair_row.get("delta_mean_abs")),
                    **summarize_curve(curve),
                }
    return out


def join_rows(
    joined_rows_path: Path,
    term_metrics: Dict[Tuple[str, str, str, str], Dict[str, float]],
) -> List[Dict[str, object]]:
    joined_rows = read_jsonl(joined_rows_path)
    out: List[Dict[str, object]] = []
    for row in joined_rows:
        model_label = normalize_model_label(str(row.get("model_id", "")))
        category = str(row.get("category", ""))
        proto_term = str(row.get("prototype_term", ""))
        inst_term = str(row.get("instance_term", ""))
        axis_block: Dict[str, object] = {}
        ok = True
        for axis in AXES:
            proto_key = (model_label, axis, category, proto_term)
            inst_key = (model_label, axis, category, inst_term)
            if proto_key not in term_metrics or inst_key not in term_metrics:
                ok = False
                break
            proto = term_metrics[proto_key]
            inst = term_metrics[inst_key]
            axis_block[axis] = {
                "prototype_mass10_compaction_ratio": proto["mass10_compaction_ratio"],
                "instance_mass10_compaction_ratio": inst["mass10_compaction_ratio"],
                "pair_mean_mass10_compaction_ratio": mean([proto["mass10_compaction_ratio"], inst["mass10_compaction_ratio"]]),
                "prototype_mass25_compaction_ratio": proto["mass25_compaction_ratio"],
                "instance_mass25_compaction_ratio": inst["mass25_compaction_ratio"],
                "pair_mean_mass25_compaction_ratio": mean([proto["mass25_compaction_ratio"], inst["mass25_compaction_ratio"]]),
                "prototype_frontier_sharpness_auc": proto["frontier_sharpness_auc"],
                "instance_frontier_sharpness_auc": inst["frontier_sharpness_auc"],
                "pair_mean_frontier_sharpness_auc": mean([proto["frontier_sharpness_auc"], inst["frontier_sharpness_auc"]]),
                "prototype_knee_mass_ratio": proto["knee_mass_ratio"],
                "instance_knee_mass_ratio": inst["knee_mass_ratio"],
                "pair_mean_knee_mass_ratio": mean([proto["knee_mass_ratio"], inst["knee_mass_ratio"]]),
                "prototype_full_layer_coverage_mass_ratio": proto["full_layer_coverage_mass_ratio"],
                "instance_full_layer_coverage_mass_ratio": inst["full_layer_coverage_mass_ratio"],
                "pair_mean_full_layer_coverage_mass_ratio": mean([proto["full_layer_coverage_mass_ratio"], inst["full_layer_coverage_mass_ratio"]]),
                "prototype_delta_l2": proto["delta_l2"],
                "instance_delta_l2": inst["delta_l2"],
                "pair_mean_delta_l2": mean([proto["delta_l2"], inst["delta_l2"]]),
                "prototype_delta_mean_abs": proto["delta_mean_abs"],
                "instance_delta_mean_abs": inst["delta_mean_abs"],
                "pair_mean_delta_mean_abs": mean([proto["delta_mean_abs"], inst["delta_mean_abs"]]),
            }
        if not ok:
            continue
        out.append(
            {
                "model_id": str(row.get("model_id", "")),
                "model_label": model_label,
                "category": category,
                "prototype_term": proto_term,
                "instance_term": inst_term,
                "strict_positive_synergy": bool(row.get("strict_positive_synergy")),
                "union_joint_adv": safe_float(row.get("union_joint_adv")),
                "union_synergy_joint": safe_float(row.get("union_synergy_joint")),
                "axes": axis_block,
            }
        )
    return out


def axis_rows(joined_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in joined_rows:
        for axis in AXES:
            out.append(
                {
                    "model_label": row["model_label"],
                    "category": row["category"],
                    "prototype_term": row["prototype_term"],
                    "instance_term": row["instance_term"],
                    "axis": axis,
                    "strict_positive_synergy": row["strict_positive_synergy"],
                    "union_joint_adv": row["union_joint_adv"],
                    "union_synergy_joint": row["union_synergy_joint"],
                    **dict(row["axes"][axis]),
                }
            )
    return out


def build_summary(joined_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    rows = axis_rows(joined_rows)
    feature_names = [
        "prototype_mass10_compaction_ratio",
        "instance_mass10_compaction_ratio",
        "pair_mean_mass10_compaction_ratio",
        "prototype_mass25_compaction_ratio",
        "instance_mass25_compaction_ratio",
        "pair_mean_mass25_compaction_ratio",
        "prototype_frontier_sharpness_auc",
        "instance_frontier_sharpness_auc",
        "pair_mean_frontier_sharpness_auc",
        "prototype_knee_mass_ratio",
        "instance_knee_mass_ratio",
        "pair_mean_knee_mass_ratio",
        "prototype_full_layer_coverage_mass_ratio",
        "instance_full_layer_coverage_mass_ratio",
        "pair_mean_full_layer_coverage_mass_ratio",
        "prototype_delta_l2",
        "instance_delta_l2",
        "pair_mean_delta_l2",
        "prototype_delta_mean_abs",
        "instance_delta_mean_abs",
        "pair_mean_delta_mean_abs",
    ]
    target_names = ("union_joint_adv", "union_synergy_joint", "strict_positive_synergy")
    findings: List[Dict[str, object]] = []
    axis_feature_stats: Dict[str, Dict[str, object]] = {}
    for axis in AXES:
        axis_feature_stats[axis] = {}
        subset = [row for row in rows if str(row["axis"]) == axis]
        positives = [row for row in subset if bool(row["strict_positive_synergy"])]
        negatives = [row for row in subset if not bool(row["strict_positive_synergy"])]
        for feature_name in feature_names:
            xs = [safe_float(row.get(feature_name)) for row in subset]
            pos_xs = [safe_float(row.get(feature_name)) for row in positives]
            neg_xs = [safe_float(row.get(feature_name)) for row in negatives]
            axis_feature_stats[axis][feature_name] = {
                "mean_value": mean(xs),
                "positive_pair_mean": mean(pos_xs),
                "non_positive_pair_mean": mean(neg_xs),
                "positive_pair_gap": mean(pos_xs) - mean(neg_xs),
                "targets": {},
            }
            for target_name in target_names:
                if target_name == "strict_positive_synergy":
                    ys = [1.0 if bool(row["strict_positive_synergy"]) else 0.0 for row in subset]
                else:
                    ys = [safe_float(row.get(target_name)) for row in subset]
                corr = pearson(xs, ys)
                axis_feature_stats[axis][feature_name]["targets"][target_name] = {"pearson_corr": corr}
                findings.append(
                    {
                        "axis": axis,
                        "feature_name": feature_name,
                        "target_name": target_name,
                        "corr": corr,
                        "positive_pair_gap": axis_feature_stats[axis][feature_name]["positive_pair_gap"],
                    }
                )
    findings.sort(key=lambda row: abs(safe_float(row["corr"])), reverse=True)
    return {
        "record_type": "stage56_natural_pair_density_curve_link_summary",
        "joined_pair_count": len(joined_rows),
        "axis_row_count": len(rows),
        "model_count": len({str(row["model_label"]) for row in rows}),
        "category_count": len({str(row["category"]) for row in rows}),
        "strict_positive_pair_ratio": mean([1.0 if bool(row["strict_positive_synergy"]) else 0.0 for row in joined_rows]),
        "axis_feature_stats": axis_feature_stats,
        "top_abs_correlations": findings[:18],
    }


def build_markdown(summary: Dict[str, object]) -> str:
    lines = [
        "# 自然语料 Pair 级连续前沿到闭包联立",
        "",
        f"- joined_pair_count: {summary['joined_pair_count']}",
        f"- axis_row_count: {summary['axis_row_count']}",
        f"- model_count: {summary['model_count']}",
        f"- category_count: {summary['category_count']}",
        "",
        "## Top Correlations",
    ]
    for row in summary["top_abs_correlations"]:
        lines.append(
            f"- {row['axis']} / {row['feature_name']} -> {row['target_name']}: "
            f"corr={row['corr']:+.4f}, positive_gap={row['positive_pair_gap']:+.4f}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Link exact pair-level density curves to stage6 pair closure")
    ap.add_argument("--pairs-json", required=True)
    ap.add_argument("--probe-json", action="append", required=True)
    ap.add_argument("--joined-rows", required=True)
    ap.add_argument("--output-dir", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        f"tests/codex_temp/stage56_natural_pair_density_curve_link_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(Path(args.pairs_json))
    term_metrics = build_term_curve_metrics([Path(x) for x in args.probe_json], manifest)
    joined_rows = join_rows(Path(args.joined_rows), term_metrics)
    summary = build_summary(joined_rows)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "joined_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in joined_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "SUMMARY.md").write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps({"output_dir": out_dir.as_posix(), "joined_pair_count": len(joined_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
