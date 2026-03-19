from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
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


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs or not ys or len(xs) != len(ys):
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in zip(xs, ys):
        dx = safe_float(x) - mx
        dy = safe_float(y) - my
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    if den_x <= 0.0 or den_y <= 0.0:
        return 0.0
    return num / ((den_x ** 0.5) * (den_y ** 0.5))


def pair_key(row: Dict[str, object]) -> Tuple[str, str, str, str]:
    return (
        str(row["model_id"]),
        str(row["category"]),
        str(row["prototype_term"]),
        str(row["instance_term"]),
    )


def component_key(row: Dict[str, object]) -> Tuple[str, str, str, str, str]:
    return (
        str(row["model_id"]),
        str(row["category"]),
        str(row["prototype_term"]),
        str(row["instance_term"]),
        str(row["component_label"]),
    )


def axis_density_value(pair_row: Dict[str, object], axis: str, component_label: str) -> float:
    axis_block = dict(dict(pair_row.get("axes", {})).get(axis, {}))
    pair_compaction = [safe_float(x) for x in list(axis_block.get("pair_compaction_curve", []))]
    pair_coverage = [safe_float(x) for x in list(axis_block.get("pair_coverage_curve", []))]
    pair_density = [c * v for c, v in zip(pair_compaction, pair_coverage)]
    if not pair_density:
        return 0.0
    total_points = len(pair_density)
    early_end = max(1, int(round(total_points * 0.2)))
    middle_end = max(early_end + 1, int(round(total_points * 0.6)))
    if component_label == "syntax_constraint_conflict":
        return mean(pair_density[early_end:middle_end])
    return mean(pair_density[middle_end:total_points])


def normalize(values: Sequence[float]) -> List[float]:
    xs = [max(0.0, safe_float(x)) for x in values]
    total = sum(xs)
    if total <= 0.0:
        return [0.0 for _ in xs]
    return [x / total for x in xs]


def outer_mean(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    return mean([safe_float(x) * safe_float(y) for x in a for y in b])


def center_of_mass(profile: Sequence[float]) -> float:
    xs = [max(0.0, safe_float(x)) for x in profile]
    total = sum(xs)
    if total <= 0.0:
        return 0.0
    return sum(idx * value for idx, value in enumerate(xs)) / total


def max_index(profile: Sequence[float]) -> int:
    if not profile:
        return 0
    return max(range(len(profile)), key=lambda idx: abs(safe_float(profile[idx])))


def join_rows(
    pair_rows: Sequence[Dict[str, object]],
    internal_rows: Sequence[Dict[str, object]],
    trajectory_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    pair_map = {pair_key(row): row for row in pair_rows}
    trajectory_map = {component_key(row): row for row in trajectory_rows}
    out: List[Dict[str, object]] = []
    for internal_row in internal_rows:
        comp_key = component_key(internal_row)
        traj = trajectory_map.get(comp_key)
        pair_row = pair_map.get(pair_key(internal_row))
        if traj is None or pair_row is None:
            continue
        axis = str(internal_row["axis"])
        component_label = str(internal_row["component_label"])
        hidden_layer = normalize(list(internal_row.get("hidden_profile", [])))
        mlp_layer = normalize(list(internal_row.get("mlp_profile", [])))
        hidden_window = normalize(list(traj.get("hidden_token_profile", [])))
        mlp_window = normalize(list(traj.get("mlp_token_profile", [])))
        weight = safe_float(internal_row.get("weight"))
        preferred_density = axis_density_value(pair_row, axis=axis, component_label=component_label)
        layer_window_hidden_energy = weight * preferred_density * outer_mean(hidden_layer, hidden_window)
        layer_window_mlp_energy = weight * preferred_density * outer_mean(mlp_layer, mlp_window)
        layer_window_cross_energy = weight * preferred_density * mean(
            [
                outer_mean(hidden_layer, mlp_window),
                outer_mean(mlp_layer, hidden_window),
            ]
        )
        out.append(
            {
                "model_id": str(internal_row["model_id"]),
                "model_label": str(pair_row.get("model_label", "")),
                "category": str(internal_row["category"]),
                "prototype_term": str(internal_row["prototype_term"]),
                "instance_term": str(internal_row["instance_term"]),
                "axis": axis,
                "component_label": component_label,
                "strict_positive_synergy": bool(pair_row.get("strict_positive_synergy")),
                "union_joint_adv": safe_float(pair_row.get("union_joint_adv")),
                "union_synergy_joint": safe_float(pair_row.get("union_synergy_joint")),
                "weight": weight,
                "preferred_density": preferred_density,
                "hidden_layer_center": center_of_mass(hidden_layer),
                "mlp_layer_center": center_of_mass(mlp_layer),
                "hidden_window_center": center_of_mass(hidden_window),
                "mlp_window_center": center_of_mass(mlp_window),
                "hidden_layer_peak": max_index(hidden_layer),
                "mlp_layer_peak": max_index(mlp_layer),
                "hidden_window_peak": max_index(hidden_window),
                "mlp_window_peak": max_index(mlp_window),
                "layer_window_hidden_energy": layer_window_hidden_energy,
                "layer_window_mlp_energy": layer_window_mlp_energy,
                "layer_window_cross_energy": layer_window_cross_energy,
                "tensor_shape": [
                    max(len(hidden_layer), len(mlp_layer)),
                    max(len(hidden_window), len(mlp_window)),
                ],
            }
        )
    return out


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    feature_names = [
        "weight",
        "preferred_density",
        "hidden_layer_center",
        "mlp_layer_center",
        "hidden_window_center",
        "mlp_window_center",
        "hidden_layer_peak",
        "mlp_layer_peak",
        "hidden_window_peak",
        "mlp_window_peak",
        "layer_window_hidden_energy",
        "layer_window_mlp_energy",
        "layer_window_cross_energy",
    ]
    targets = ("union_joint_adv", "union_synergy_joint", "strict_positive_synergy")
    findings: List[Dict[str, object]] = []
    per_component: Dict[str, object] = {}
    for component_label in sorted({str(row["component_label"]) for row in rows}):
        subset = [row for row in rows if str(row["component_label"]) == component_label]
        positives = [row for row in subset if bool(row["strict_positive_synergy"])]
        negatives = [row for row in subset if not bool(row["strict_positive_synergy"])]
        block: Dict[str, object] = {
            "case_count": len(subset),
            "tensor_shapes": sorted({tuple(row["tensor_shape"]) for row in subset}),
            "feature_stats": {},
        }
        for feature_name in feature_names:
            xs = [safe_float(row.get(feature_name)) for row in subset]
            pos_xs = [safe_float(row.get(feature_name)) for row in positives]
            neg_xs = [safe_float(row.get(feature_name)) for row in negatives]
            feature_stat = {
                "mean_value": mean(xs),
                "positive_pair_mean": mean(pos_xs),
                "non_positive_pair_mean": mean(neg_xs),
                "positive_pair_gap": mean(pos_xs) - mean(neg_xs),
                "targets": {},
            }
            for target_name in targets:
                ys = (
                    [1.0 if bool(row["strict_positive_synergy"]) else 0.0 for row in subset]
                    if target_name == "strict_positive_synergy"
                    else [safe_float(row.get(target_name)) for row in subset]
                )
                corr = pearson(xs, ys)
                feature_stat["targets"][target_name] = {"pearson_corr": corr}
                findings.append(
                    {
                        "component_label": component_label,
                        "feature_name": feature_name,
                        "target_name": target_name,
                        "corr": corr,
                        "positive_pair_gap": feature_stat["positive_pair_gap"],
                    }
                )
            block["feature_stats"][feature_name] = feature_stat
        per_component[component_label] = block
    findings.sort(key=lambda row: abs(safe_float(row["corr"])), reverse=True)
    return {
        "record_type": "stage56_component_specific_highdim_field_summary",
        "joined_row_count": len(rows),
        "component_labels": sorted({str(row["component_label"]) for row in rows}),
        "per_component": per_component,
        "top_abs_correlations": findings[:30],
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 组件特异高维场摘要",
        "",
        f"- joined_row_count: {summary['joined_row_count']}",
        f"- component_labels: {', '.join(summary['component_labels'])}",
        "",
        "## Top Correlations",
    ]
    for row in summary["top_abs_correlations"]:
        lines.append(
            f"- {row['component_label']} / {row['feature_name']} -> {row['target_name']}: "
            f"corr={safe_float(row['corr']):+.4f}, positive_gap={safe_float(row['positive_pair_gap']):+.4f}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build component-specific layer-window tensor summaries")
    ap.add_argument(
        "--pair-density-jsonl",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_pair_density_tensor_field_20260319_1512" / "joined_rows.jsonl"),
    )
    ap.add_argument(
        "--internal-joined-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_field_internal_subfield_map_all3_12cat_allpairs_20260319_0122" / "joined_rows.json"),
    )
    ap.add_argument(
        "--trajectory-joined-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_component_trajectory_window_map_all3_12cat_allpairs_20260319_0137" / "joined_rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_component_specific_highdim_field_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pair_rows = read_jsonl(Path(args.pair_density_jsonl))
    internal_rows = list(read_json(Path(args.internal_joined_json)).get("rows", []))
    trajectory_rows = list(read_json(Path(args.trajectory_joined_json)).get("rows", []))
    joined_rows = join_rows(pair_rows, internal_rows, trajectory_rows)
    summary = build_summary(joined_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "joined_rows.json").write_text(
        json.dumps({"rows": joined_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "joined_row_count": len(joined_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
