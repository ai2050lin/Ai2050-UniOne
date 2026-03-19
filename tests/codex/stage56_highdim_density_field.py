from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]

AXES = ("style", "logic", "syntax")
COMPONENT_TO_AXIS = {
    "logic_prototype": "logic",
    "logic_fragile_bridge": "logic",
    "syntax_constraint_conflict": "syntax",
}


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


def parse_layer_index(label: str) -> int:
    text = str(label)
    marker = "layer_"
    if marker not in text:
        return 0
    rest = text.split(marker, 1)[1]
    number = []
    for ch in rest:
        if ch.isdigit():
            number.append(ch)
        else:
            break
    return int("".join(number)) if number else 0


def parse_tail_position(label: str) -> int:
    text = str(label)
    marker = "tail_pos_"
    if marker not in text:
        return -1
    rest = text.split(marker, 1)[1]
    try:
        return int(rest)
    except Exception:
        return -1


def peak_index(values: Sequence[float]) -> int:
    if not values:
        return 0
    return max(range(len(values)), key=lambda idx: abs(safe_float(values[idx])))


def alignment_score(a: int, b: int, width: int) -> float:
    if width <= 1:
        return 1.0
    return 1.0 - (abs(a - b) / float(width - 1))


def window_preclosure_score(hidden_pos: int, mlp_pos: int) -> float:
    # 理想窗口大致落在句尾前 -9 到 -5 之间。
    target = -7.0
    span = 8.0
    hidden_score = max(0.0, 1.0 - abs(hidden_pos - target) / span)
    mlp_score = max(0.0, 1.0 - abs(mlp_pos - target) / span)
    return mean([hidden_score, mlp_score])


def segment_mean(values: Sequence[float], start: int, end: int) -> float:
    return mean([safe_float(v) for v in list(values)[start:end]])


def density_segment_features(axis_block: Dict[str, object]) -> Dict[str, float]:
    mass_ratios = list(axis_block.get("mass_ratios", []))
    total_points = len(mass_ratios)
    if total_points <= 0:
        return {
            "middle_density_volume": 0.0,
            "late_density_volume": 0.0,
            "early_density_volume": 0.0,
        }
    early_end = max(1, int(round(total_points * 0.2)))
    middle_end = max(early_end + 1, int(round(total_points * 0.6)))
    pair_compaction = list(axis_block.get("pair_compaction_curve", []))
    pair_coverage = list(axis_block.get("pair_coverage_curve", []))
    pair_density = [
        safe_float(c) * safe_float(v)
        for c, v in zip(pair_compaction, pair_coverage)
    ]
    return {
        "early_density_volume": segment_mean(pair_density, 0, early_end),
        "middle_density_volume": segment_mean(pair_density, early_end, middle_end),
        "late_density_volume": segment_mean(pair_density, middle_end, total_points),
    }


def build_component_bundle(
    pair_row: Dict[str, object],
    internal_row: Dict[str, object],
    trajectory_row: Dict[str, object],
) -> Dict[str, object]:
    axis = str(internal_row["axis"])
    axis_block = dict(dict(pair_row.get("axes", {})).get(axis, {}))
    density = density_segment_features(axis_block)
    hidden_profile = list(internal_row.get("hidden_profile", []))
    mlp_profile = list(internal_row.get("mlp_profile", []))
    hidden_token_profile = list(trajectory_row.get("hidden_token_profile", []))
    mlp_token_profile = list(trajectory_row.get("mlp_token_profile", []))
    hidden_peak_layer = peak_index(hidden_profile)
    mlp_peak_layer = peak_index(mlp_profile)
    hidden_peak_window = peak_index(hidden_token_profile)
    mlp_peak_window = peak_index(mlp_token_profile)
    hidden_tail_position = parse_tail_position(trajectory_row.get("dominant_hidden_tail_position", "tail_pos_-1"))
    mlp_tail_position = parse_tail_position(trajectory_row.get("dominant_mlp_tail_position", "tail_pos_-1"))
    layer_width = max(len(hidden_profile), len(mlp_profile), 1)
    window_width = max(len(hidden_token_profile), len(mlp_token_profile), 1)
    layer_coherence = alignment_score(hidden_peak_layer, mlp_peak_layer, layer_width)
    window_coherence = alignment_score(hidden_peak_window, mlp_peak_window, window_width)
    cross_scale_coherence = mean([layer_coherence, window_coherence])
    weight = safe_float(internal_row.get("weight"))

    preferred_density = density["middle_density_volume"]
    if str(internal_row["component_label"]) in {"logic_prototype", "logic_fragile_bridge"}:
        preferred_density = density["late_density_volume"]

    return {
        "axis": axis,
        "component_label": str(internal_row["component_label"]),
        "weight": weight,
        "tensor_shape": [layer_width, 2, 2, len(axis_block.get("mass_ratios", [])), window_width],
        "hidden_peak_layer_index": hidden_peak_layer,
        "mlp_peak_layer_index": mlp_peak_layer,
        "hidden_peak_window_index": hidden_peak_window,
        "mlp_peak_window_index": mlp_peak_window,
        "dominant_hidden_layer": str(internal_row.get("dominant_hidden_layer", "layer_0")),
        "dominant_mlp_layer": str(internal_row.get("dominant_mlp_layer", "layer_0")),
        "dominant_hidden_tail_position": str(trajectory_row.get("dominant_hidden_tail_position", "tail_pos_-1")),
        "dominant_mlp_tail_position": str(trajectory_row.get("dominant_mlp_tail_position", "tail_pos_-1")),
        "hidden_tail_position_index": hidden_tail_position,
        "mlp_tail_position_index": mlp_tail_position,
        "layer_coherence": layer_coherence,
        "window_coherence": window_coherence,
        "cross_scale_coherence": cross_scale_coherence,
        "window_preclosure_score": window_preclosure_score(hidden_tail_position, mlp_tail_position),
        "early_density_volume": density["early_density_volume"],
        "middle_density_volume": density["middle_density_volume"],
        "late_density_volume": density["late_density_volume"],
        "preferred_density_volume": preferred_density,
        "weighted_preferred_density_volume": weight * preferred_density,
        "weighted_cross_scale_energy": weight * preferred_density * cross_scale_coherence,
    }


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
        pair_row = pair_map.get(pair_key(internal_row))
        trajectory_row = trajectory_map.get(comp_key)
        if pair_row is None or trajectory_row is None:
            continue
        component_bundle = build_component_bundle(pair_row, internal_row, trajectory_row)
        out.append(
            {
                "model_id": str(internal_row["model_id"]),
                "model_label": str(pair_row.get("model_label", "")),
                "category": str(internal_row["category"]),
                "prototype_term": str(internal_row["prototype_term"]),
                "instance_term": str(internal_row["instance_term"]),
                "component_label": str(internal_row["component_label"]),
                "axis": str(component_bundle["axis"]),
                "strict_positive_synergy": bool(pair_row.get("strict_positive_synergy")),
                "union_joint_adv": safe_float(pair_row.get("union_joint_adv")),
                "union_synergy_joint": safe_float(pair_row.get("union_synergy_joint")),
                "component": component_bundle,
            }
        )
    return out


def axis_rows(joined_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in joined_rows:
        comp = dict(row["component"])
        rows.append(
            {
                "model_label": row["model_label"],
                "category": row["category"],
                "prototype_term": row["prototype_term"],
                "instance_term": row["instance_term"],
                "component_label": row["component_label"],
                "axis": row["axis"],
                "strict_positive_synergy": row["strict_positive_synergy"],
                "union_joint_adv": row["union_joint_adv"],
                "union_synergy_joint": row["union_synergy_joint"],
                **comp,
            }
        )
    return rows


def build_summary(joined_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    rows = axis_rows(joined_rows)
    feature_names = [
        "weight",
        "layer_coherence",
        "window_coherence",
        "cross_scale_coherence",
        "window_preclosure_score",
        "early_density_volume",
        "middle_density_volume",
        "late_density_volume",
        "preferred_density_volume",
        "weighted_preferred_density_volume",
        "weighted_cross_scale_energy",
        "hidden_peak_layer_index",
        "mlp_peak_layer_index",
        "hidden_tail_position_index",
        "mlp_tail_position_index",
    ]
    targets = ("union_joint_adv", "union_synergy_joint", "strict_positive_synergy")
    per_component: Dict[str, object] = {}
    findings: List[Dict[str, object]] = []
    for component_label in sorted({str(row["component_label"]) for row in rows}):
        subset = [row for row in rows if str(row.get("component_label")) == component_label]
        positives = [row for row in subset if bool(row.get("strict_positive_synergy"))]
        negatives = [row for row in subset if not bool(row.get("strict_positive_synergy"))]
        block: Dict[str, object] = {"case_count": len(subset), "feature_stats": {}}
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
                    [1.0 if bool(row.get("strict_positive_synergy")) else 0.0 for row in subset]
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
        "record_type": "stage56_highdim_density_field_summary",
        "joined_row_count": len(joined_rows),
        "component_labels": sorted({str(row["component_label"]) for row in rows}),
        "per_component": per_component,
        "top_abs_correlations": findings[:30],
    }


def build_markdown(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 高维连续密度场摘要",
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
    ap = argparse.ArgumentParser(description="Build a factorized high-dimensional density field from pair, layer, and window data")
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
        default=str(ROOT / "tests" / "codex_temp" / "stage56_highdim_density_field_20260319"),
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
    (output_dir / "REPORT.md").write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "joined_row_count": len(joined_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
