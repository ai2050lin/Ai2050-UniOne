from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from stage56_density_frontier_curve import infer_model_label, load_json, safe_float
from stage56_density_frontier_closure_link import mean, normalize_model_label, pearson, read_jsonl
from stage56_natural_pair_density_curve_link import load_manifest

AXES = ("style", "logic", "syntax")


def ratio_key(value: float) -> str:
    return f"{int(round(float(value) * 100.0)):02d}"


def curve_to_maps(curve: Sequence[Dict[str, object]]) -> Tuple[List[float], Dict[str, float], Dict[str, float]]:
    mass_ratios: List[float] = []
    compaction: Dict[str, float] = {}
    coverage: Dict[str, float] = {}
    for row in curve:
        mass_ratio = safe_float(row.get("mass_ratio"))
        key = ratio_key(mass_ratio)
        mass_ratios.append(mass_ratio)
        compaction[key] = safe_float(row.get("compaction_ratio"))
        coverage[key] = safe_float(row.get("layer_coverage_ratio"))
    return mass_ratios, compaction, coverage


def load_term_payloads(
    probe_json_paths: Sequence[Path],
    manifest: Dict[Tuple[str, str], Dict[str, str]],
) -> Tuple[Dict[Tuple[str, str, str, str], Dict[str, object]], List[float]]:
    out: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    global_mass_ratios: List[float] = []
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
                mass_ratios, compaction_map, coverage_map = curve_to_maps(curve)
                if not global_mass_ratios:
                    global_mass_ratios = list(mass_ratios)
                out[(model_label, str(axis), str(key_info["category"]), str(key_info["term"]))] = {
                    "mass_ratios": mass_ratios,
                    "compaction_curve": compaction_map,
                    "coverage_curve": coverage_map,
                    "delta_l2": safe_float(pair_row.get("delta_l2")),
                    "delta_mean_abs": safe_float(pair_row.get("delta_mean_abs")),
                }
    return out, global_mass_ratios


def vector_from_map(curve_map: Dict[str, float], mass_ratios: Sequence[float]) -> List[float]:
    return [safe_float(curve_map.get(ratio_key(mass_ratio))) for mass_ratio in mass_ratios]


def segment_mean(values: Sequence[float], start_idx: int, end_idx: int) -> float:
    return mean([safe_float(value) for value in list(values)[start_idx:end_idx]])


def l1_mean(a: Sequence[float], b: Sequence[float]) -> float:
    return mean([abs(safe_float(x) - safe_float(y)) for x, y in zip(a, b)])


def build_axis_tensor(
    proto: Dict[str, object],
    inst: Dict[str, object],
    mass_ratios: Sequence[float],
) -> Dict[str, object]:
    proto_compaction = vector_from_map(dict(proto["compaction_curve"]), mass_ratios)
    inst_compaction = vector_from_map(dict(inst["compaction_curve"]), mass_ratios)
    proto_coverage = vector_from_map(dict(proto["coverage_curve"]), mass_ratios)
    inst_coverage = vector_from_map(dict(inst["coverage_curve"]), mass_ratios)
    pair_compaction = [mean([a, b]) for a, b in zip(proto_compaction, inst_compaction)]
    pair_coverage = [mean([a, b]) for a, b in zip(proto_coverage, inst_coverage)]

    total_points = len(mass_ratios)
    early_end = max(1, int(round(total_points * 0.2)))
    middle_end = max(early_end + 1, int(round(total_points * 0.6)))

    return {
        "mass_ratios": list(mass_ratios),
        "tensor_shape": [2, 2, total_points],
        "prototype_compaction_curve": proto_compaction,
        "instance_compaction_curve": inst_compaction,
        "prototype_coverage_curve": proto_coverage,
        "instance_coverage_curve": inst_coverage,
        "pair_compaction_curve": pair_compaction,
        "pair_coverage_curve": pair_coverage,
        "role_asymmetry_compaction_l1": l1_mean(proto_compaction, inst_compaction),
        "role_asymmetry_coverage_l1": l1_mean(proto_coverage, inst_coverage),
        "channel_alignment_proto": pearson(proto_compaction, proto_coverage),
        "channel_alignment_instance": pearson(inst_compaction, inst_coverage),
        "role_alignment_compaction": pearson(proto_compaction, inst_compaction),
        "role_alignment_coverage": pearson(proto_coverage, inst_coverage),
        "pair_compaction_early_mean": segment_mean(pair_compaction, 0, early_end),
        "pair_compaction_middle_mean": segment_mean(pair_compaction, early_end, middle_end),
        "pair_compaction_late_mean": segment_mean(pair_compaction, middle_end, total_points),
        "pair_coverage_early_mean": segment_mean(pair_coverage, 0, early_end),
        "pair_coverage_middle_mean": segment_mean(pair_coverage, early_end, middle_end),
        "pair_coverage_late_mean": segment_mean(pair_coverage, middle_end, total_points),
        "pair_delta_l2": mean([safe_float(proto["delta_l2"]), safe_float(inst["delta_l2"])]),
        "pair_delta_mean_abs": mean([safe_float(proto["delta_mean_abs"]), safe_float(inst["delta_mean_abs"])]),
    }


def join_rows(
    joined_rows_path: Path,
    term_payloads: Dict[Tuple[str, str, str, str], Dict[str, object]],
    mass_ratios: Sequence[float],
) -> List[Dict[str, object]]:
    rows = read_jsonl(joined_rows_path)
    out: List[Dict[str, object]] = []
    for row in rows:
        model_label = normalize_model_label(str(row.get("model_id", "")))
        category = str(row.get("category", ""))
        prototype_term = str(row.get("prototype_term", ""))
        instance_term = str(row.get("instance_term", ""))
        axes: Dict[str, object] = {}
        ok = True
        for axis in AXES:
            proto = term_payloads.get((model_label, axis, category, prototype_term))
            inst = term_payloads.get((model_label, axis, category, instance_term))
            if not proto or not inst:
                ok = False
                break
            axes[axis] = build_axis_tensor(proto, inst, mass_ratios)
        if not ok:
            continue
        out.append(
            {
                "model_id": str(row.get("model_id", "")),
                "model_label": model_label,
                "category": category,
                "prototype_term": prototype_term,
                "instance_term": instance_term,
                "strict_positive_synergy": bool(row.get("strict_positive_synergy")),
                "union_joint_adv": safe_float(row.get("union_joint_adv")),
                "union_synergy_joint": safe_float(row.get("union_synergy_joint")),
                "axes": axes,
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
        "role_asymmetry_compaction_l1",
        "role_asymmetry_coverage_l1",
        "channel_alignment_proto",
        "channel_alignment_instance",
        "role_alignment_compaction",
        "role_alignment_coverage",
        "pair_compaction_early_mean",
        "pair_compaction_middle_mean",
        "pair_compaction_late_mean",
        "pair_coverage_early_mean",
        "pair_coverage_middle_mean",
        "pair_coverage_late_mean",
        "pair_delta_l2",
        "pair_delta_mean_abs",
    ]
    targets = ("union_joint_adv", "union_synergy_joint", "strict_positive_synergy")
    per_axis: Dict[str, object] = {}
    findings: List[Dict[str, object]] = []
    for axis in AXES:
        subset = [row for row in rows if str(row.get("axis")) == axis]
        positives = [row for row in subset if bool(row.get("strict_positive_synergy"))]
        negatives = [row for row in subset if not bool(row.get("strict_positive_synergy"))]
        axis_block: Dict[str, object] = {"pair_count": len(subset), "feature_stats": {}}
        for feature_name in feature_names:
            xs = [safe_float(row.get(feature_name)) for row in subset]
            pos_xs = [safe_float(row.get(feature_name)) for row in positives]
            neg_xs = [safe_float(row.get(feature_name)) for row in negatives]
            stat = {
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
                stat["targets"][target_name] = {"pearson_corr": corr}
                findings.append(
                    {
                        "axis": axis,
                        "feature_name": feature_name,
                        "target_name": target_name,
                        "corr": corr,
                        "positive_pair_gap": stat["positive_pair_gap"],
                    }
                )
            axis_block["feature_stats"][feature_name] = stat
        per_axis[axis] = axis_block
    findings.sort(key=lambda row: abs(safe_float(row["corr"])), reverse=True)
    return {
        "record_type": "stage56_pair_density_tensor_field_summary",
        "joined_pair_count": len(joined_rows),
        "axis_row_count": len(rows),
        "per_axis": per_axis,
        "top_abs_correlations": findings[:24],
    }


def build_markdown(summary: Dict[str, object]) -> str:
    lines = [
        "# Pair 级连续密度场张量摘要",
        "",
        f"- joined_pair_count: {summary['joined_pair_count']}",
        f"- axis_row_count: {summary['axis_row_count']}",
        "",
        "## Top Tensor Correlations",
    ]
    for row in summary["top_abs_correlations"]:
        lines.append(
            f"- {row['axis']} / {row['feature_name']} -> {row['target_name']}: "
            f"corr={safe_float(row['corr']):+.4f}, positive_gap={safe_float(row['positive_pair_gap']):+.4f}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build pair-level density tensor field and link it to closure")
    ap.add_argument("--pairs-json", required=True)
    ap.add_argument("--probe-json", action="append", required=True)
    ap.add_argument("--joined-rows", required=True)
    ap.add_argument("--output-dir", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tests/codex_temp/stage56_pair_density_tensor_field_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(Path(args.pairs_json))
    term_payloads, mass_ratios = load_term_payloads([Path(path) for path in args.probe_json], manifest)
    joined_rows = join_rows(Path(args.joined_rows), term_payloads, mass_ratios)
    summary = build_summary(joined_rows)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "joined_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in joined_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "SUMMARY.md").write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps({"output_dir": out_dir.as_posix(), "joined_pair_count": len(joined_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
