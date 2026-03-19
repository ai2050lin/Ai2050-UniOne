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
TARGETS = ("union_joint_adv", "union_synergy_joint", "strict_positive_synergy")


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


def build_term_curve_payloads(
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


def join_rows(
    joined_rows_path: Path,
    term_payloads: Dict[Tuple[str, str, str, str], Dict[str, object]],
    mass_ratios: Sequence[float],
) -> List[Dict[str, object]]:
    source_rows = read_jsonl(joined_rows_path)
    out: List[Dict[str, object]] = []
    keys = [ratio_key(value) for value in mass_ratios]
    for row in source_rows:
        model_label = normalize_model_label(str(row.get("model_id", "")))
        category = str(row.get("category", ""))
        prototype_term = str(row.get("prototype_term", ""))
        instance_term = str(row.get("instance_term", ""))
        axis_block: Dict[str, object] = {}
        ok = True
        for axis in AXES:
            proto_key = (model_label, axis, category, prototype_term)
            inst_key = (model_label, axis, category, instance_term)
            proto = term_payloads.get(proto_key)
            inst = term_payloads.get(inst_key)
            if not proto or not inst:
                ok = False
                break
            pair_compaction = {
                key: mean([safe_float(proto["compaction_curve"].get(key)), safe_float(inst["compaction_curve"].get(key))])
                for key in keys
            }
            pair_coverage = {
                key: mean([safe_float(proto["coverage_curve"].get(key)), safe_float(inst["coverage_curve"].get(key))])
                for key in keys
            }
            axis_block[axis] = {
                "pair_compaction_curve": pair_compaction,
                "pair_coverage_curve": pair_coverage,
                "pair_mean_delta_l2": mean([safe_float(proto["delta_l2"]), safe_float(inst["delta_l2"])]),
                "pair_mean_delta_mean_abs": mean([safe_float(proto["delta_mean_abs"]), safe_float(inst["delta_mean_abs"])]),
            }
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
                "axes": axis_block,
            }
        )
    return out


def longest_positive_band(points: Sequence[Dict[str, object]], threshold: float = 0.20) -> Dict[str, object]:
    best: Dict[str, object] = {"length": 0, "start_mass_ratio": 0.0, "end_mass_ratio": 0.0, "mean_corr": 0.0}
    current: List[Dict[str, object]] = []
    for row in points:
        if safe_float(row.get("corr")) >= threshold:
            current.append(row)
            continue
        if current:
            mean_corr = mean([safe_float(item.get("corr")) for item in current])
            if len(current) > int(best["length"]) or (
                len(current) == int(best["length"]) and mean_corr > safe_float(best["mean_corr"])
            ):
                best = {
                    "length": len(current),
                    "start_mass_ratio": safe_float(current[0].get("mass_ratio")),
                    "end_mass_ratio": safe_float(current[-1].get("mass_ratio")),
                    "mean_corr": mean_corr,
                }
            current = []
    if current:
        mean_corr = mean([safe_float(item.get("corr")) for item in current])
        if len(current) > int(best["length"]) or (
            len(current) == int(best["length"]) and mean_corr > safe_float(best["mean_corr"])
        ):
            best = {
                "length": len(current),
                "start_mass_ratio": safe_float(current[0].get("mass_ratio")),
                "end_mass_ratio": safe_float(current[-1].get("mass_ratio")),
                "mean_corr": mean_corr,
            }
    return best


def axis_rows(joined_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in joined_rows:
        for axis in AXES:
            rows.append(
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
    return rows


def build_summary(joined_rows: Sequence[Dict[str, object]], mass_ratios: Sequence[float]) -> Dict[str, object]:
    rows = axis_rows(joined_rows)
    per_axis: Dict[str, object] = {}
    top_findings: List[Dict[str, object]] = []
    for axis in AXES:
        subset = [row for row in rows if str(row["axis"]) == axis]
        positives = [row for row in subset if bool(row["strict_positive_synergy"])]
        negatives = [row for row in subset if not bool(row["strict_positive_synergy"])]
        axis_block: Dict[str, object] = {
            "pair_count": len(subset),
            "continuous_compaction_corr": {},
            "continuous_coverage_corr": {},
            "best_points": {},
            "positive_bands": {},
        }
        for target_name in TARGETS:
            target_values = (
                [1.0 if bool(row["strict_positive_synergy"]) else 0.0 for row in subset]
                if target_name == "strict_positive_synergy"
                else [safe_float(row.get(target_name)) for row in subset]
            )
            compaction_points: List[Dict[str, object]] = []
            coverage_points: List[Dict[str, object]] = []
            for mass_ratio in mass_ratios:
                key = ratio_key(mass_ratio)
                compaction_values = [safe_float(dict(row["pair_compaction_curve"]).get(key)) for row in subset]
                coverage_values = [safe_float(dict(row["pair_coverage_curve"]).get(key)) for row in subset]
                pos_compaction = [safe_float(dict(row["pair_compaction_curve"]).get(key)) for row in positives]
                neg_compaction = [safe_float(dict(row["pair_compaction_curve"]).get(key)) for row in negatives]
                pos_coverage = [safe_float(dict(row["pair_coverage_curve"]).get(key)) for row in positives]
                neg_coverage = [safe_float(dict(row["pair_coverage_curve"]).get(key)) for row in negatives]
                compaction_row = {
                    "mass_ratio": float(mass_ratio),
                    "corr": pearson(compaction_values, target_values),
                    "positive_gap": mean(pos_compaction) - mean(neg_compaction),
                }
                coverage_row = {
                    "mass_ratio": float(mass_ratio),
                    "corr": pearson(coverage_values, target_values),
                    "positive_gap": mean(pos_coverage) - mean(neg_coverage),
                }
                compaction_points.append(compaction_row)
                coverage_points.append(coverage_row)
                top_findings.append({"axis": axis, "channel": "compaction", "target_name": target_name, **compaction_row})
                top_findings.append({"axis": axis, "channel": "coverage", "target_name": target_name, **coverage_row})
            axis_block["continuous_compaction_corr"][target_name] = compaction_points
            axis_block["continuous_coverage_corr"][target_name] = coverage_points
            axis_block["best_points"][target_name] = {
                "compaction": max(compaction_points, key=lambda item: abs(safe_float(item["corr"]))),
                "coverage": max(coverage_points, key=lambda item: abs(safe_float(item["corr"]))),
            }
            axis_block["positive_bands"][target_name] = {
                "compaction": longest_positive_band(compaction_points),
                "coverage": longest_positive_band(coverage_points),
            }
        per_axis[axis] = axis_block
    top_findings.sort(key=lambda row: abs(safe_float(row["corr"])), reverse=True)
    return {
        "record_type": "stage56_pair_density_field_closure_summary",
        "joined_pair_count": len(joined_rows),
        "axis_row_count": len(rows),
        "mass_ratio_count": len(mass_ratios),
        "strict_positive_pair_ratio": mean([1.0 if bool(row.get("strict_positive_synergy")) else 0.0 for row in joined_rows]),
        "per_axis": per_axis,
        "top_abs_mass_point_correlations": top_findings[:24],
    }


def build_markdown(summary: Dict[str, object]) -> str:
    lines = [
        "# Pair 级连续密度场到闭包联立",
        "",
        f"- joined_pair_count: {summary['joined_pair_count']}",
        f"- axis_row_count: {summary['axis_row_count']}",
        f"- mass_ratio_count: {summary['mass_ratio_count']}",
        f"- strict_positive_pair_ratio: {safe_float(summary['strict_positive_pair_ratio']):.4f}",
        "",
        "## Top Mass-Point Correlations",
    ]
    for row in summary["top_abs_mass_point_correlations"]:
        lines.append(
            f"- {row['axis']} / {row['channel']} / {row['target_name']} / mass={safe_float(row['mass_ratio']):.2f}: "
            f"corr={safe_float(row['corr']):+.4f}, positive_gap={safe_float(row['positive_gap']):+.4f}"
        )
    lines.extend(["", "## Positive Bands"])
    for axis, block in dict(summary.get("per_axis", {})).items():
        lines.append(f"- {axis}")
        for target_name, target_block in dict(block.get("positive_bands", {})).items():
            compaction = dict(target_block.get("compaction", {}))
            coverage = dict(target_block.get("coverage", {}))
            lines.append(
                f"  - {target_name}: compaction_band={safe_float(compaction.get('start_mass_ratio')):.2f}-"
                f"{safe_float(compaction.get('end_mass_ratio')):.2f} (len={int(compaction.get('length', 0))}, "
                f"mean_corr={safe_float(compaction.get('mean_corr')):+.4f}), "
                f"coverage_band={safe_float(coverage.get('start_mass_ratio')):.2f}-"
                f"{safe_float(coverage.get('end_mass_ratio')):.2f} (len={int(coverage.get('length', 0))}, "
                f"mean_corr={safe_float(coverage.get('mean_corr')):+.4f})"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Lift pair frontier summaries into continuous density-field closure correlations")
    ap.add_argument("--pairs-json", required=True)
    ap.add_argument("--probe-json", action="append", required=True)
    ap.add_argument("--joined-rows", required=True)
    ap.add_argument("--output-dir", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tests/codex_temp/stage56_pair_density_field_closure_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(Path(args.pairs_json))
    term_payloads, mass_ratios = build_term_curve_payloads([Path(path) for path in args.probe_json], manifest)
    joined_rows = join_rows(Path(args.joined_rows), term_payloads, mass_ratios)
    summary = build_summary(joined_rows, mass_ratios)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "joined_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in joined_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "SUMMARY.md").write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps({"output_dir": out_dir.as_posix(), "joined_pair_count": len(joined_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
