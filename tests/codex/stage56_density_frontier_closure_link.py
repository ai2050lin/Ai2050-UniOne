from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

from stage56_density_frontier_curve import infer_model_label, knee_mass_ratio, load_json, safe_float

AXES = ("style", "logic", "syntax")


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sx = sum(x * x for x in dx) ** 0.5
    sy = sum(y * y for y in dy) ** 0.5
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    return float(sum(a * b for a, b in zip(dx, dy)) / (sx * sy))


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def normalize_model_label(model_id: str) -> str:
    text = str(model_id).lower()
    if "deepseek" in text:
        return "DeepSeek-7B"
    if "qwen" in text:
        return "Qwen3-4B"
    if "glm" in text:
        return "GLM-4-9B"
    return str(model_id)


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


def decompose_axis_fields(row: Dict[str, object], axis: str) -> Dict[str, float]:
    axis_values = dict((row.get("axes") or {}).get(axis) or {})
    bridge = max(0.0, safe_float(axis_values.get("bridge_field_proxy")))
    conflict = max(0.0, safe_float(axis_values.get("conflict_field_proxy")))
    mismatch = max(0.0, safe_float(axis_values.get("mismatch_field_proxy")))
    joint_adv = safe_float(row.get("union_joint_adv"))
    synergy = safe_float(row.get("union_synergy_joint"))
    strict_positive = bool(row.get("strict_positive_synergy"))

    stable_bridge = bridge if joint_adv > 0.0 and synergy > 0.0 else 0.0
    fragile_bridge = bridge if bridge > 0.0 and stable_bridge == 0.0 else 0.0
    constraint_conflict = conflict if (synergy > 0.0 or strict_positive) else 0.0
    mismatch_exposure = mismatch if joint_adv > 0.0 and synergy >= 0.0 else 0.0

    return {
        "stable_bridge": stable_bridge,
        "fragile_bridge": fragile_bridge,
        "constraint_conflict": constraint_conflict,
        "mismatch_exposure": mismatch_exposure,
    }


def summarize_frontier_axis(probe: Dict[str, object], label: str) -> List[Dict[str, object]]:
    rows = []
    for axis, dim in (probe.get("dimensions") or {}).items():
        curve = list(dim.get("frontier_curve") or [])
        rows.append(
            {
                "model": label,
                "axis": axis,
                "mass10_compaction_ratio": safe_float(next((r["compaction_ratio"] for r in curve if safe_float(r["mass_ratio"]) == 0.10), 0.0)),
                "mass25_compaction_ratio": safe_float(next((r["compaction_ratio"] for r in curve if safe_float(r["mass_ratio"]) == 0.25), 0.0)),
                "frontier_sharpness_auc": auc_gap(curve),
                "knee_mass_ratio": knee_mass_ratio(curve),
                "pair_delta_cosine_mean": safe_float(dim.get("pair_delta_cosine_mean", 0.0)),
                "specific_selected_ratio": safe_float(dim.get("specific_selected_ratio", 0.0)),
            }
        )
    return rows


def summarize_closure_axis(rows: Sequence[Dict[str, object]], model_label: str, axis: str) -> Dict[str, object]:
    axis_rows = [row for row in rows if normalize_model_label(str(row.get("model_id"))) == model_label]
    synergy = [safe_float(row.get("union_synergy_joint")) for row in axis_rows]
    proto = [safe_float((row.get("axes") or {}).get(axis, {}).get("prototype_field_proxy")) for row in axis_rows]
    bridge = [safe_float((row.get("axes") or {}).get(axis, {}).get("bridge_field_proxy")) for row in axis_rows]
    conflict = [safe_float((row.get("axes") or {}).get(axis, {}).get("conflict_field_proxy")) for row in axis_rows]
    mismatch = [safe_float((row.get("axes") or {}).get(axis, {}).get("mismatch_field_proxy")) for row in axis_rows]

    decomposed = [decompose_axis_fields(row, axis) for row in axis_rows]
    bridge_total = sum(max(0.0, x) for x in bridge)
    conflict_total = sum(max(0.0, x) for x in conflict)
    mismatch_total = sum(max(0.0, x) for x in mismatch)

    return {
        "mean_prototype_field_proxy": mean(proto),
        "mean_bridge_field_proxy": mean(bridge),
        "mean_conflict_field_proxy": mean(conflict),
        "mean_mismatch_field_proxy": mean(mismatch),
        "corr_prototype_to_union_synergy": pearson(proto, synergy),
        "corr_bridge_to_union_synergy": pearson(bridge, synergy),
        "corr_conflict_to_union_synergy": pearson(conflict, synergy),
        "corr_mismatch_to_union_synergy": pearson(mismatch, synergy),
        "share_stable_bridge": (sum(x["stable_bridge"] for x in decomposed) / bridge_total) if bridge_total > 0.0 else 0.0,
        "share_fragile_bridge": (sum(x["fragile_bridge"] for x in decomposed) / bridge_total) if bridge_total > 0.0 else 0.0,
        "share_constraint_conflict": (sum(x["constraint_conflict"] for x in decomposed) / conflict_total) if conflict_total > 0.0 else 0.0,
        "share_mismatch_exposure": (sum(x["mismatch_exposure"] for x in decomposed) / mismatch_total) if mismatch_total > 0.0 else 0.0,
    }


def build_joint_rows(probes: Sequence[Path], joined_rows_path: Path) -> List[Dict[str, object]]:
    joined_rows = read_jsonl(joined_rows_path)
    rows: List[Dict[str, object]] = []
    for probe_path in probes:
        probe = load_json(probe_path)
        label = infer_model_label(probe, probe_path)
        frontier_rows = summarize_frontier_axis(probe, label)
        for frontier_row in frontier_rows:
            closure = summarize_closure_axis(joined_rows, label, str(frontier_row["axis"]))
            rows.append({**frontier_row, **closure})
    rows.sort(key=lambda row: (str(row["model"]), str(row["axis"])))
    return rows


def build_link_summary(joint_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    frontier_metrics = [
        "mass10_compaction_ratio",
        "mass25_compaction_ratio",
        "frontier_sharpness_auc",
        "knee_mass_ratio",
        "pair_delta_cosine_mean",
        "specific_selected_ratio",
    ]
    closure_metrics = [
        "mean_prototype_field_proxy",
        "mean_bridge_field_proxy",
        "mean_conflict_field_proxy",
        "mean_mismatch_field_proxy",
        "corr_prototype_to_union_synergy",
        "corr_bridge_to_union_synergy",
        "corr_conflict_to_union_synergy",
        "corr_mismatch_to_union_synergy",
        "share_stable_bridge",
        "share_fragile_bridge",
        "share_constraint_conflict",
        "share_mismatch_exposure",
    ]
    findings = []
    for left in frontier_metrics:
        xs = [safe_float(row[left]) for row in joint_rows]
        for right in closure_metrics:
            ys = [safe_float(row[right]) for row in joint_rows]
            findings.append(
                {
                    "frontier_metric": left,
                    "closure_metric": right,
                    "corr": pearson(xs, ys),
                }
            )
    findings.sort(key=lambda row: abs(safe_float(row["corr"])), reverse=True)
    return {
        "row_count": len(joint_rows),
        "joint_rows": list(joint_rows),
        "top_abs_correlations": findings[:12],
    }


def build_markdown(summary: Dict[str, object]) -> str:
    lines = [
        "# 密度前沿到闭包联立",
        "",
        f"- joint_rows: {summary['row_count']}",
        "",
        "## Top Correlations",
    ]
    for row in summary["top_abs_correlations"]:
        lines.append(
            f"- {row['frontier_metric']} -> {row['closure_metric']}: corr={row['corr']:+.4f}"
        )
    lines.extend(["", "## Model-Axis Rows"])
    for row in summary["joint_rows"]:
        lines.append(
            f"- {row['model']} / {row['axis']}: "
            f"mass10={row['mass10_compaction_ratio']:.4f}, "
            f"auc={row['frontier_sharpness_auc']:.4f}, "
            f"pair_cos={row['pair_delta_cosine_mean']:.4f}, "
            f"proto_corr={row['corr_prototype_to_union_synergy']:+.4f}, "
            f"bridge_corr={row['corr_bridge_to_union_synergy']:+.4f}, "
            f"conflict_corr={row['corr_conflict_to_union_synergy']:+.4f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-json", action="append", required=True)
    ap.add_argument("--joined-rows", required=True)
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        f"tests/codex_temp/stage56_density_frontier_closure_link_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    joint_rows = build_joint_rows([Path(x) for x in args.probe_json], Path(args.joined_rows))
    summary = build_link_summary(joint_rows)

    json_path = out_dir / "summary.json"
    md_path = out_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps({"summary_json": json_path.as_posix(), "summary_md": md_path.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
