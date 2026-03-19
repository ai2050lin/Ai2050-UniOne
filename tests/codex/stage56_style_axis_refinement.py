from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from stage56_fullsample_regression_runner import fit_linear_regression, read_json, read_jsonl, safe_float

ROOT = Path(__file__).resolve().parents[2]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def pair_key(row: Dict[str, object]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("model_id", "")),
        str(row.get("category", "")),
        str(row.get("prototype_term", "")),
        str(row.get("instance_term", "")),
    )


def sign_of(value: float) -> str:
    if value > 1e-12:
        return "positive"
    if value < -1e-12:
        return "negative"
    return "neutral"


def build_rows(
    pair_density_rows: Sequence[Dict[str, object]],
    complete_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    complete_group = {pair_key(row): dict(row) for row in complete_rows if dict(row).get("component_label") == "logic_prototype"}
    out: List[Dict[str, object]] = []
    for row in pair_density_rows:
        key = pair_key(row)
        axes = dict(row.get("axes", {}))
        style_block = dict(axes.get("style", {}))
        complete = complete_group.get(key, {})
        if not style_block or not complete:
            continue

        style_compaction_mid = safe_float(style_block.get("pair_compaction_middle_mean"))
        style_coverage_mid = safe_float(style_block.get("pair_coverage_middle_mean"))
        style_delta_l2 = safe_float(style_block.get("pair_delta_l2"))
        style_delta_mean_abs = safe_float(style_block.get("pair_delta_mean_abs"))
        style_role_align_compaction = safe_float(style_block.get("role_alignment_compaction"))
        style_role_align_coverage = safe_float(style_block.get("role_alignment_coverage"))

        out.append(
            {
                "model_id": key[0],
                "category": key[1],
                "prototype_term": key[2],
                "instance_term": key[3],
                "style_compaction_mid": style_compaction_mid,
                "style_coverage_mid": style_coverage_mid,
                "style_delta_l2": style_delta_l2,
                "style_delta_mean_abs": style_delta_mean_abs,
                "style_role_align_compaction": style_role_align_compaction,
                "style_role_align_coverage": style_role_align_coverage,
                "style_midfield": mean([style_compaction_mid, style_coverage_mid]),
                "style_alignment": mean([style_role_align_compaction, style_role_align_coverage]),
                "style_reorder_pressure": mean([style_delta_l2, style_delta_mean_abs]),
                "style_gap": style_coverage_mid - style_compaction_mid,
                "union_joint_adv": safe_float(row.get("union_joint_adv", complete.get("union_joint_adv"))),
                "union_synergy_joint": safe_float(row.get("union_synergy_joint", complete.get("union_synergy_joint"))),
                "strict_positive_synergy": 1.0 if bool(row.get("strict_positive_synergy", complete.get("strict_positive_synergy"))) else 0.0,
            }
        )
    return out


def feature_names() -> List[str]:
    return [
        "style_compaction_mid",
        "style_coverage_mid",
        "style_delta_l2",
        "style_delta_mean_abs",
        "style_role_align_compaction",
        "style_role_align_coverage",
        "style_midfield",
        "style_alignment",
        "style_reorder_pressure",
        "style_gap",
    ]


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    names = feature_names()
    fits = [
        fit_linear_regression(rows, names, "union_joint_adv"),
        fit_linear_regression(rows, names, "union_synergy_joint"),
        fit_linear_regression(rows, names, "strict_positive_synergy"),
    ]
    sign_matrix = {
        feature: {
            dict(fit).get("target_name", ""): sign_of(safe_float(dict(dict(fit).get("weights", {})).get(feature)))
            for fit in fits
        }
        for feature in names
    }
    stable_features: List[Dict[str, object]] = []
    for feature, targets in sign_matrix.items():
        signs = {sign for sign in targets.values() if sign != "neutral"}
        if len(signs) == 1 and signs:
            stable_features.append({"feature": feature, "sign": next(iter(signs))})
    return {
        "record_type": "stage56_style_axis_refinement_summary",
        "row_count": len(rows),
        "feature_names": names,
        "fits": fits,
        "sign_matrix": sign_matrix,
        "stable_features": stable_features,
        "main_judgment": (
            "style 已经从粗总项拆成中段、对齐、重排压力和差值等细通道，"
            "可以直接检查风格到底在哪些局部子通道上表现出稳定方向。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 style 细化摘要",
        "",
        f"- row_count: {summary.get('row_count', 0)}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Stable Features",
    ]
    for row in list(summary.get("stable_features", [])):
        row = dict(row)
        lines.append(f"- {row.get('feature', '')}: {row.get('sign', '')}")
    lines.extend(["", "## Fits"])
    for fit in list(summary.get("fits", [])):
        fit = dict(fit)
        lines.append(f"- target: {fit.get('target_name', '')}")
        for key, value in dict(fit.get("weights", {})).items():
            lines.append(f"  {key}: {safe_float(value):+.6f}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Refine style into finer sample-level regression channels")
    ap.add_argument(
        "--pair-density-jsonl",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_pair_density_tensor_field_20260319_1512" / "joined_rows.jsonl"),
    )
    ap.add_argument(
        "--complete-joined-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_complete_highdim_field_20260319_1645" / "joined_rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_style_axis_refinement_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pair_density_rows = read_jsonl(Path(args.pair_density_jsonl))
    complete_rows = list(read_json(Path(args.complete_joined_json)).get("rows", []))
    rows = build_rows(pair_density_rows, complete_rows)
    summary = build_summary(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
