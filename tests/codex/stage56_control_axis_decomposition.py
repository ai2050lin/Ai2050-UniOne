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


def build_rows(
    pair_density_rows: Sequence[Dict[str, object]],
    complete_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    complete_group = {pair_key(row): dict(row) for row in complete_rows if dict(row).get("component_label") == "logic_prototype"}
    out: List[Dict[str, object]] = []
    for row in pair_density_rows:
        axes = dict(row.get("axes", {}))
        key = pair_key(row)
        complete = complete_group.get(key, {})
        if not axes or not complete:
            continue
        built = {
            "model_id": key[0],
            "category": key[1],
            "prototype_term": key[2],
            "instance_term": key[3],
            "union_joint_adv": safe_float(row.get("union_joint_adv", complete.get("union_joint_adv"))),
            "union_synergy_joint": safe_float(row.get("union_synergy_joint", complete.get("union_synergy_joint"))),
            "strict_positive_synergy": 1.0 if bool(row.get("strict_positive_synergy", complete.get("strict_positive_synergy"))) else 0.0,
        }
        for axis_name in ("style", "logic", "syntax"):
            block = dict(axes.get(axis_name, {}))
            built[f"{axis_name}_compaction_mid"] = safe_float(block.get("pair_compaction_middle_mean"))
            built[f"{axis_name}_coverage_mid"] = safe_float(block.get("pair_coverage_middle_mean"))
            built[f"{axis_name}_delta_l2"] = safe_float(block.get("pair_delta_l2"))
            built[f"{axis_name}_delta_mean_abs"] = safe_float(block.get("pair_delta_mean_abs"))
            built[f"{axis_name}_role_align_compaction"] = safe_float(block.get("role_alignment_compaction"))
            built[f"{axis_name}_role_align_coverage"] = safe_float(block.get("role_alignment_coverage"))
        out.append(built)
    return out


def feature_names() -> List[str]:
    names: List[str] = []
    for axis_name in ("style", "logic", "syntax"):
        names.extend(
            [
                f"{axis_name}_compaction_mid",
                f"{axis_name}_coverage_mid",
                f"{axis_name}_delta_l2",
                f"{axis_name}_delta_mean_abs",
                f"{axis_name}_role_align_compaction",
                f"{axis_name}_role_align_coverage",
            ]
        )
    return names


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    names = feature_names()
    fits = [
        fit_linear_regression(rows, names, "union_joint_adv"),
        fit_linear_regression(rows, names, "union_synergy_joint"),
        fit_linear_regression(rows, names, "strict_positive_synergy"),
    ]
    return {
        "record_type": "stage56_control_axis_decomposition_summary",
        "row_count": len(rows),
        "feature_names": names,
        "fits": fits,
        "mean_logic_compaction_mid": mean([safe_float(row.get("logic_compaction_mid")) for row in rows]),
        "mean_syntax_coverage_mid": mean([safe_float(row.get("syntax_coverage_mid")) for row in rows]),
        "main_judgment": (
            "控制轴已经从粗总项拆成中段压缩、覆盖、扰动强度和角色对齐等细通道，"
            "可以直接检查到底是哪类控制子通道在样本级推动或破坏闭包。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 控制轴分解回归摘要",
        "",
        f"- row_count: {summary.get('row_count', 0)}",
        f"- mean_logic_compaction_mid: {safe_float(summary.get('mean_logic_compaction_mid')):+.6f}",
        f"- mean_syntax_coverage_mid: {safe_float(summary.get('mean_syntax_coverage_mid')):+.6f}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Fits",
    ]
    for fit in list(summary.get("fits", [])):
        fit = dict(fit)
        lines.append(f"- target: {fit.get('target_name', '')}")
        for key, value in dict(fit.get("weights", {})).items():
            lines.append(f"  {key}: {safe_float(value):+.6f}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Decompose style / logic / syntax into finer regression channels")
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
        default=str(ROOT / "tests" / "codex_temp" / "stage56_control_axis_decomposition_20260319"),
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
