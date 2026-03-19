from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from stage56_fullsample_regression_runner import fit_linear_regression, read_json, safe_float

ROOT = Path(__file__).resolve().parents[2]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def build_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        identity_margin = safe_float(row.get("identity_margin_term"))
        syntax_conflict = safe_float(row.get("syntax_constraint_conflict_term"))
        logic_fragile_bridge = safe_float(row.get("logic_fragile_bridge_term"))
        style_alignment = safe_float(row.get("style_alignment_term"))
        positive_core = mean([identity_margin, syntax_conflict])
        negative_core = mean([logic_fragile_bridge, style_alignment])
        stable_core_balance = positive_core - negative_core
        out.append(
            {
                **dict(row),
                "positive_core_term": positive_core,
                "negative_core_term": negative_core,
                "stable_core_balance": stable_core_balance,
            }
        )
    return out


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    feature_names = [
        "positive_core_term",
        "negative_core_term",
        "stable_core_balance",
    ]
    fits = [
        fit_linear_regression(rows, feature_names, "union_joint_adv"),
        fit_linear_regression(rows, feature_names, "union_synergy_joint"),
        fit_linear_regression(rows, feature_names, "strict_positive_synergy"),
    ]
    return {
        "record_type": "stage56_stable_core_compression_summary",
        "row_count": len(rows),
        "feature_names": feature_names,
        "mean_positive_core_term": mean([safe_float(row.get("positive_core_term")) for row in rows]),
        "mean_negative_core_term": mean([safe_float(row.get("negative_core_term")) for row in rows]),
        "mean_stable_core_balance": mean([safe_float(row.get("stable_core_balance")) for row in rows]),
        "fits": fits,
        "compressed_equation_text": (
            "U_core(pair) = b1 * positive_core + b2 * negative_core + b3 * stable_core_balance"
        ),
        "main_judgment": (
            "当前稳定核已经被压成正核、负核与核间边距三项，"
            "可以直接检查主方程是否开始从多变量收缩到更短的核结构。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 稳定核压缩摘要",
        "",
        f"- row_count: {summary.get('row_count', 0)}",
        f"- mean_positive_core_term: {safe_float(summary.get('mean_positive_core_term')):+.6f}",
        f"- mean_negative_core_term: {safe_float(summary.get('mean_negative_core_term')):+.6f}",
        f"- mean_stable_core_balance: {safe_float(summary.get('mean_stable_core_balance')):+.6f}",
        f"- compressed_equation_text: {summary.get('compressed_equation_text', '')}",
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
    ap = argparse.ArgumentParser(description="Compress the stable core terms into a shorter core equation")
    ap.add_argument(
        "--refit-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_master_equation_refit_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_stable_core_compression_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(read_json(Path(args.refit_rows_json)).get("rows", []))
    compressed_rows = build_rows(rows)
    summary = build_summary(compressed_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(json.dumps({"rows": compressed_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(compressed_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
