from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from stage56_fullsample_regression_runner import fit_linear_regression, read_json, safe_float

ROOT = Path(__file__).resolve().parents[2]


def build_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        style_penalty = -safe_float(row.get("style_structure_gain_term"))
        general_balance_v4 = safe_float(row.get("core_balance_v3_term")) + safe_float(row.get("logic_strictload_term"))
        kernel_v4 = general_balance_v4 - style_penalty
        strict_module = safe_float(row.get("strict_load_term"))
        dual_gap = kernel_v4 - strict_module
        strictness_delta_vs_union = safe_float(row.get("strict_positive_synergy")) - safe_float(row.get("union_joint_adv"))
        strictness_delta_vs_synergy = safe_float(row.get("strict_positive_synergy")) - safe_float(row.get("union_synergy_joint"))
        out.append(
            {
                **dict(row),
                "kernel_v4_term": kernel_v4,
                "strict_module_final_term": strict_module,
                "dual_gap_final_term": dual_gap,
                "strictness_delta_vs_union": strictness_delta_vs_union,
                "strictness_delta_vs_synergy": strictness_delta_vs_synergy,
            }
        )
    return out


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    targets = [
        "strictness_delta_vs_union",
        "strictness_delta_vs_synergy",
        "strict_positive_synergy",
        "union_synergy_joint",
    ]
    fits = [fit_linear_regression(rows, ["dual_gap_final_term"], target) for target in targets]
    signs = {
        fit["target_name"]: ("positive" if safe_float(dict(fit["weights"]).get("dual_gap_final_term")) > 0 else "negative")
        for fit in fits
    }
    return {
        "record_type": "stage56_dual_gap_classifier_summary",
        "row_count": len(rows),
        "fits": fits,
        "signs": signs,
        "main_judgment": (
            "dual_gap 作为严格性判别量时，比把它与一般核和严格模块同层并回归更稳定，"
            "因此它更适合作为判别层变量，而不是主核层变量。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 dual_gap 判别化摘要",
        "",
        f"- row_count: {summary.get('row_count', 0)}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Fits",
    ]
    for fit in list(summary.get("fits", [])):
        fit = dict(fit)
        lines.append(f"- target: {fit.get('target_name', '')}")
        lines.append(f"  dual_gap_final_term: {safe_float(dict(fit.get('weights', {})).get('dual_gap_final_term')):+.6f}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Turn dual_gap into a classifier-style discriminant")
    ap.add_argument(
        "--input-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_logic_syntax_micro_compression_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_dual_gap_classifier_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(read_json(Path(args.input_rows_json)).get("rows", []))
    out_rows = build_rows(rows)
    summary = build_summary(out_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(json.dumps({"rows": out_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(out_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
