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
        core_balance_v3 = (
            safe_float(row.get("positive_mass_v2_term"))
            - safe_float(row.get("destructive_core_term"))
            - safe_float(row.get("alignment_load_v2_term"))
        )
        style_gain = safe_float(row.get("style_structure_gain_term"))
        strict_load = safe_float(row.get("strict_load_term"))
        closed_form_kernel_v3 = core_balance_v3 + style_gain
        strict_kernel_v3 = closed_form_kernel_v3 + strict_load
        out.append(
            {
                **dict(row),
                "core_balance_v3_term": core_balance_v3,
                "closed_form_kernel_v3_term": closed_form_kernel_v3,
                "strict_kernel_v3_term": strict_kernel_v3,
            }
        )
    return out


def feature_names() -> List[str]:
    return [
        "core_balance_v3_term",
        "strict_load_term",
        "style_structure_gain_term",
        "closed_form_kernel_v3_term",
        "strict_kernel_v3_term",
    ]


def sign_of(value: float) -> str:
    if value > 1e-12:
        return "positive"
    if value < -1e-12:
        return "negative"
    return "neutral"


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    names = feature_names()
    targets = ["union_joint_adv", "union_synergy_joint", "strict_positive_synergy"]
    fits = [fit_linear_regression(rows, names, target) for target in targets]
    sign_matrix = {
        feature: {
            target: sign_of(safe_float(dict(fit["weights"]).get(feature)))
            for fit, target in zip(fits, targets)
        }
        for feature in names
    }
    stable_features: List[Dict[str, object]] = []
    for feature, target_signs in sign_matrix.items():
        signs = {value for value in target_signs.values() if value != "neutral"}
        if len(signs) == 1 and signs:
            stable_features.append({"feature": feature, "sign": next(iter(signs))})
    return {
        "record_type": "stage56_closed_form_final_kernel_summary",
        "row_count": len(rows),
        "feature_names": names,
        "fits": fits,
        "sign_matrix": sign_matrix,
        "stable_features": stable_features,
        "equation_text": (
            "core_balance_v3 = positive_mass_v2 - destructive_core - alignment_load_v2; "
            "closed_form_kernel_v3 = core_balance_v3 + style_structure_gain; "
            "strict_kernel_v3 = closed_form_kernel_v3 + strict_load"
        ),
        "main_judgment": (
            "闭式核已进一步压成核心平衡、严格负载和风格微修正三层，"
            "现在可以直接比较一般闭包核和严格闭包核是否需要分成两条主式。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 闭式核最终候选摘要",
        "",
        f"- row_count: {summary.get('row_count', 0)}",
        f"- equation_text: {summary.get('equation_text', '')}",
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
    ap = argparse.ArgumentParser(description="Refit a shorter closed-form kernel candidate")
    ap.add_argument(
        "--input-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_destructive_negative_resplit_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_closed_form_final_kernel_20260319"),
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
