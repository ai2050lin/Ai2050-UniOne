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
        positive_mass = (
            safe_float(row.get("identity_margin_term"))
            + safe_float(row.get("syntax_constraint_conflict_term"))
            + safe_float(row.get("frontier_positive_migration_term"))
            + safe_float(row.get("window_gate_positive_term"))
        )
        destructive_negative = safe_float(row.get("destructive_negative_term"))
        alignment_load = safe_float(row.get("alignment_load_term"))
        closed_form_balance_v2 = positive_mass - destructive_negative
        strict_balance_v2 = positive_mass - destructive_negative + alignment_load
        out.append(
            {
                **dict(row),
                "positive_mass_v2_term": positive_mass,
                "destructive_negative_v2_term": destructive_negative,
                "alignment_load_v2_term": alignment_load,
                "closed_form_balance_v2_term": closed_form_balance_v2,
                "strict_balance_v2_term": strict_balance_v2,
            }
        )
    return out


def feature_names() -> List[str]:
    return [
        "positive_mass_v2_term",
        "destructive_negative_v2_term",
        "alignment_load_v2_term",
        "closed_form_balance_v2_term",
        "strict_balance_v2_term",
    ]


def sign_of(value: float) -> str:
    if value > 1e-12:
        return "positive"
    if value < -1e-12:
        return "negative"
    return "neutral"


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
        "record_type": "stage56_closed_form_kernel_refit_summary",
        "row_count": len(rows),
        "feature_names": names,
        "fits": fits,
        "sign_matrix": sign_matrix,
        "stable_features": stable_features,
        "equation_text": (
            "C_v2(pair) = positive_mass - destructive_negative; "
            "S_v2(pair) = positive_mass - destructive_negative + alignment_load"
        ),
        "main_judgment": (
            "闭式核已被重写成一般闭包边距和严格闭包边距两条核，"
            "可以直接判断是否需要把对齐负荷从负质量中单独拆出。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 闭式核重拟合摘要",
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
    ap = argparse.ArgumentParser(description="Refit closed-form kernel with destructive negative and alignment load split")
    ap.add_argument(
        "--negative-split-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_negative_mass_deep_split_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_closed_form_kernel_refit_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(read_json(Path(args.negative_split_rows_json)).get("rows", []))
    refit_rows = build_rows(rows)
    summary = build_summary(refit_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(json.dumps({"rows": refit_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(refit_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
