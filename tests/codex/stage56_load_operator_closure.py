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
        gs = safe_float(row.get("gs_load_channel_term"))
        sd = safe_float(row.get("sd_load_channel_term"))
        out.append(
            {
                **dict(row),
                "load_mean_term": (gs + sd) / 2.0,
                "load_contrast_term": (sd - gs) / 2.0,
                "load_abs_sum_term": abs(gs) + abs(sd),
            }
        )
    return out


def sign_of(value: float) -> str:
    if value > 1e-12:
        return "positive"
    if value < -1e-12:
        return "negative"
    return "neutral"


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    targets = ["union_joint_adv", "union_synergy_joint", "strict_positive_synergy"]
    names = ["load_mean_term", "load_contrast_term", "load_abs_sum_term"]
    fits = [fit_linear_regression(rows, names, target) for target in targets]
    sign_matrix = {
        feature: {
            fit["target_name"]: sign_of(safe_float(dict(fit["weights"]).get(feature)))
            for fit in fits
        }
        for feature in names
    }
    return {
        "record_type": "stage56_load_operator_closure_summary",
        "row_count": len(rows),
        "feature_names": names,
        "fits": fits,
        "sign_matrix": sign_matrix,
        "stable_negative_features": [
            feature
            for feature, target_signs in sign_matrix.items()
            if all(value == "negative" for value in target_signs.values())
        ],
        "strict_selective_features": [
            feature
            for feature, target_signs in sign_matrix.items()
            if target_signs.get("strict_positive_synergy") == "positive"
            and target_signs.get("union_joint_adv") == "negative"
            and target_signs.get("union_synergy_joint") == "negative"
        ],
        "operator_equations": {
            "load_base_operator": "L_base(gs, sd) = (gs + sd) / 2",
            "strict_select_operator": "L_select(gs, sd) = (sd - gs) / 2",
        },
        "main_judgment": (
            "gs 与 sd 可以继续压成两个更一般的算子："
            "基础负载算子 L_base 在三目标上稳定为负，"
            "严格选择算子 L_select 只在严格闭包目标上转成正。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 负载算子收口摘要",
        "",
        f"- row_count: {summary.get('row_count', 0)}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Operator Equations",
    ]
    for key, value in dict(summary.get("operator_equations", {})).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Stable Negative Features"])
    for feature in list(summary.get("stable_negative_features", [])):
        lines.append(f"- {feature}")
    lines.extend(["", "## Strict Selective Features"])
    for feature in list(summary.get("strict_selective_features", [])):
        lines.append(f"- {feature}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compress gs/sd into more general load operators")
    ap.add_argument(
        "--input-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_coupling_channel_canonicalization_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_load_operator_closure_20260320"),
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
