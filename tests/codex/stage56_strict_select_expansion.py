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
        union_adv = safe_float(row.get("union_joint_adv"))
        union_syn = safe_float(row.get("union_synergy_joint"))
        strict = safe_float(row.get("strict_positive_synergy"))
        general_mean = (union_adv + union_syn) / 2.0
        out.append(
            {
                **dict(row),
                "strictness_delta_vs_union": strict - union_adv,
                "strictness_delta_vs_synergy": strict - union_syn,
                "strictness_delta_vs_mean": strict - general_mean,
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
    feature_names = ["load_contrast_term", "load_mean_term"]
    targets = [
        "strict_positive_synergy",
        "strictness_delta_vs_union",
        "strictness_delta_vs_synergy",
        "strictness_delta_vs_mean",
    ]
    fits = [fit_linear_regression(rows, feature_names, target) for target in targets]
    sign_matrix = {
        feature: {
            fit["target_name"]: sign_of(safe_float(dict(fit["weights"]).get(feature)))
            for fit in fits
        }
        for feature in feature_names
    }
    stable_strict_features = [
        feature
        for feature, target_signs in sign_matrix.items()
        if all(target_signs.get(target) == "positive" for target in targets)
    ]
    return {
        "record_type": "stage56_strict_select_expansion_summary",
        "row_count": len(rows),
        "feature_names": feature_names,
        "targets": targets,
        "fits": fits,
        "sign_matrix": sign_matrix,
        "stable_strict_features": stable_strict_features,
        "main_judgment": (
            "这一轮专门检查严格选择算子是否能扩展到更多严格性目标。"
            "如果 load_contrast 在这些目标上持续为正，它就不再只是单一严格目标的特例。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 严格选择算子扩展摘要",
        "",
        f"- row_count: {summary.get('row_count', 0)}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Stable Strict Features",
    ]
    for feature in list(summary.get("stable_strict_features", [])):
        lines.append(f"- {feature}")
    lines.extend(["", "## Sign Matrix"])
    for feature, target_signs in dict(summary.get("sign_matrix", {})).items():
        lines.append(f"- {feature}: {json.dumps(target_signs, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Expand strict select operator to more strictness targets")
    ap.add_argument(
        "--input-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_load_operator_closure_20260320" / "rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_strict_select_expansion_20260320"),
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
