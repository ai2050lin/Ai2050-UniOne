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
        destructive_negative_term = (
            safe_float(row.get("logic_fragile_bridge_term"))
            + safe_float(row.get("frontier_negative_base_term"))
            + safe_float(row.get("window_gate_negative_term"))
        )
        alignment_load_term = safe_float(row.get("style_alignment_term"))
        negative_mass_rebalanced = destructive_negative_term - alignment_load_term
        out.append(
            {
                **dict(row),
                "destructive_negative_term": destructive_negative_term,
                "alignment_load_term": alignment_load_term,
                "negative_mass_rebalanced_term": negative_mass_rebalanced,
            }
        )
    return out


def feature_names() -> List[str]:
    return [
        "destructive_negative_term",
        "alignment_load_term",
        "negative_mass_rebalanced_term",
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
        "record_type": "stage56_negative_mass_deep_split_summary",
        "row_count": len(rows),
        "feature_names": names,
        "fits": fits,
        "sign_matrix": sign_matrix,
        "stable_features": stable_features,
        "main_judgment": (
            "负质量已被继续拆成破坏性负荷、对齐负荷和重平衡负质量，"
            "可以直接判断 strict_positive_synergy 中翻正的到底是哪一部分。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 负质量深拆摘要",
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
    ap = argparse.ArgumentParser(description="Deep split negative mass into destructive and alignment loads")
    ap.add_argument(
        "--refit-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_frontier_migration_master_refit_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_negative_mass_deep_split_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(read_json(Path(args.refit_rows_json)).get("rows", []))
    split_rows = build_rows(rows)
    summary = build_summary(split_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(json.dumps({"rows": split_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(split_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
