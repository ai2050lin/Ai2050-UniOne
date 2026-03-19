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
        logic_prototype = safe_float(row.get("logic_prototype_term"))
        identity_margin = safe_float(row.get("identity_margin_term"))
        frontier = safe_float(row.get("frontier_term"))
        syntax_conflict = safe_float(row.get("syntax_constraint_conflict_term"))
        window_dominance = safe_float(row.get("window_dominance_term"))
        style_alignment = safe_float(row.get("style_alignment_term"))
        style_midfield = safe_float(row.get("style_midfield_term"))

        out.append(
            {
                **dict(row),
                "logic_prototype_margin_term": logic_prototype * identity_margin,
                "logic_prototype_frontier_term": logic_prototype * frontier,
                "logic_prototype_syntax_term": logic_prototype * syntax_conflict,
                "window_dominance_style_alignment_term": window_dominance * style_alignment,
                "window_dominance_style_midfield_term": window_dominance * style_midfield,
                "window_dominance_frontier_term": window_dominance * frontier,
            }
        )
    return out


def feature_names() -> List[str]:
    return [
        "logic_prototype_margin_term",
        "logic_prototype_frontier_term",
        "logic_prototype_syntax_term",
        "window_dominance_style_alignment_term",
        "window_dominance_style_midfield_term",
        "window_dominance_frontier_term",
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
        "record_type": "stage56_mixed_term_split_summary",
        "row_count": len(rows),
        "feature_names": names,
        "fits": fits,
        "sign_matrix": sign_matrix,
        "stable_features": stable_features,
        "main_judgment": (
            "logic_prototype 与 window_dominance 已被拆成耦合子项，"
            "可以直接检查它们到底是通过身份边距、前沿还是风格子通道进入混合态。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 混合项拆分摘要",
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
    ap = argparse.ArgumentParser(description="Split mixed terms into interaction-style subterms")
    ap.add_argument(
        "--refit-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_master_equation_refit_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_mixed_term_split_20260319"),
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
