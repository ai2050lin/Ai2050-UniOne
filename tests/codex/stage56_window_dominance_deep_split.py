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
        identity_margin = safe_float(row.get("identity_margin_term"))
        syntax_conflict = safe_float(row.get("syntax_constraint_conflict_term"))
        logic_fragile = safe_float(row.get("logic_fragile_bridge_term"))
        style_alignment = safe_float(row.get("style_alignment_term"))
        frontier = safe_float(row.get("frontier_term"))
        window_dominance = safe_float(row.get("window_dominance_term"))

        positive_core = 0.5 * (identity_margin + syntax_conflict)
        negative_core = 0.5 * (logic_fragile + style_alignment)

        out.append(
            {
                **dict(row),
                "window_identity_term": window_dominance * identity_margin,
                "window_syntax_term": window_dominance * syntax_conflict,
                "window_fragile_term": window_dominance * logic_fragile,
                "window_style_term": window_dominance * style_alignment,
                "window_frontier_term": window_dominance * frontier,
                "window_positive_core_term": window_dominance * positive_core,
                "window_negative_core_term": window_dominance * negative_core,
            }
        )
    return out


def feature_names() -> List[str]:
    return [
        "window_identity_term",
        "window_syntax_term",
        "window_fragile_term",
        "window_style_term",
        "window_frontier_term",
        "window_positive_core_term",
        "window_negative_core_term",
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
        "record_type": "stage56_window_dominance_deep_split_summary",
        "row_count": len(rows),
        "feature_names": names,
        "fits": fits,
        "sign_matrix": sign_matrix,
        "stable_features": stable_features,
        "main_judgment": (
            "窗口主导性已从单一总项继续拆成身份、句法、脆弱桥接、风格对齐、前沿，以及正核/负核耦合子项，"
            "可以直接判断它到底更像严格闭包门，还是负核放大器。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 窗口主导性深拆摘要",
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
    ap = argparse.ArgumentParser(description="Deep split window dominance into coupled subterms")
    ap.add_argument(
        "--refit-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_master_equation_refit_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_window_dominance_deep_split_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(read_json(Path(args.refit_rows_json)).get("rows", []))
    deep_rows = build_rows(rows)
    summary = build_summary(deep_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(json.dumps({"rows": deep_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(deep_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
