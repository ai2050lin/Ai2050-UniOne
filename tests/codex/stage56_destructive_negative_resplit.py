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
        destructive_core = safe_float(row.get("logic_fragile_bridge_term"))
        strict_load = (
            safe_float(row.get("frontier_negative_base_term"))
            + safe_float(row.get("window_gate_negative_term"))
        )
        destructive_alignment = destructive_core + safe_float(row.get("alignment_load_v2_term"))
        out.append(
            {
                **dict(row),
                "destructive_core_term": destructive_core,
                "strict_load_term": strict_load,
                "destructive_alignment_term": destructive_alignment,
            }
        )
    return out


def feature_names() -> List[str]:
    return [
        "destructive_core_term",
        "strict_load_term",
        "destructive_alignment_term",
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
        "record_type": "stage56_destructive_negative_resplit_summary",
        "row_count": len(rows),
        "feature_names": names,
        "fits": fits,
        "sign_matrix": sign_matrix,
        "stable_features": stable_features,
        "equation_text": (
            "destructive_core = logic_fragile_bridge; "
            "strict_load = frontier_negative_base + window_gate_negative; "
            "destructive_alignment = destructive_core + alignment_load_v2"
        ),
        "main_judgment": (
            "破坏负荷已重新拆成真破坏核、严格闭包负载和带对齐负荷的破坏组合，"
            "当前最关键的问题是 strict_positive_synergy 上翻正的到底是 strict_load 还是 destructive_core。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 破坏负荷再拆摘要",
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
    ap = argparse.ArgumentParser(description="Resplit destructive negative into core and strict load")
    ap.add_argument(
        "--input-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_control_axis_closed_form_integration_20260319" / "rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_destructive_negative_resplit_20260319"),
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
