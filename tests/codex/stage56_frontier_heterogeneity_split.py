from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from stage56_fullsample_regression_runner import fit_linear_regression, read_jsonl, safe_float

ROOT = Path(__file__).resolve().parents[2]


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def build_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        axes = dict(row.get("axes", {}))
        compaction_mid = mean([safe_float(dict(axes.get(axis, {})).get("pair_compaction_middle_mean")) for axis in ("style", "logic", "syntax")])
        coverage_mid = mean([safe_float(dict(axes.get(axis, {})).get("pair_coverage_middle_mean")) for axis in ("style", "logic", "syntax")])
        separation_term = mean(
            [
                safe_float(dict(axes.get(axis, {})).get("role_asymmetry_compaction_l1"))
                + safe_float(dict(axes.get(axis, {})).get("role_asymmetry_coverage_l1"))
                for axis in ("style", "logic", "syntax")
            ]
        )
        compaction_late_shift = mean(
            [
                safe_float(dict(axes.get(axis, {})).get("pair_compaction_late_mean"))
                - safe_float(dict(axes.get(axis, {})).get("pair_compaction_early_mean"))
                for axis in ("style", "logic", "syntax")
            ]
        )
        coverage_late_shift = mean(
            [
                safe_float(dict(axes.get(axis, {})).get("pair_coverage_late_mean"))
                - safe_float(dict(axes.get(axis, {})).get("pair_coverage_early_mean"))
                for axis in ("style", "logic", "syntax")
            ]
        )
        balance_term = coverage_mid - compaction_mid
        out.append(
            {
                "model_id": str(row.get("model_id", "")),
                "category": str(row.get("category", "")),
                "prototype_term": str(row.get("prototype_term", "")),
                "instance_term": str(row.get("instance_term", "")),
                "frontier_compaction_term": compaction_mid,
                "frontier_coverage_term": coverage_mid,
                "frontier_separation_term": separation_term,
                "frontier_compaction_late_shift": compaction_late_shift,
                "frontier_coverage_late_shift": coverage_late_shift,
                "frontier_balance_term": balance_term,
                "union_joint_adv": safe_float(row.get("union_joint_adv")),
                "union_synergy_joint": safe_float(row.get("union_synergy_joint")),
                "strict_positive_synergy": 1.0 if bool(row.get("strict_positive_synergy")) else 0.0,
            }
        )
    return out


def feature_names() -> List[str]:
    return [
        "frontier_compaction_term",
        "frontier_coverage_term",
        "frontier_separation_term",
        "frontier_compaction_late_shift",
        "frontier_coverage_late_shift",
        "frontier_balance_term",
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
        "record_type": "stage56_frontier_heterogeneity_split_summary",
        "row_count": len(rows),
        "feature_names": names,
        "fits": fits,
        "sign_matrix": sign_matrix,
        "stable_features": stable_features,
        "main_judgment": (
            "前沿层已从单一混合项继续拆成压缩、覆盖、分离、晚移和覆盖减压缩差值，"
            "可以直接判断前沿到底是通过哪类机制进入目标分裂。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 前沿异质性拆分摘要",
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
    ap = argparse.ArgumentParser(description="Split frontier heterogeneity into compaction, coverage, separation and shift terms")
    ap.add_argument(
        "--pair-density-jsonl",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_pair_density_tensor_field_20260319_1512" / "joined_rows.jsonl"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_frontier_heterogeneity_split_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pair_density_rows = read_jsonl(Path(args.pair_density_jsonl))
    rows = build_rows(pair_density_rows)
    summary = build_summary(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
