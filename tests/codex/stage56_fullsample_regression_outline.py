from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]


def build_outline() -> Dict[str, object]:
    feature_families: List[Dict[str, object]] = [
        {
            "family": "静态本体项",
            "features": [
                "atlas_static_hat",
                "offset_static_hat",
            ],
            "role": "提供家族身份和家族内偏移的样本级估计量",
        },
        {
            "family": "动态前沿项",
            "features": [
                "frontier_positive_corr",
                "frontier_negative_corr",
                "pair_compaction_middle_mean",
                "pair_coverage_middle_mean",
            ],
            "role": "提供高质量前沿压缩、分离和对比强度",
        },
        {
            "family": "内部子场项",
            "features": [
                "logic_prototype_score",
                "logic_fragile_bridge_score",
                "syntax_constraint_conflict_score",
            ],
            "role": "提供真正执行功能的机制级变量",
        },
        {
            "family": "窗口闭包项",
            "features": [
                "hidden_window_center",
                "mlp_window_center",
                "mean_union_synergy_joint",
                "strict_positive_synergy",
            ],
            "role": "提供词元窗口上的预收束与最终闭包结果",
        },
        {
            "family": "控制轴项",
            "features": [
                "style_control",
                "logic_control",
                "syntax_control",
            ],
            "role": "提供风格、逻辑、句法对主方程的调制项",
        },
    ]

    regression_targets = [
        "union_joint_adv",
        "union_synergy_joint",
        "strict_positive_synergy",
    ]

    return {
        "record_type": "stage56_fullsample_regression_outline_summary",
        "main_judgment": (
            "当前最需要推进的不是继续增加摘要层权重，而是把静态项、动态项、控制轴项一起下钻到样本级回归。"
        ),
        "feature_families": feature_families,
        "regression_targets": regression_targets,
        "proto_regression_equation": (
            "Y_pair = β_static * X_static + β_frontier * X_frontier + "
            "β_subfield * X_subfield + β_window * X_window + β_control * X_control + ε"
        ),
        "success_criteria": [
            "静态项在样本级不再退化成纯先验占位符",
            "控制轴项在样本级出现稳定符号",
            "窗口项对 strict_positive_synergy 的解释力高于摘要层平均量",
            "动态前沿项仍保持主导解释力",
        ],
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 全样本回归骨架摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        f"- proto_regression_equation: {summary.get('proto_regression_equation', '')}",
        "",
        "## Feature Families",
    ]
    for row in list(summary.get("feature_families", [])):
        row = dict(row)
        lines.append(
            f"- {row.get('family', '')}: {', '.join(row.get('features', []))} / {row.get('role', '')}"
        )
    lines.extend(["", "## Success Criteria"])
    for row in list(summary.get("success_criteria", [])):
        lines.append(f"- {row}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Outline the first full-sample regression plan for the unified master equation")
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_fullsample_regression_outline_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_outline()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "record_type": summary["record_type"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
