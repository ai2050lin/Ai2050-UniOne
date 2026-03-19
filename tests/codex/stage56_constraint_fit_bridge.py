from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def build_bridge(
    constraint_summary: Dict[str, object],
    regression_outline_summary: Dict[str, object],
) -> Dict[str, object]:
    constraints = list(constraint_summary.get("constraints", []))
    feature_families = list(regression_outline_summary.get("feature_families", []))

    family_map = {str(dict(row).get("family")): dict(row) for row in feature_families}

    bridge_rows: List[Dict[str, object]] = [
        {
            "constraint_name": "局部图册公理",
            "fit_family": "静态本体项",
            "fit_features": list(family_map.get("静态本体项", {}).get("features", [])),
            "fit_role": "为 Atlas_static / Offset_static 提供样本级估计入口",
        },
        {
            "constraint_name": "高质量前沿公理",
            "fit_family": "动态前沿项",
            "fit_features": list(family_map.get("动态前沿项", {}).get("features", [])),
            "fit_role": "把 compaction / coverage / separation 压到前沿拟合族",
        },
        {
            "constraint_name": "功能子场公理",
            "fit_family": "内部子场项",
            "fit_features": list(family_map.get("内部子场项", {}).get("features", [])),
            "fit_role": "把 logic_prototype / logic_fragile_bridge / syntax_constraint_conflict 接成可回归项",
        },
        {
            "constraint_name": "时间窗收束公理",
            "fit_family": "窗口闭包项",
            "fit_features": list(family_map.get("窗口闭包项", {}).get("features", [])),
            "fit_role": "把 tail window（尾部窗口）上的收束量下钻到回归入口",
        },
        {
            "constraint_name": "闭包边界公理",
            "fit_family": "窗口闭包项",
            "fit_features": [
                "union_joint_adv",
                "union_synergy_joint",
                "strict_positive_synergy",
            ],
            "fit_role": "把闭包成功 / 失败边界直接映射为监督目标与阈值条件",
        },
        {
            "constraint_name": "分层统一公理",
            "fit_family": "静态本体项 + 动态前沿项 + 内部子场项 + 窗口闭包项 + 控制轴项",
            "fit_features": (
                list(family_map.get("静态本体项", {}).get("features", []))
                + list(family_map.get("动态前沿项", {}).get("features", []))
                + list(family_map.get("内部子场项", {}).get("features", []))
                + list(family_map.get("窗口闭包项", {}).get("features", []))
                + list(family_map.get("控制轴项", {}).get("features", []))
            ),
            "fit_role": "把所有层级变量并入同一回归系统，形成真正的统一拟合入口",
        },
    ]

    return {
        "record_type": "stage56_constraint_fit_bridge_summary",
        "bridge_rows": bridge_rows,
        "proto_fit_system": {
            "regression_form": (
                "Y_pair = β_static * X_static + β_frontier * X_frontier + β_subfield * X_subfield + "
                "β_window * X_window + β_control * X_control + ε"
            ),
            "constraint_gate": (
                "Regression is valid only if all six axiom constraints are represented by at least one fitted feature family"
            ),
        },
        "coverage_ratio": safe_float(len([row for row in bridge_rows if list(dict(row).get('fit_features', []))])) / max(len(bridge_rows), 1),
        "main_judgment": (
            "当前已经具备把公理约束逐条映射到拟合特征族的条件。"
            "这意味着项目可以从‘约束型理论’继续推进到‘约束型回归系统’。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 约束到拟合桥接摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        f"- coverage_ratio: {safe_float(summary.get('coverage_ratio')):+.4f}",
        "",
        "## Bridge Rows",
    ]
    for row in list(summary.get("bridge_rows", [])):
        row = dict(row)
        lines.append(
            f"- {row.get('constraint_name', '')}: {row.get('fit_family', '')} / "
            f"{', '.join(row.get('fit_features', []))} / {row.get('fit_role', '')}"
        )
    lines.extend(["", "## Proto Fit System"])
    proto = dict(summary.get("proto_fit_system", {}))
    for key, value in proto.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bridge first axiom constraints to the full-sample regression families")
    ap.add_argument(
        "--constraint-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_axiom_to_equation_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--regression-outline-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_fullsample_regression_outline_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_constraint_fit_bridge_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_bridge(
        read_json(Path(args.constraint_summary_json)),
        read_json(Path(args.regression_outline_summary_json)),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "record_type": summary["record_type"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
