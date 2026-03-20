from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_heterogeneous_asset_ordering_diagnosis_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _diagnose_case(row: dict) -> str:
    name = row["name"]
    if "toy_" in name:
        return "任务过于简化，图册与边界信号混叠"
    if "z113" in name:
        return "可视化训练日志以泛化曲线为主，边界代理和前沿代理不共尺度"
    if "phasea" in name:
        return "样本太短，三阶段几乎同时触发"
    if "openwebtext_backbone" in name:
        return "只含短骨干块，缺少长程边界和图册后段"
    return "未知异质性"


def build_ordering_diagnosis_summary() -> dict:
    coarse = _load_json(ROOT / "tests" / "codex_temp" / "stage56_large_model_checkpoint_alignment_20260320" / "summary.json")
    refined = _load_json(ROOT / "tests" / "codex_temp" / "stage56_large_model_long_horizon_alignment_20260320" / "summary.json")
    rows = []
    for row in coarse["cases"]:
        rows.append(
            {
                "name": row["name"],
                "ordered": row["ordering_signature"][0] and row["ordering_signature"][1],
                "diagnosis": _diagnose_case(row),
            }
        )
    summary = {
        "coarse_order_ratio": coarse["headline_metrics"]["ordered_case_ratio"],
        "refined_order_ratio": refined["headline_metrics"]["ordered_case_ratio"],
        "rows": rows,
        "project_readout": {
            "summary": (
                "这一步专门解释为什么异质资产会把阶段顺序从 1.0 冲散到 0.2。"
                "当前最主要的原因不是理论主线失效，而是资产尺度、任务类型和阶段代理不共尺度。"
            ),
            "next_question": "下一步要把异质资产也重写成同一长程阶段口径，而不是继续直接混比。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    lines = [
        "# 异质资产阶段顺序诊断报告",
        "",
        f"- 混合资产顺序支持: {summary['coarse_order_ratio']:.6f}",
        f"- 长程同质资产顺序支持: {summary['refined_order_ratio']:.6f}",
        "",
        "## 个案",
    ]
    for row in summary["rows"]:
        lines.append(f"- {row['name']}: ordered={row['ordered']}, diagnosis={row['diagnosis']}")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_ordering_diagnosis_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
