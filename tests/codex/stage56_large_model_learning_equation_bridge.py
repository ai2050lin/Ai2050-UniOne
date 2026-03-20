from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_model_learning_equation_bridge_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_large_model_learning_equation_summary() -> dict:
    alignment = _load_json(ROOT / "tests" / "codex_temp" / "stage56_large_model_long_horizon_alignment_20260320" / "summary.json")
    stability = _load_json(ROOT / "tests" / "codex_temp" / "stage56_large_model_long_horizon_stability_20260320" / "summary.json")
    formula = _load_json(ROOT / "tests" / "codex_temp" / "stage56_large_model_formula_validation_20260320" / "summary.json")

    frontier_drive = 1.0 / max(alignment["headline_metrics"]["frontier_mean_step"], 1e-6)
    boundary_drive = 1.0 / max(alignment["headline_metrics"]["boundary_mean_step"], 1e-6)
    atlas_drive = 1.0 / max(alignment["headline_metrics"]["atlas_mean_step"], 1e-6)

    stability_term = stability["headline_metrics"]["stability_mean"]
    plasticity_term = stability["headline_metrics"]["plasticity_mean"]
    risk_term = stability["headline_metrics"]["risk_mean"]

    g_term = formula["headline_metrics"]["g_proxy"]
    lbase_term = formula["headline_metrics"]["l_base_proxy"]
    lselect_term = formula["headline_metrics"]["l_select_proxy"]

    summary = {
        "headline_metrics": {
            "atlas_learning_drive_large": atlas_drive * stability_term,
            "frontier_learning_drive_large": frontier_drive * plasticity_term,
            "boundary_learning_drive_large": boundary_drive * (plasticity_term + stability_term - risk_term),
            "large_formula_support": mean([g_term, lbase_term, lselect_term]),
            "ordering_support": alignment["headline_metrics"]["ordered_case_ratio"],
        },
        "project_readout": {
            "summary": (
                "这一步把长程阶段顺序、大模型长期稳定性和 G/L_base/L_select 主结构并回同一条学习方程桥接。"
                "目标是判断更大模型口径下，前沿、边界、图册三类学习驱动是否仍保持不同时间尺度。"
            ),
            "next_question": "下一步要把这些大模型驱动量和当前小原型的学习方程第二版直接比较，检查是否存在跨规模同构。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 大模型学习方程桥接报告",
        "",
        f"- atlas_learning_drive_large: {hm['atlas_learning_drive_large']:.6f}",
        f"- frontier_learning_drive_large: {hm['frontier_learning_drive_large']:.6f}",
        f"- boundary_learning_drive_large: {hm['boundary_learning_drive_large']:.6f}",
        f"- large_formula_support: {hm['large_formula_support']:.6f}",
        f"- ordering_support: {hm['ordering_support']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_model_learning_equation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
