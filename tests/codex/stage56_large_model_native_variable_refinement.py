from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_model_native_variable_refinement_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_native_variable_summary() -> dict:
    extended = _load_json(ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_extended_continual_assessment.json")
    true_long = _load_json(ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_true_long_run_assessment.json")
    longterm = _load_json(ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_longterm_assessment.json")
    qwen_recovery = _load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_real_model_recovery_proxy_atlas_20260310.json")
    qwen_structure = _load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_real_model_structure_atlas_20260310.json")

    g_native = mean([
        extended.get("train_score", 0.0),
        extended.get("language_score", 0.0),
        true_long.get("long_run_score", 0.0),
        longterm.get("train_score", 0.0),
    ])
    s_native = mean([
        extended.get("theorem_score", 0.0),
        true_long.get("theorem_score", 0.0),
        longterm.get("theorem_score", 0.0),
        qwen_recovery["headline_metrics"].get("qwen_recovery_proxy_score", 0.0),
    ])
    l_base_native = mean([
        1.0 - extended.get("write_score", 0.0),
        1.0 - true_long.get("write_score", 0.0),
        1.0 - longterm.get("write_score", 0.0),
        qwen_structure["headline_metrics"].get("qwen_orientation_gap_abs", 0.0),
    ])
    l_select_native = mean([
        extended.get("recovery_depth", 0.0),
        extended.get("closure_bonus", 0.0),
        true_long.get("recovery_bonus", 0.0),
        qwen_recovery["headline_metrics"].get("qwen_task_side_gain", 0.0),
    ])

    summary = {
        "headline_metrics": {
            "G_native": g_native,
            "S_native": s_native,
            "L_base_native": l_base_native,
            "L_select_native": l_select_native,
            "native_balance": (g_native + s_native + l_select_native) - l_base_native,
        },
        "project_readout": {
            "summary": (
                "这一步把大模型代理量往更接近原生结构量推进。"
                "目标不是立刻得到最终闭式变量，而是减少只靠单一相关代理来定义 G/S/L_base/L_select 的风险。"
            ),
            "next_question": "下一步要看这些更原生结构量是否比旧代理在跨资产上更稳定。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 大模型原生变量细化报告",
        "",
        f"- G_native: {hm['G_native']:.6f}",
        f"- S_native: {hm['S_native']:.6f}",
        f"- L_base_native: {hm['L_base_native']:.6f}",
        f"- L_select_native: {hm['L_select_native']:.6f}",
        f"- native_balance: {hm['native_balance']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_native_variable_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
