from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_model_long_horizon_stability_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_long_horizon_stability_summary() -> dict:
    true_long_assess = _load_json(ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_true_long_run_assessment.json")
    longterm_assess = _load_json(ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_longterm_assessment.json")
    persistent_assess = _load_json(ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_persistent_assessment.json")
    extended_assess = _load_json(ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_extended_continual_assessment.json")
    recovery_chain = _load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_online_recovery_chain_20260310.json")

    rows = [
        {
            "name": "openwebtext_true_long",
            "plasticity": true_long_assess["online_score"],
            "stability": true_long_assess["read_score"] * true_long_assess["rollback_score"],
            "risk": 1.0 - true_long_assess["write_score"],
        },
        {
            "name": "openwebtext_longterm",
            "plasticity": longterm_assess["online_score"],
            "stability": longterm_assess["read_score"] * longterm_assess["rollback_score"],
            "risk": 1.0 - longterm_assess["write_score"],
        },
        {
            "name": "openwebtext_persistent",
            "plasticity": persistent_assess["online_score"],
            "stability": persistent_assess["read_score"] * persistent_assess["rollback_score"],
            "risk": 1.0 - persistent_assess["write_score"],
        },
        {
            "name": "openwebtext_extended",
            "plasticity": extended_assess["online_score"],
            "stability": extended_assess["read_score"] * extended_assess["rollback_score"],
            "risk": 1.0 - extended_assess["write_score"],
        },
        {
            "name": "qwen3_4b_recovery_chain",
            "plasticity": recovery_chain["gains"]["qwen_online_recovery_gain"],
            "stability": recovery_chain["models"]["qwen3_4b"]["systems"]["online_recovery_aware"]["mean_post_recovery_stability"],
            "risk": recovery_chain["models"]["qwen3_4b"]["systems"]["online_recovery_aware"]["rollback_trigger_rate"],
        },
        {
            "name": "deepseek_7b_recovery_chain",
            "plasticity": recovery_chain["gains"]["deepseek_online_recovery_gain"],
            "stability": recovery_chain["models"]["deepseek_7b"]["systems"]["online_recovery_aware"]["mean_post_recovery_stability"],
            "risk": recovery_chain["models"]["deepseek_7b"]["systems"]["online_recovery_aware"]["rollback_trigger_rate"],
        },
    ]

    summary = {
        "case_count": len(rows),
        "rows": rows,
        "headline_metrics": {
            "plasticity_mean": mean([r["plasticity"] for r in rows]),
            "stability_mean": mean([r["stability"] for r in rows]),
            "risk_mean": mean([r["risk"] for r in rows]),
            "best_balance_case": max(rows, key=lambda r: r["plasticity"] * r["stability"] - r["risk"])["name"],
        },
        "project_readout": {
            "summary": (
                "这一步把更长期的 openwebtext 训练评估块和真实模型在线恢复链并到同一长期稳定性坐标。"
                "目标是判断大模型长期在线下，是否仍然存在塑性、稳定与风险三者之间的系统张力。"
            ),
            "next_question": "下一步要把这些长期稳定性量并回当前 G/S/L_base/L_select 的学习方程。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    lines = [
        "# 大模型长期在线稳定性报告",
        "",
        f"- 案例数: {summary['case_count']}",
        f"- 平均塑性: {summary['headline_metrics']['plasticity_mean']:.6f}",
        f"- 平均稳定性: {summary['headline_metrics']['stability_mean']:.6f}",
        f"- 平均风险: {summary['headline_metrics']['risk_mean']:.6f}",
        f"- 最佳平衡案例: {summary['headline_metrics']['best_balance_case']}",
        "",
        "## 个案",
    ]
    for row in summary["rows"]:
        lines.append(
            f"- {row['name']}: plasticity={row['plasticity']:.6f}, "
            f"stability={row['stability']:.6f}, risk={row['risk']:.6f}"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_long_horizon_stability_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
