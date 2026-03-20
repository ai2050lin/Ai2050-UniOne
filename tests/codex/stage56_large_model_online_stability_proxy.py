from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_model_online_stability_proxy_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_large_model_stability_summary() -> dict:
    openweb_block = _load_json(ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_real_training_curve_block.json")
    online_chain = _load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_online_recovery_chain_20260310.json")
    stage_heads = _load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_online_learnable_stage_heads_20260310.json")
    long_suite = _load_json(ROOT / "tests" / "codex_temp" / "stage56_long_context_online_language_suite_20260320" / "summary.json")

    openweb_online = openweb_block.get("online_rounds", [])
    openweb_plasticity = mean([row.get("delta", 0.0) for row in openweb_online]) if openweb_online else 0.0
    openweb_stability = mean([row.get("stable_read", 0.0) for row in openweb_online]) if openweb_online else 0.0
    openweb_write = mean([row.get("online_write_scale", 0.0) for row in openweb_online]) if openweb_online else 0.0

    qwen = online_chain["models"]["qwen3_4b"]["systems"]["online_recovery_aware"]
    deepseek = online_chain["models"]["deepseek_7b"]["systems"]["online_recovery_aware"]
    qwen_heads = stage_heads["models"]["qwen3_4b"]["online_learnable_stage_heads"]
    deepseek_heads = stage_heads["models"]["deepseek_7b"]["online_learnable_stage_heads"]

    long_ctx = long_suite["long_context"]
    short_ctx = long_suite["short_context"]

    rows = [
        {
            "name": "openwebtext_backbone_v2",
            "plasticity_gain": openweb_plasticity,
            "stability_score": openweb_stability,
            "risk_load": openweb_write,
        },
        {
            "name": "qwen3_4b_recovery",
            "plasticity_gain": qwen["success_rate"] - online_chain["models"]["qwen3_4b"]["systems"]["online_no_recovery"]["success_rate"],
            "stability_score": qwen["mean_post_recovery_stability"],
            "risk_load": qwen["rollback_trigger_rate"] * (1.0 - qwen["rollback_recovery_rate"]),
        },
        {
            "name": "deepseek_7b_recovery",
            "plasticity_gain": deepseek["success_rate"] - online_chain["models"]["deepseek_7b"]["systems"]["online_no_recovery"]["success_rate"],
            "stability_score": deepseek["mean_post_recovery_stability"],
            "risk_load": deepseek["rollback_trigger_rate"] * (1.0 - deepseek["rollback_recovery_rate"]),
        },
        {
            "name": "qwen3_4b_learnable_heads",
            "plasticity_gain": stage_heads["gains"]["qwen_learned_minus_fixed_success"],
            "stability_score": 1.0 - stage_heads["headline_metrics"]["qwen_learned_trigger_rate"],
            "risk_load": qwen_heads.get("tool_failure_rate", 0.0),
        },
        {
            "name": "deepseek_7b_learnable_heads",
            "plasticity_gain": stage_heads["gains"]["deepseek_learned_minus_fixed_success"],
            "stability_score": 1.0 - stage_heads["headline_metrics"]["deepseek_learned_trigger_rate"],
            "risk_load": deepseek_heads.get("tool_failure_rate", 0.0),
        },
        {
            "name": "prototype_short_context",
            "plasticity_gain": short_ctx["deltas"]["novel_accuracy_delta"],
            "stability_score": 1.0 - short_ctx["deltas"]["forgetting"],
            "risk_load": short_ctx["deltas"]["base_perplexity_delta"],
        },
        {
            "name": "prototype_long_context",
            "plasticity_gain": long_ctx["deltas"]["novel_accuracy_delta"],
            "stability_score": 1.0 - long_ctx["deltas"]["forgetting"],
            "risk_load": long_ctx["deltas"]["base_perplexity_delta"],
        },
    ]

    summary = {
        "case_count": len(rows),
        "rows": rows,
        "headline_metrics": {
            "plasticity_mean": mean([r["plasticity_gain"] for r in rows]),
            "stability_mean": mean([r["stability_score"] for r in rows]),
            "risk_load_mean": mean([r["risk_load"] for r in rows]),
            "best_plasticity_case": max(rows, key=lambda r: r["plasticity_gain"])["name"],
            "highest_risk_case": max(rows, key=lambda r: r["risk_load"])["name"],
        },
        "project_readout": {
            "summary": (
                "这一步把 openwebtext 训练块、Qwen/DeepSeek 在线恢复链、"
                "可学习阶段头和当前语言原型放进同一稳定性-可塑性坐标。"
                "目标是判断大模型资产上是否也出现“学习更强、风险负载也更大”的系统规律。"
            ),
            "next_question": "下一步要进一步把这些稳定性代理并回当前 G/S/L_base/L_select 的系统短式。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    lines = [
        "# 大模型在线稳定性代理报告",
        "",
        f"- 资产数量: {summary['case_count']}",
        f"- 平均可塑性增益: {summary['headline_metrics']['plasticity_mean']:.6f}",
        f"- 平均稳定性分数: {summary['headline_metrics']['stability_mean']:.6f}",
        f"- 平均风险负载: {summary['headline_metrics']['risk_load_mean']:.6f}",
        f"- 最高可塑性案例: {summary['headline_metrics']['best_plasticity_case']}",
        f"- 最高风险案例: {summary['headline_metrics']['highest_risk_case']}",
        "",
        "## 个案",
    ]
    for row in summary["rows"]:
        lines.append(
            f"- {row['name']}: plasticity={row['plasticity_gain']:.6f}, "
            f"stability={row['stability_score']:.6f}, risk={row['risk_load']:.6f}"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_model_stability_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
