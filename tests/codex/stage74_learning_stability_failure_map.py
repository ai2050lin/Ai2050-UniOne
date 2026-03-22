from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage74_learning_stability_failure_map_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_language_task_boundary_trigger import build_language_task_boundary_trigger_summary
from stage70_direct_stability_counterexample_probe import build_direct_stability_counterexample_probe_summary
from stage72_language_projection_covariance import build_language_projection_covariance_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_learning_stability_failure_map_summary() -> dict:
    task = build_language_task_boundary_trigger_summary()
    counter = build_direct_stability_counterexample_probe_summary()["headline_metrics"]
    projection = build_language_projection_covariance_summary()["headline_metrics"]

    ht = task["headline_metrics"]

    base_forgetting_risk = _clip01((ht["stressed_long_forgetting"] - 0.20) / 0.80)
    base_perplexity_stress = _clip01((ht["stressed_base_perplexity_delta"] - 900.0) / 500.0)
    base_novel_risk = _clip01((0.90 - ht["stressed_novel_accuracy_after"]) / 0.30)

    scenarios = [
        {
            "name": "semantic_patch_insert",
            "novelty_load": 0.72,
            "retention_conflict": 0.44,
            "context_shift": 0.22,
            "route_load": 0.28,
        },
        {
            "name": "route_rebind_insert",
            "novelty_load": 0.65,
            "retention_conflict": 0.58,
            "context_shift": 0.36,
            "route_load": 0.70,
        },
        {
            "name": "context_switch_write",
            "novelty_load": 0.52,
            "retention_conflict": 0.48,
            "context_shift": 0.82,
            "route_load": 0.54,
        },
        {
            "name": "compositional_binding_write",
            "novelty_load": 0.78,
            "retention_conflict": 0.74,
            "context_shift": 0.61,
            "route_load": 0.76,
        },
        {
            "name": "long_horizon_refresh",
            "novelty_load": 0.44,
            "retention_conflict": 0.86,
            "context_shift": 0.24,
            "route_load": 0.40,
        },
    ]

    scenario_records = []
    guarded_scores = []
    recovery_buffers = []
    failure_intensities = []

    for scenario in scenarios:
        novelty = scenario["novelty_load"]
        retention = scenario["retention_conflict"]
        context_shift = scenario["context_shift"]
        route = scenario["route_load"]

        forgetting_risk = _clip01(
            0.44 * base_forgetting_risk
            + 0.20 * novelty
            + 0.16 * retention
            + 0.12 * context_shift
            + 0.08 * route
            - 0.10 * projection["projection_counterexample_resistance"]
        )
        novelty_drop_risk = _clip01(
            0.34 * base_novel_risk
            + 0.22 * novelty
            + 0.18 * context_shift
            + 0.16 * retention
            + 0.10 * route
            - 0.12 * projection["language_projection_repair_score"]
        )
        perplexity_stress = _clip01(
            0.40 * base_perplexity_stress
            + 0.22 * route
            + 0.16 * novelty
            + 0.12 * context_shift
            + 0.10 * retention
            - 0.10 * projection["context_shift_resilience"]
        )
        guarded_update_score = _clip01(
            0.28 * (1.0 - forgetting_risk)
            + 0.24 * (1.0 - novelty_drop_risk)
            + 0.20 * (1.0 - perplexity_stress)
            + 0.16 * (1.0 - counter["counterexample_pressure"])
            + 0.12 * projection["language_projection_repair_score"]
        )
        recovery_buffer = _clip01(
            0.42 * (1.0 - forgetting_risk)
            + 0.28 * (1.0 - perplexity_stress)
            + 0.18 * projection["language_projection_repair_score"]
            + 0.12 * (1.0 - route)
        )
        failure_intensity = _clip01(
            0.36 * forgetting_risk
            + 0.26 * novelty_drop_risk
            + 0.18 * perplexity_stress
            + 0.10 * context_shift
            + 0.10 * retention
        )

        scenario_records.append(
            {
                "name": scenario["name"],
                "forgetting_risk": forgetting_risk,
                "novelty_drop_risk": novelty_drop_risk,
                "perplexity_stress": perplexity_stress,
                "guarded_update_score": guarded_update_score,
                "recovery_buffer": recovery_buffer,
                "failure_intensity": failure_intensity,
            }
        )
        guarded_scores.append(guarded_update_score)
        recovery_buffers.append(recovery_buffer)
        failure_intensities.append(failure_intensity)

    worst_case = max(scenario_records, key=lambda item: item["failure_intensity"])
    average_guarded_update_score = sum(guarded_scores) / len(guarded_scores)
    average_recovery_buffer = sum(recovery_buffers) / len(recovery_buffers)
    worst_case_failure_intensity = worst_case["failure_intensity"]
    learning_failure_surface_coverage = _clip01(
        0.34 * (sum(failure_intensities) / len(failure_intensities))
        + 0.26 * worst_case_failure_intensity
        + 0.20 * base_perplexity_stress
        + 0.20 * float(task["task_trigger"]["triggered"])
    )
    bounded_learning_window_score = _clip01(
        0.38 * average_guarded_update_score
        + 0.26 * average_recovery_buffer
        + 0.20 * (1.0 - worst_case_failure_intensity)
        + 0.16 * (1.0 - base_forgetting_risk)
    )
    stability_repair_priority = _clip01(
        0.44 * worst_case_failure_intensity
        + 0.30 * (1.0 - bounded_learning_window_score)
        + 0.26 * base_perplexity_stress
    )
    learning_stability_failure_map_score = _clip01(
        0.34 * average_guarded_update_score
        + 0.26 * bounded_learning_window_score
        + 0.20 * average_recovery_buffer
        + 0.20 * (1.0 - stability_repair_priority)
    )

    return {
        "headline_metrics": {
            "learning_failure_surface_coverage": learning_failure_surface_coverage,
            "average_guarded_update_score": average_guarded_update_score,
            "average_recovery_buffer": average_recovery_buffer,
            "bounded_learning_window_score": bounded_learning_window_score,
            "worst_case_failure_name": worst_case["name"],
            "worst_case_failure_intensity": worst_case_failure_intensity,
            "stability_repair_priority": stability_repair_priority,
            "learning_stability_failure_map_score": learning_stability_failure_map_score,
        },
        "scenario_records": scenario_records,
        "base_risk_state": {
            "base_forgetting_risk": base_forgetting_risk,
            "base_perplexity_stress": base_perplexity_stress,
            "base_novel_risk": base_novel_risk,
        },
        "status": {
            "status_short": (
                "learning_stability_failure_map_ready"
                if learning_stability_failure_map_score >= 0.66 and worst_case_failure_intensity <= 0.56
                else "learning_stability_failure_map_transition"
            ),
            "status_label": "学习稳态失效图谱已经具备可执行轮廓，但最坏写入场景仍会明显抬高失稳风险",
        },
        "project_readout": {
            "summary": "这一轮把 learning_stability 的失败来源拆成不同写入场景，第一次能直接比较哪类知识并入、哪类上下文切换、哪类路由负载最危险。",
            "next_question": "下一步要针对 worst_case 场景单独设计有界更新修复块，验证是否能把最坏失败强度继续压低。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage74 Learning Stability Failure Map",
        "",
        f"- learning_failure_surface_coverage: {hm['learning_failure_surface_coverage']:.6f}",
        f"- average_guarded_update_score: {hm['average_guarded_update_score']:.6f}",
        f"- average_recovery_buffer: {hm['average_recovery_buffer']:.6f}",
        f"- bounded_learning_window_score: {hm['bounded_learning_window_score']:.6f}",
        f"- worst_case_failure_name: {hm['worst_case_failure_name']}",
        f"- worst_case_failure_intensity: {hm['worst_case_failure_intensity']:.6f}",
        f"- stability_repair_priority: {hm['stability_repair_priority']:.6f}",
        f"- learning_stability_failure_map_score: {hm['learning_stability_failure_map_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_learning_stability_failure_map_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
