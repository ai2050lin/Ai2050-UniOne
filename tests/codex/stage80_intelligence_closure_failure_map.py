from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage80_intelligence_closure_failure_map_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage70_direct_identity_lock import build_direct_identity_lock_summary
from stage72_language_projection_covariance import build_language_projection_covariance_summary
from stage73_falsifiability_boundary_hardening import build_falsifiability_boundary_hardening_summary
from stage76_sqrt_repair_generalization import build_sqrt_repair_generalization_summary
from stage79_route_conflict_native_measure import build_route_conflict_native_measure_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_intelligence_closure_failure_map_summary() -> dict:
    identity = build_direct_identity_lock_summary()["headline_metrics"]
    projection = build_language_projection_covariance_summary()["headline_metrics"]
    fals_boundary = build_falsifiability_boundary_hardening_summary()["headline_metrics"]
    repair = build_sqrt_repair_generalization_summary()["headline_metrics"]
    route_conflict = build_route_conflict_native_measure_summary()["headline_metrics"]

    scenarios = [
        {
            "name": "multi_hop_composition",
            "composition_load": 0.82,
            "context_span": 0.56,
            "repair_dependence": 0.62,
            "abstraction_gap": 0.48,
        },
        {
            "name": "context_transfer",
            "composition_load": 0.58,
            "context_span": 0.84,
            "repair_dependence": 0.54,
            "abstraction_gap": 0.42,
        },
        {
            "name": "novelty_generalization",
            "composition_load": 0.64,
            "context_span": 0.50,
            "repair_dependence": 0.78,
            "abstraction_gap": 0.61,
        },
        {
            "name": "conflict_recovery",
            "composition_load": 0.46,
            "context_span": 0.44,
            "repair_dependence": 0.86,
            "abstraction_gap": 0.38,
        },
        {
            "name": "abstraction_compression",
            "composition_load": 0.74,
            "context_span": 0.40,
            "repair_dependence": 0.48,
            "abstraction_gap": 0.82,
        },
    ]

    scenario_records = []
    recovery_scores = []
    abstraction_scores = []
    failure_scores = []

    for scenario in scenarios:
        closure_drift = _clip01(
            0.24 * scenario["composition_load"]
            + 0.20 * scenario["context_span"]
            + 0.18 * scenario["repair_dependence"]
            + 0.14 * scenario["abstraction_gap"]
            + 0.14 * route_conflict["route_conflict_mass"]
            + 0.10 * (1.0 - fals_boundary["weakest_failure_mode_score"])
        )
        recovery_coherence = _clip01(
            0.28 * route_conflict["conflict_resolution_readiness"]
            + 0.22 * route_conflict["training_route_alignment"]
            + 0.20 * repair["repaired_bounded_learning_window"]
            + 0.16 * identity["identity_lock_confidence"]
            + 0.14 * (1.0 - closure_drift)
        )
        abstraction_bridge = _clip01(
            0.30 * projection["language_projection_repair_score"]
            + 0.24 * route_conflict["inference_route_coherence"]
            + 0.18 * identity["locked_identity_readiness"]
            + 0.14 * (1.0 - scenario["abstraction_gap"])
            + 0.14 * (1.0 - closure_drift)
        )
        failure_intensity = _clip01(
            0.34 * closure_drift
            + 0.22 * (1.0 - recovery_coherence)
            + 0.20 * (1.0 - abstraction_bridge)
            + 0.14 * scenario["repair_dependence"]
            + 0.10 * scenario["abstraction_gap"]
        )

        scenario_records.append(
            {
                "name": scenario["name"],
                "closure_drift": closure_drift,
                "recovery_coherence": recovery_coherence,
                "abstraction_bridge": abstraction_bridge,
                "failure_intensity": failure_intensity,
            }
        )
        recovery_scores.append(recovery_coherence)
        abstraction_scores.append(abstraction_bridge)
        failure_scores.append(failure_intensity)

    worst_case = max(scenario_records, key=lambda item: item["failure_intensity"])
    average_recovery_coherence = sum(recovery_scores) / len(recovery_scores)
    abstraction_bridge_strength = sum(abstraction_scores) / len(abstraction_scores)
    closure_failure_surface_coverage = _clip01(
        0.34 * (sum(failure_scores) / len(failure_scores))
        + 0.28 * worst_case["failure_intensity"]
        + 0.20 * (1.0 - route_conflict["training_route_alignment"])
        + 0.18 * (1.0 - identity["locked_identity_readiness"])
    )
    closure_repair_priority = _clip01(
        0.40 * worst_case["failure_intensity"]
        + 0.30 * (1.0 - average_recovery_coherence)
        + 0.30 * (1.0 - abstraction_bridge_strength)
    )
    intelligence_closure_failure_map_score = _clip01(
        0.28 * average_recovery_coherence
        + 0.24 * abstraction_bridge_strength
        + 0.20 * (1.0 - worst_case["failure_intensity"])
        + 0.16 * route_conflict["route_computation_closure_score"]
        + 0.12 * (1.0 - closure_repair_priority)
    )

    return {
        "headline_metrics": {
            "closure_failure_surface_coverage": closure_failure_surface_coverage,
            "average_recovery_coherence": average_recovery_coherence,
            "abstraction_bridge_strength": abstraction_bridge_strength,
            "worst_case_name": worst_case["name"],
            "worst_case_failure_intensity": worst_case["failure_intensity"],
            "closure_repair_priority": closure_repair_priority,
            "intelligence_closure_failure_map_score": intelligence_closure_failure_map_score,
        },
        "scenario_records": scenario_records,
        "status": {
            "status_short": (
                "intelligence_closure_failure_map_ready"
                if intelligence_closure_failure_map_score >= 0.80 and worst_case["failure_intensity"] <= 0.42
                else "intelligence_closure_failure_map_transition"
            ),
            "status_label": "智能闭合失效图谱已经出现可执行轮廓，但高抽象压缩和新颖泛化仍然容易拉开前向路由与反向修复的耦合。",
        },
        "project_readout": {
            "summary": "这一轮把 intelligence_closure 拆成多跳组合、上下文迁移、新颖泛化、冲突恢复和抽象压缩五类任务场景，第一次能直接比较哪类任务组合最容易击穿闭合。",
            "next_question": "下一步要把前向路由与反向修复写进同一个统一块，避免 intelligence_closure 仍然只是由两个子分数间接拼接。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage80 Intelligence Closure Failure Map",
        "",
        f"- closure_failure_surface_coverage: {hm['closure_failure_surface_coverage']:.6f}",
        f"- average_recovery_coherence: {hm['average_recovery_coherence']:.6f}",
        f"- abstraction_bridge_strength: {hm['abstraction_bridge_strength']:.6f}",
        f"- worst_case_name: {hm['worst_case_name']}",
        f"- worst_case_failure_intensity: {hm['worst_case_failure_intensity']:.6f}",
        f"- closure_repair_priority: {hm['closure_repair_priority']:.6f}",
        f"- intelligence_closure_failure_map_score: {hm['intelligence_closure_failure_map_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_intelligence_closure_failure_map_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
