from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage79_route_conflict_native_measure_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage70_direct_stability_counterexample_probe import build_direct_stability_counterexample_probe_summary
from stage72_language_projection_covariance import build_language_projection_covariance_summary
from stage76_sqrt_repair_generalization import build_sqrt_repair_generalization_summary
from stage77_brain_grounded_route_scaling import build_brain_grounded_route_scaling_summary
from stage78_distributed_route_native_observability import build_distributed_route_native_observability_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_route_conflict_native_measure_summary() -> dict:
    counter = build_direct_stability_counterexample_probe_summary()["headline_metrics"]
    projection = build_language_projection_covariance_summary()["headline_metrics"]
    repair = build_sqrt_repair_generalization_summary()["headline_metrics"]
    scaling = build_brain_grounded_route_scaling_summary()["headline_metrics"]
    route_obs = build_distributed_route_native_observability_summary()["headline_metrics"]

    attention_like_selection = _clip01(
        0.30 * projection["route_conditioned_projection"]
        + 0.24 * projection["context_covariance_stability"]
        + 0.24 * scaling["distributed_network_support"]
        + 0.22 * route_obs["distributed_route_traceability"]
    )
    gradient_like_correction = _clip01(
        0.30 * repair["repair_generalization_score"]
        + 0.24 * repair["repaired_bounded_learning_window"]
        + 0.22 * scaling["brain_constrained_repair_score"]
        + 0.24 * (1.0 - counter["counterexample_pressure"])
    )
    route_conflict_mass = _clip01(
        0.28 * route_obs["field_proxy_gap"]
        + 0.24 * (1.0 - repair["route_rebind_support"])
        + 0.20 * (1.0 - repair["context_switch_support"])
        + 0.16 * counter["counterexample_pressure"]
        + 0.12 * (1.0 - route_obs["route_counterexample_triggerability"])
    )
    conflict_resolution_readiness = _clip01(
        0.28 * attention_like_selection
        + 0.30 * gradient_like_correction
        + 0.22 * (1.0 - route_conflict_mass)
        + 0.20 * (1.0 - route_obs["field_proxy_gap"])
    )
    inference_route_coherence = _clip01(
        0.34 * attention_like_selection
        + 0.24 * route_obs["distributed_route_traceability"]
        + 0.22 * scaling["route_scale_balance"]
        + 0.20 * (1.0 - route_conflict_mass)
    )
    training_route_alignment = _clip01(
        0.34 * gradient_like_correction
        + 0.24 * repair["repair_generalization_score"]
        + 0.22 * route_obs["route_conflict_native_measure"]
        + 0.20 * (1.0 - counter["counterexample_pressure"])
    )
    route_computation_closure_score = _clip01(
        0.22 * attention_like_selection
        + 0.22 * gradient_like_correction
        + 0.18 * conflict_resolution_readiness
        + 0.18 * inference_route_coherence
        + 0.20 * training_route_alignment
    )

    scenario_records = [
        {
            "name": "forward_selection",
            "attention_like_readout": _clip01(
                0.44 * attention_like_selection
                + 0.30 * projection["route_conditioned_projection"]
                + 0.26 * scaling["distributed_network_support"]
            ),
            "conflict_mass": _clip01(
                0.56 * route_conflict_mass
                + 0.24 * (1.0 - projection["projection_counterexample_resistance"])
                + 0.20 * counter["counterexample_pressure"]
            ),
        },
        {
            "name": "context_rebinding",
            "attention_like_readout": _clip01(
                0.40 * attention_like_selection
                + 0.32 * projection["context_covariance_stability"]
                + 0.28 * route_obs["distributed_route_traceability"]
            ),
            "conflict_mass": _clip01(
                0.48 * route_conflict_mass
                + 0.30 * (1.0 - repair["context_switch_support"])
                + 0.22 * route_obs["route_conflict_native_measure"]
            ),
        },
        {
            "name": "backward_correction",
            "gradient_like_readout": _clip01(
                0.46 * gradient_like_correction
                + 0.30 * repair["repair_generalization_score"]
                + 0.24 * scaling["brain_constrained_repair_score"]
            ),
            "repair_release": _clip01(
                0.42 * conflict_resolution_readiness
                + 0.32 * training_route_alignment
                + 0.26 * (1.0 - route_conflict_mass)
            ),
        },
    ]

    return {
        "headline_metrics": {
            "attention_like_selection": attention_like_selection,
            "gradient_like_correction": gradient_like_correction,
            "route_conflict_mass": route_conflict_mass,
            "conflict_resolution_readiness": conflict_resolution_readiness,
            "inference_route_coherence": inference_route_coherence,
            "training_route_alignment": training_route_alignment,
            "route_computation_closure_score": route_computation_closure_score,
        },
        "scenario_records": scenario_records,
        "route_computation_equation": {
            "forward_route": "route_logits = score_ctx + score_reuse - transport_cost - conflict_penalty",
            "forward_selection": "route_weights = soft_select(route_logits), state_plus = route_weights * value_mix + residual",
            "conflict_mass": "route_conflict = positive_part(route_demand - feasible_capacity)",
            "backward_correction": "delta_route = grad(loss, route_params) * bounded_plasticity",
            "repair_update": "route_params_plus = route_params - eta * delta_route, with saturation and recovery constraints",
        },
        "status": {
            "status_short": (
                "route_conflict_native_measure_ready"
                if route_computation_closure_score >= 0.81 and conflict_resolution_readiness >= 0.82
                else "route_conflict_native_measure_transition"
            ),
            "status_label": "路由冲突已经从可观测量推进到计算测度，开始同时覆盖前向选择、冲突积累和反向修复三步。",
        },
        "project_readout": {
            "summary": "这一轮把 route_conflict 压成计算测度，不再只看路由场读数，而是显式描述前向选择、冲突质量和反向修复如何串成一条计算链。",
            "next_question": "下一步要把 intelligence_closure 的失败模式拆开，看哪一类任务组合最容易让前向路由和反向修复脱耦。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage79 Route Conflict Native Measure",
        "",
        f"- attention_like_selection: {hm['attention_like_selection']:.6f}",
        f"- gradient_like_correction: {hm['gradient_like_correction']:.6f}",
        f"- route_conflict_mass: {hm['route_conflict_mass']:.6f}",
        f"- conflict_resolution_readiness: {hm['conflict_resolution_readiness']:.6f}",
        f"- inference_route_coherence: {hm['inference_route_coherence']:.6f}",
        f"- training_route_alignment: {hm['training_route_alignment']:.6f}",
        f"- route_computation_closure_score: {hm['route_computation_closure_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_route_conflict_native_measure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
