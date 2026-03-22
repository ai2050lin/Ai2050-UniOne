from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage81_forward_backward_unification_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage70_direct_identity_lock import build_direct_identity_lock_summary
from stage72_language_projection_covariance import build_language_projection_covariance_summary
from stage76_sqrt_repair_generalization import build_sqrt_repair_generalization_summary
from stage79_route_conflict_native_measure import build_route_conflict_native_measure_summary
from stage80_intelligence_closure_failure_map import build_intelligence_closure_failure_map_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_forward_backward_unification_summary() -> dict:
    identity = build_direct_identity_lock_summary()["headline_metrics"]
    projection = build_language_projection_covariance_summary()["headline_metrics"]
    repair = build_sqrt_repair_generalization_summary()["headline_metrics"]
    route = build_route_conflict_native_measure_summary()["headline_metrics"]
    closure = build_intelligence_closure_failure_map_summary()["headline_metrics"]

    forward_selectivity = _clip01(
        0.30 * route["attention_like_selection"]
        + 0.26 * route["inference_route_coherence"]
        + 0.24 * projection["route_conditioned_projection"]
        + 0.20 * projection["context_covariance_stability"]
    )
    backward_fidelity = _clip01(
        0.30 * route["gradient_like_correction"]
        + 0.24 * route["training_route_alignment"]
        + 0.24 * repair["repair_generalization_score"]
        + 0.22 * repair["repaired_bounded_learning_window"]
    )
    novelty_binding_alignment = _clip01(
        0.28 * (1.0 - closure["worst_case_failure_intensity"])
        + 0.24 * route["training_route_alignment"]
        + 0.18 * repair["route_rebind_support"]
        + 0.16 * (1.0 - route["route_conflict_mass"])
        + 0.14 * identity["locked_identity_readiness"]
    )
    loop_stability_gain = _clip01(
        0.26 * forward_selectivity
        + 0.26 * backward_fidelity
        + 0.22 * route["conflict_resolution_readiness"]
        + 0.14 * novelty_binding_alignment
        + 0.12 * (1.0 - closure["closure_repair_priority"])
    )
    forward_backward_unification_score = _clip01(
        0.24 * forward_selectivity
        + 0.22 * backward_fidelity
        + 0.20 * novelty_binding_alignment
        + 0.18 * loop_stability_gain
        + 0.16 * (1.0 - route["route_conflict_mass"])
    )

    scenario_records = [
        {
            "name": "stable_context_loop",
            "forward_readout": _clip01(
                0.44 * forward_selectivity
                + 0.30 * projection["context_shift_resilience"]
                + 0.26 * route["inference_route_coherence"]
            ),
            "backward_readout": _clip01(
                0.42 * backward_fidelity
                + 0.34 * repair["repair_generalization_score"]
                + 0.24 * identity["identity_lock_confidence"]
            ),
            "loop_coupling": _clip01(
                0.46 * loop_stability_gain
                + 0.30 * (1.0 - route["route_conflict_mass"])
                + 0.24 * novelty_binding_alignment
            ),
        },
        {
            "name": "novelty_generalization_loop",
            "forward_readout": _clip01(
                0.40 * forward_selectivity
                + 0.32 * projection["route_conditioned_projection"]
                + 0.28 * (1.0 - closure["worst_case_failure_intensity"])
            ),
            "backward_readout": _clip01(
                0.40 * backward_fidelity
                + 0.34 * route["training_route_alignment"]
                + 0.26 * repair["repaired_bounded_learning_window"]
            ),
            "loop_coupling": _clip01(
                0.38 * novelty_binding_alignment
                + 0.34 * loop_stability_gain
                + 0.28 * (1.0 - closure["closure_repair_priority"])
            ),
        },
        {
            "name": "conflict_repair_loop",
            "forward_readout": _clip01(
                0.42 * forward_selectivity
                + 0.30 * route["attention_like_selection"]
                + 0.28 * (1.0 - route["route_conflict_mass"])
            ),
            "backward_readout": _clip01(
                0.44 * backward_fidelity
                + 0.30 * route["gradient_like_correction"]
                + 0.26 * route["conflict_resolution_readiness"]
            ),
            "loop_coupling": _clip01(
                0.40 * loop_stability_gain
                + 0.34 * route["conflict_resolution_readiness"]
                + 0.26 * (1.0 - route["route_conflict_mass"])
            ),
        },
    ]

    return {
        "headline_metrics": {
            "forward_selectivity": forward_selectivity,
            "backward_fidelity": backward_fidelity,
            "novelty_binding_alignment": novelty_binding_alignment,
            "loop_stability_gain": loop_stability_gain,
            "forward_backward_unification_score": forward_backward_unification_score,
        },
        "scenario_records": scenario_records,
        "forward_backward_equation": {
            "forward_selection": "state_plus = residual + soft_select(score_ctx + score_reuse - cost - conflict) * value_mix",
            "loss_feedback": "delta = grad(loss, state_plus)",
            "backward_repair": "param_plus = param - eta * delta * bounded_plasticity",
            "loop_coupling": "coupling = align(forward_route_field, backward_repair_field, novelty_binding)",
        },
        "status": {
            "status_short": (
                "forward_backward_unification_ready"
                if forward_backward_unification_score >= 0.82 and novelty_binding_alignment >= 0.74
                else "forward_backward_unification_transition"
            ),
            "status_label": "前向路由与反向修复已经出现统一闭环轮廓，但新颖泛化的耦合仍不够强稳。",
        },
        "project_readout": {
            "summary": "这一轮把前向选路、反向修复和新颖绑定写进同一个统一块，避免 intelligence_closure 继续只由几个间接摘要项拼接出来。",
            "next_question": "下一步要围绕 novelty_generalization 单独做修复块，检查前向与反向的耦合能否继续增强。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage81 Forward Backward Unification",
        "",
        f"- forward_selectivity: {hm['forward_selectivity']:.6f}",
        f"- backward_fidelity: {hm['backward_fidelity']:.6f}",
        f"- novelty_binding_alignment: {hm['novelty_binding_alignment']:.6f}",
        f"- loop_stability_gain: {hm['loop_stability_gain']:.6f}",
        f"- forward_backward_unification_score: {hm['forward_backward_unification_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_forward_backward_unification_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
