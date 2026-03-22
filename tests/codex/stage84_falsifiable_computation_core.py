from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage84_falsifiable_computation_core_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage72_language_projection_covariance import build_language_projection_covariance_summary
from stage73_falsifiability_boundary_hardening import build_falsifiability_boundary_hardening_summary
from stage79_route_conflict_native_measure import build_route_conflict_native_measure_summary
from stage82_novelty_generalization_repair import build_novelty_generalization_repair_summary
from stage83_forward_backward_theorem_kernel import build_forward_backward_theorem_kernel_summary
from stage88_external_counterexample_expansion import build_external_counterexample_expansion_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def build_falsifiable_computation_core_summary() -> dict:
    projection = build_language_projection_covariance_summary()["headline_metrics"]
    boundary = build_falsifiability_boundary_hardening_summary()
    route = build_route_conflict_native_measure_summary()["headline_metrics"]
    novelty = build_novelty_generalization_repair_summary()["headline_metrics"]
    theorem = build_forward_backward_theorem_kernel_summary()["headline_metrics"]
    external_expansion = build_external_counterexample_expansion_summary()["headline_metrics"]

    hb = boundary["headline_metrics"]
    failure_mode_map = boundary["failure_mode_map"]

    executable_theorem_discrimination = _clip01(
        0.26 * hb["boundary_counterexample_discrimination"]
        + 0.24 * hb["task_counterexample_activation"]
        + 0.24 * (1.0 - theorem["projection_error_bound"])
        + 0.26 * (1.0 - theorem["repair_contraction_ratio"])
    )
    counterexample_contact_strength = _clip01(
        0.22 * hb["executable_boundary_coverage"]
        + 0.20 * route["route_computation_closure_score"]
        + 0.20 * theorem["theorem_premise_satisfaction"]
        + 0.20 * (1.0 - novelty["best_failure_after"])
        + 0.10 * external_expansion["expanded_trigger_rate"]
        + 0.08 * external_expansion["strongest_refutation_strength"]
    )
    route_projection_failure_traceability = _clip01(
        0.28 * (1.0 - route["route_conflict_mass"])
        + 0.26 * (1.0 - projection["projection_gap"])
        + 0.24 * hb["boundary_counterexample_discrimination"]
        + 0.22 * theorem["cross_projection_consistency"]
    )
    shared_state_refutation_power = _clip01(
        0.28 * hb["shared_state_rejection_power"]
        + 0.22 * executable_theorem_discrimination
        + 0.20 * theorem["theorem_conclusion_strength"]
        + 0.18 * failure_mode_map["shared_state"]["mode_score"]
        + 0.12 * external_expansion["external_counterexample_expansion_score"]
    )

    scenario_records = [
        {
            "name": "context_covariance_break",
            "trigger_contact": _clip01(
                0.44 * hb["task_counterexample_activation"]
                + 0.30 * (1.0 - projection["projection_counterexample_resistance"])
                + 0.26 * (1.0 - projection["projection_gap"])
            ),
            "observable_failure": _clip01(
                0.42 * (1.0 - failure_mode_map["context_covariance"]["mode_score"])
                + 0.34 * projection["projection_gap"]
                + 0.24 * theorem["projection_error_bound"]
            ),
            "core_rejection": _clip01(
                0.38 * executable_theorem_discrimination
                + 0.34 * route_projection_failure_traceability
                + 0.28 * hb["shared_state_rejection_power"]
            ),
        },
        {
            "name": "route_conflict_overflow",
            "trigger_contact": _clip01(
                0.42 * route["route_conflict_mass"]
                + 0.32 * hb["boundary_counterexample_discrimination"]
                + 0.26 * hb["task_counterexample_activation"]
            ),
            "observable_failure": _clip01(
                0.40 * route["route_conflict_mass"]
                + 0.34 * (1.0 - route["conflict_resolution_readiness"])
                + 0.26 * theorem["repair_contraction_ratio"]
            ),
            "core_rejection": _clip01(
                0.40 * executable_theorem_discrimination
                + 0.30 * counterexample_contact_strength
                + 0.30 * shared_state_refutation_power
            ),
        },
        {
            "name": "novelty_bounded_break",
            "trigger_contact": _clip01(
                0.40 * hb["task_counterexample_activation"]
                + 0.34 * failure_mode_map["learning_stability"]["trigger_demonstrated"]
                + 0.26 * (1.0 - novelty["best_failure_after"])
            ),
            "observable_failure": _clip01(
                0.42 * (1.0 - failure_mode_map["learning_stability"]["mode_score"])
                + 0.32 * (1.0 - theorem["bounded_novelty_margin"])
                + 0.26 * novelty["best_failure_after"]
            ),
            "core_rejection": _clip01(
                0.38 * executable_theorem_discrimination
                + 0.30 * counterexample_contact_strength
                + 0.32 * shared_state_refutation_power
            ),
        },
        {
            "name": "shared_state_decoupling",
            "trigger_contact": _clip01(
                0.40 * hb["shared_state_rejection_power"]
                + 0.32 * hb["boundary_counterexample_discrimination"]
                + 0.28 * counterexample_contact_strength
            ),
            "observable_failure": _clip01(
                0.42 * (1.0 - failure_mode_map["shared_state"]["mode_score"])
                + 0.30 * theorem["projection_error_bound"]
                + 0.28 * (1.0 - theorem["cross_projection_consistency"])
            ),
            "core_rejection": _clip01(
                0.44 * shared_state_refutation_power
                + 0.30 * executable_theorem_discrimination
                + 0.26 * theorem["theorem_conclusion_strength"]
            ),
        },
    ]

    for record in scenario_records:
        record["counterexample_intensity"] = _clip01(
            0.36 * record["trigger_contact"]
            + 0.34 * record["observable_failure"]
            + 0.30 * (1.0 - record["core_rejection"])
        )

    hardest_counterexample = max(scenario_records, key=lambda item: item["counterexample_intensity"])

    falsifiable_computation_core_score = _clip01(
        0.22 * executable_theorem_discrimination
        + 0.22 * counterexample_contact_strength
        + 0.20 * route_projection_failure_traceability
        + 0.20 * shared_state_refutation_power
        + 0.16 * (1.0 - hardest_counterexample["counterexample_intensity"])
    )

    return {
        "headline_metrics": {
            "executable_theorem_discrimination": executable_theorem_discrimination,
            "counterexample_contact_strength": counterexample_contact_strength,
            "route_projection_failure_traceability": route_projection_failure_traceability,
            "shared_state_refutation_power": shared_state_refutation_power,
            "hardest_counterexample_name": hardest_counterexample["name"],
            "hardest_counterexample_intensity": hardest_counterexample["counterexample_intensity"],
            "falsifiable_computation_core_score": falsifiable_computation_core_score,
        },
        "scenario_records": scenario_records,
        "external_counterexample_expansion_bridge": external_expansion,
        "falsifiable_core_equations": {
            "counterexample_trigger": "T_fail = contact(observable_break, theorem_break, route_conflict, task_trigger)",
            "core_rejection": "R_core = reject(shared_state, projection_bound, repair_contraction, novelty_bound)",
            "falsifiable_core": "F_core = align(T_fail, observable_failure) * (1 - R_core)",
            "verdict_rule": "if F_core > stability_margin then theorem_kernel fails else theorem_kernel survives",
        },
        "status": {
            "status_short": (
                "falsifiable_computation_core_ready"
                if falsifiable_computation_core_score >= 0.84
                and hardest_counterexample["counterexample_intensity"] <= 0.40
                else "falsifiable_computation_core_transition"
            ),
            "status_label": "可判伪计算主核已经开始把定理约束和反例触发写进同一块，但最坏反例还没有被完全压进低风险区。",
        },
        "project_readout": {
            "summary": "这一轮把 theorem kernel、语言投影反例、路由冲突和共享状态拒真能力压成同一个 falsifiable computation core，不再只看边界摘要项。",
            "next_question": "下一步要把 brain_grounding 单独做成反例包，检查哪些神经层失配会最直接打穿当前主核。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage84 Falsifiable Computation Core",
        "",
        f"- executable_theorem_discrimination: {hm['executable_theorem_discrimination']:.6f}",
        f"- counterexample_contact_strength: {hm['counterexample_contact_strength']:.6f}",
        f"- route_projection_failure_traceability: {hm['route_projection_failure_traceability']:.6f}",
        f"- shared_state_refutation_power: {hm['shared_state_refutation_power']:.6f}",
        f"- hardest_counterexample_name: {hm['hardest_counterexample_name']}",
        f"- hardest_counterexample_intensity: {hm['hardest_counterexample_intensity']:.6f}",
        f"- falsifiable_computation_core_score: {hm['falsifiable_computation_core_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_falsifiable_computation_core_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
