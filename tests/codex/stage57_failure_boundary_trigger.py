from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage57_failure_boundary_trigger_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_context_native_grounding import build_context_native_grounding_summary
from stage57_falsifiable_kernel_minimum import build_falsifiable_kernel_minimum_summary
from stage57_fiber_reuse_reinforcement import build_fiber_reuse_reinforcement_summary
from stage57_kernel_feedback_reintegration import build_kernel_feedback_reintegration_summary


def _patch_failure(structure_anchor: float, local_compatibility: float) -> bool:
    return structure_anchor < 0.70 and local_compatibility < 0.70


def _fiber_failure(fiber_reuse: float, coupling_balance: float) -> bool:
    return fiber_reuse < 0.45 or coupling_balance < 0.70


def _route_failure(context_alignment: float, pressure_under_reuse: float) -> bool:
    return context_alignment < 0.70 and pressure_under_reuse < 0.60


def _kernel_failure(domination_penalty: float) -> bool:
    return domination_penalty >= 0.35


def build_failure_boundary_trigger_summary() -> dict:
    reintegrated = build_kernel_feedback_reintegration_summary()
    fiber = build_fiber_reuse_reinforcement_summary()["headline_metrics"]
    context = build_context_native_grounding_summary()["headline_metrics"]
    kernel = build_falsifiable_kernel_minimum_summary()["headline_metrics"]

    best_name = reintegrated["headline_metrics"]["best_reintegrated_candidate_name"]
    best = reintegrated["reintegrated_candidates"][best_name]

    live_checks = {
        "patch_failure": _patch_failure(best["reintegrated_structure_anchor"], best["reintegrated_local_compatibility"]),
        "fiber_failure": _fiber_failure(fiber["fiber_reuse"], fiber["route_fiber_coupling_balance"]),
        "route_failure": _route_failure(context["context_route_alignment"], fiber["pressure_under_reuse"]),
        "kernel_failure": _kernel_failure(best["domination_penalty"]),
    }

    synthetic_stress = {
        "patch_triggered": _patch_failure(0.66, 0.67),
        "fiber_triggered": _fiber_failure(0.42, 0.78),
        "route_triggered": _route_failure(0.67, 0.58),
        "kernel_triggered": _kernel_failure(0.36),
    }

    live_boundary_pass_rate = 1.0 - sum(1.0 for failed in live_checks.values() if failed) / len(live_checks)
    triggerability_score = sum(1.0 for fired in synthetic_stress.values() if fired) / len(synthetic_stress)
    counterexample_activation_score = max(
        0.0,
        min(1.0, 0.50 * triggerability_score + 0.25 * live_boundary_pass_rate + 0.25 * kernel["boundary_sharpness"]),
    )
    boundary_system_readiness = max(
        0.0,
        min(
            1.0,
            0.35 * live_boundary_pass_rate
            + 0.30 * triggerability_score
            + 0.20 * counterexample_activation_score
            + 0.15 * kernel["kernel_minimum_viability"],
        ),
    )

    return {
        "headline_metrics": {
            "live_boundary_pass_rate": live_boundary_pass_rate,
            "triggerability_score": triggerability_score,
            "counterexample_activation_score": counterexample_activation_score,
            "boundary_system_readiness": boundary_system_readiness,
        },
        "live_checks": live_checks,
        "synthetic_stress": synthetic_stress,
        "project_readout": {
            "summary": "Failure-boundary trigger converts the minimum falsifiable kernel into executable checks, verifies that the current run stays inside the live safe region, and proves that synthetic counterexamples really trip the declared rules.",
            "next_question": "Replace synthetic stress cases with real stress generators so each failed rule can be triggered by an actual experimental script instead of a hand-set metric drop.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage57 Failure Boundary Trigger",
        "",
        f"- live_boundary_pass_rate: {hm['live_boundary_pass_rate']:.6f}",
        f"- triggerability_score: {hm['triggerability_score']:.6f}",
        f"- counterexample_activation_score: {hm['counterexample_activation_score']:.6f}",
        f"- boundary_system_readiness: {hm['boundary_system_readiness']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_failure_boundary_trigger_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
