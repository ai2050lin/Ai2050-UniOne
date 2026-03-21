from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage57_falsifiable_kernel_minimum_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_context_native_grounding import build_context_native_grounding_summary
from stage57_fiber_reuse_reinforcement import build_fiber_reuse_reinforcement_summary
from stage57_learning_rule_dual_candidate_review import build_learning_rule_dual_candidate_review_summary


def build_falsifiable_kernel_minimum_summary() -> dict:
    dual = build_learning_rule_dual_candidate_review_summary()
    fiber = build_fiber_reuse_reinforcement_summary()
    context = build_context_native_grounding_summary()

    dual_hm = dual["headline_metrics"]
    fiber_hm = fiber["headline_metrics"]
    context_hm = context["headline_metrics"]

    falsifiability_coverage = max(
        0.0,
        min(
            1.0,
            0.30 * dual_hm["best_candidate_overall_readiness"]
            + 0.25 * fiber_hm["reinforcement_readiness"]
            + 0.25 * context_hm["context_native_readiness"]
            + 0.20 * (1.0 - dual_hm["best_candidate_domination_penalty"]),
        ),
    )
    boundary_sharpness = max(
        0.0,
        min(
            1.0,
            0.40 * dual_hm["readiness_margin"]
            + 0.30 * fiber_hm["route_fiber_coupling_balance"]
            + 0.30 * context_hm["context_upgrade_gain"],
        ),
    )
    counterexample_readiness = max(
        0.0,
        min(
            1.0,
            0.35 * (1.0 - dual_hm["best_candidate_domination_penalty"])
            + 0.35 * fiber_hm["pressure_under_reuse"]
            + 0.30 * context_hm["context_route_alignment"],
        ),
    )
    kernel_minimum_viability = max(
        0.0,
        min(
            1.0,
            0.35 * falsifiability_coverage
            + 0.30 * boundary_sharpness
            + 0.35 * counterexample_readiness,
        ),
    )

    failure_boundaries = {
        "patch_failure_rule": "If structure_anchor_score < 0.70 and local_law_compatibility < 0.70, reject the claim that bounded learning alone can recover patch structure.",
        "fiber_failure_rule": "If fiber_reuse < 0.45 or route_fiber_coupling_balance < 0.70, reject the claim that fiber is a generated object rather than a named middle-layer label.",
        "route_failure_rule": "If context_route_alignment < 0.70 and pressure_under_reuse < 0.60, reject the claim that route structure remains stable under contextual and reuse pressure.",
        "kernel_failure_rule": "If the winning bounded law still depends on domination_penalty >= 0.35, reject the current minimum kernel as numerically patched instead of structurally grounded.",
    }

    return {
        "headline_metrics": {
            "falsifiability_coverage": falsifiability_coverage,
            "boundary_sharpness": boundary_sharpness,
            "counterexample_readiness": counterexample_readiness,
            "kernel_minimum_viability": kernel_minimum_viability,
        },
        "failure_boundaries": failure_boundaries,
        "project_readout": {
            "summary": "The minimum falsifiable kernel turns the current best learning rule, fiber reinforcement, and context grounding outputs into explicit failure boundaries for patch, fiber, route, and kernel claims.",
            "next_question": "Run the four work orders together and use any failed boundary to decide which claim must be rewritten first.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage57 Falsifiable Kernel Minimum",
        "",
        f"- falsifiability_coverage: {hm['falsifiability_coverage']:.6f}",
        f"- boundary_sharpness: {hm['boundary_sharpness']:.6f}",
        f"- counterexample_readiness: {hm['counterexample_readiness']:.6f}",
        f"- kernel_minimum_viability: {hm['kernel_minimum_viability']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_falsifiable_kernel_minimum_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
