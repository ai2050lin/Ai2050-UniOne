from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage57_kernel_feedback_reintegration_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_context_native_grounding import build_context_native_grounding_summary
from stage57_fiber_reuse_reinforcement import build_fiber_reuse_reinforcement_summary
from stage57_learning_rule_dual_candidate_review import build_learning_rule_dual_candidate_review_summary


def build_kernel_feedback_reintegration_summary() -> dict:
    dual = build_learning_rule_dual_candidate_review_summary()
    fiber = build_fiber_reuse_reinforcement_summary()["headline_metrics"]
    context = build_context_native_grounding_summary()["headline_metrics"]

    reintegrated = {}
    for mode, metrics in dual["candidate_review"].items():
        feedback_gain = max(
            0.0,
            min(
                1.0,
                0.30 * fiber["fiber_reuse"]
                + 0.20 * fiber["route_fiber_coupling_balance"]
                + 0.20 * context["context_native_readiness"]
                + 0.15 * context["context_route_alignment"]
                + 0.15 * (1.0 - metrics["domination_penalty"]),
            ),
        )
        reintegrated_structure_anchor = max(
            0.0,
            min(
                1.0,
                metrics["structure_anchor_score"]
                + 0.10 * fiber["fiber_reuse"]
                + 0.08 * context["context_native_readiness"]
                - 0.06 * metrics["domination_penalty"],
            ),
        )
        reintegrated_local_compatibility = max(
            0.0,
            min(
                1.0,
                0.50 * metrics["local_law_compatibility"]
                + 0.20 * fiber["reinforcement_readiness"]
                + 0.15 * context["conditional_gate_stability"]
                + 0.15 * feedback_gain,
            ),
        )
        reintegrated_overall_readiness = max(
            0.0,
            min(
                1.0,
                0.30 * metrics["overall_readiness"]
                + 0.20 * reintegrated_structure_anchor
                + 0.22 * reintegrated_local_compatibility
                + 0.16 * feedback_gain
                + 0.12 * (1.0 - metrics["domination_penalty"]),
            ),
        )
        reintegrated[mode] = {
            "feedback_gain": feedback_gain,
            "reintegrated_structure_anchor": reintegrated_structure_anchor,
            "reintegrated_local_compatibility": reintegrated_local_compatibility,
            "reintegrated_overall_readiness": reintegrated_overall_readiness,
            "domination_penalty": metrics["domination_penalty"],
        }

    best_name, best_metrics = max(reintegrated.items(), key=lambda item: item[1]["reintegrated_overall_readiness"])
    other_name = "log" if best_name == "sqrt" else "sqrt"
    reintegrated_margin = best_metrics["reintegrated_overall_readiness"] - reintegrated[other_name]["reintegrated_overall_readiness"]

    return {
        "headline_metrics": {
            "best_reintegrated_candidate_name": best_name,
            "best_reintegrated_overall_readiness": best_metrics["reintegrated_overall_readiness"],
            "best_reintegrated_structure_anchor": best_metrics["reintegrated_structure_anchor"],
            "best_reintegrated_local_compatibility": best_metrics["reintegrated_local_compatibility"],
            "best_feedback_gain": best_metrics["feedback_gain"],
            "reintegrated_margin": reintegrated_margin,
        },
        "reintegrated_candidates": reintegrated,
        "project_readout": {
            "summary": "Kernel feedback reintegration feeds fiber reuse and context grounding back into the bounded-law duel so the winner is chosen under coupled structural pressure instead of isolated scoring.",
            "next_question": "Use the reintegrated winner as the kernel entry point for live boundary-trigger checks and later multi-scale language-brain coupling tests.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage57 Kernel Feedback Reintegration",
        "",
        f"- best_reintegrated_candidate_name: {hm['best_reintegrated_candidate_name']}",
        f"- best_reintegrated_overall_readiness: {hm['best_reintegrated_overall_readiness']:.6f}",
        f"- best_reintegrated_structure_anchor: {hm['best_reintegrated_structure_anchor']:.6f}",
        f"- best_reintegrated_local_compatibility: {hm['best_reintegrated_local_compatibility']:.6f}",
        f"- best_feedback_gain: {hm['best_feedback_gain']:.6f}",
        f"- reintegrated_margin: {hm['reintegrated_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_kernel_feedback_reintegration_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
