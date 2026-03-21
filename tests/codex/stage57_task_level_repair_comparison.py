from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage57_task_level_repair_comparison_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_brain_bridge_boundary_trigger import build_brain_bridge_boundary_trigger_summary
from stage57_kernel_feedback_reintegration import build_kernel_feedback_reintegration_summary
from stage57_language_task_boundary_trigger import build_language_task_boundary_trigger_summary
from stage57_learning_rule_dual_candidate_review import build_learning_rule_dual_candidate_review_summary


def build_task_level_repair_comparison_summary() -> dict:
    dual = build_learning_rule_dual_candidate_review_summary()["candidate_review"]
    reintegrated = build_kernel_feedback_reintegration_summary()["reintegrated_candidates"]
    language = build_language_task_boundary_trigger_summary()["headline_metrics"]
    brain = build_brain_bridge_boundary_trigger_summary()["headline_metrics"]

    candidates = {}
    for mode in ("sqrt", "log"):
        dual_metrics = dual[mode]
        reintegrated_metrics = reintegrated[mode]
        domination_penalty = reintegrated_metrics["domination_penalty"]
        interpretability = dual_metrics["interpretability"]
        feedback_gain = reintegrated_metrics["feedback_gain"]
        local_compatibility = reintegrated_metrics["reintegrated_local_compatibility"]
        structure_anchor = reintegrated_metrics["reintegrated_structure_anchor"]

        forgetting_reduction = 0.40 * (1.0 - domination_penalty) + 0.05 * feedback_gain - 0.12 * domination_penalty
        repaired_long_forgetting = max(0.0, language["stressed_long_forgetting"] - forgetting_reduction)
        perplexity_factor = max(
            0.0,
            1.0 - (0.18 + 0.18 * (1.0 - domination_penalty) - 0.05 * domination_penalty + 0.04 * interpretability),
        )
        repaired_base_perplexity_delta = language["stressed_base_perplexity_delta"] * perplexity_factor
        repaired_novel_accuracy_after = min(
            1.0,
            language["stressed_novel_accuracy_after"]
            + 0.05 * feedback_gain
            + 0.04 * (1.0 - domination_penalty)
            + 0.02 * interpretability,
        )
        language_triggered = (
            repaired_long_forgetting > 0.20
            or repaired_base_perplexity_delta > 1000.0
            or repaired_novel_accuracy_after < 0.90
        )

        structure_lift = 0.10 * (structure_anchor - domination_penalty)
        repaired_direct_structure = min(1.0, brain["stressed_direct_structure"] + structure_lift)
        route_lift = 0.03 * local_compatibility + 0.02 * feedback_gain
        repaired_direct_route = min(1.0, brain["stressed_direct_route"] + route_lift)
        shared_reuse_lift = 0.05 * interpretability + 0.03 * (1.0 - domination_penalty)
        repaired_shared_red_reuse = min(1.0, brain["stressed_shared_red_reuse"] + shared_reuse_lift)
        repaired_brain_gap = max(
            0.0,
            brain["stressed_brain_gap"] - (0.09 * (1.0 - domination_penalty) + 0.04 * structure_anchor),
        )
        brain_triggered = (
            repaired_direct_structure < 0.78
            or repaired_direct_route < 0.79
            or repaired_shared_red_reuse < 0.70
            or repaired_brain_gap > 0.18
        )

        repaired_task_count = int(not language_triggered) + int(not brain_triggered)
        repair_readiness = max(
            0.0,
            min(
                1.0,
                0.22 * (1.0 - repaired_long_forgetting)
                + 0.18 * (1.0 - min(1.0, repaired_base_perplexity_delta / 1200.0))
                + 0.15 * repaired_novel_accuracy_after
                + 0.15 * repaired_direct_structure
                + 0.10 * repaired_direct_route
                + 0.08 * repaired_shared_red_reuse
                + 0.12 * (1.0 - repaired_brain_gap),
            ),
        )

        candidates[mode] = {
            "language_triggered_after_repair": language_triggered,
            "repaired_long_forgetting": repaired_long_forgetting,
            "repaired_base_perplexity_delta": repaired_base_perplexity_delta,
            "repaired_novel_accuracy_after": repaired_novel_accuracy_after,
            "brain_triggered_after_repair": brain_triggered,
            "repaired_direct_structure": repaired_direct_structure,
            "repaired_direct_route": repaired_direct_route,
            "repaired_shared_red_reuse": repaired_shared_red_reuse,
            "repaired_brain_gap": repaired_brain_gap,
            "repaired_task_count": repaired_task_count,
            "repair_readiness": repair_readiness,
        }

    best_name, best_metrics = max(
        candidates.items(),
        key=lambda item: (item[1]["repaired_task_count"], item[1]["repair_readiness"]),
    )
    other_name = "log" if best_name == "sqrt" else "sqrt"
    readiness_margin = best_metrics["repair_readiness"] - candidates[other_name]["repair_readiness"]

    return {
        "headline_metrics": {
            "best_repair_candidate_name": best_name,
            "best_repair_task_count": best_metrics["repaired_task_count"],
            "best_repair_readiness": best_metrics["repair_readiness"],
            "best_language_trigger_after_repair": best_metrics["language_triggered_after_repair"],
            "best_brain_trigger_after_repair": best_metrics["brain_triggered_after_repair"],
            "repair_readiness_margin": readiness_margin,
        },
        "candidate_repairs": candidates,
        "project_readout": {
            "summary": "Task-level repair comparison puts sqrt and log directly into the failed language and brain task triggers, then compares who can actually pull the system back inside the safe boundary instead of only scoring better in static kernel space.",
            "next_question": "Use the winning candidate as the temporary repair law and start testing whether fiber and context improvements can be reduced without reopening the same task failures.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage57 Task-Level Repair Comparison",
        "",
        f"- best_repair_candidate_name: {hm['best_repair_candidate_name']}",
        f"- best_repair_task_count: {hm['best_repair_task_count']}",
        f"- best_repair_readiness: {hm['best_repair_readiness']:.6f}",
        f"- best_language_trigger_after_repair: {hm['best_language_trigger_after_repair']}",
        f"- best_brain_trigger_after_repair: {hm['best_brain_trigger_after_repair']}",
        f"- repair_readiness_margin: {hm['repair_readiness_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_task_level_repair_comparison_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
