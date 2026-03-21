from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage57_minimal_repair_set_ablation_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_context_native_grounding import build_context_native_grounding_summary
from stage57_fiber_reuse_reinforcement import build_fiber_reuse_reinforcement_summary
from stage57_task_level_repair_comparison import build_task_level_repair_comparison_summary


def build_minimal_repair_set_ablation_summary() -> dict:
    repairs = build_task_level_repair_comparison_summary()
    sqrt_repair = repairs["candidate_repairs"]["sqrt"]
    fiber = build_fiber_reuse_reinforcement_summary()["headline_metrics"]
    context = build_context_native_grounding_summary()["headline_metrics"]

    cases = {
        "full_repair": {"fiber_scale": 1.0, "context_scale": 1.0},
        "drop_fiber": {"fiber_scale": 0.0, "context_scale": 1.0},
        "drop_context": {"fiber_scale": 1.0, "context_scale": 0.0},
        "drop_both": {"fiber_scale": 0.0, "context_scale": 0.0},
    }

    ablations = {}
    for name, scales in cases.items():
        fiber_loss = (1.0 - scales["fiber_scale"]) * (
            0.13 * fiber["fiber_reuse"] + 0.08 * fiber["reinforcement_readiness"] + 0.04 * fiber["pressure_under_reuse"]
        )
        context_loss = (1.0 - scales["context_scale"]) * (
            0.12 * context["context_native_readiness"] + 0.08 * context["context_route_alignment"] + 0.03 * context["conditional_gate_stability"]
        )
        total_loss = fiber_loss + context_loss

        lang_forgetting = min(1.0, sqrt_repair["repaired_long_forgetting"] + 0.60 * total_loss)
        lang_perplexity = sqrt_repair["repaired_base_perplexity_delta"] * (1.0 + 0.80 * total_loss)
        lang_novel_accuracy = max(0.0, sqrt_repair["repaired_novel_accuracy_after"] - 0.18 * total_loss)
        language_trigger = (
            lang_forgetting > 0.20
            or lang_perplexity > 1000.0
            or lang_novel_accuracy < 0.90
        )

        brain_structure = max(0.0, sqrt_repair["repaired_direct_structure"] - 0.25 * total_loss)
        brain_route = max(0.0, sqrt_repair["repaired_direct_route"] - 0.20 * total_loss)
        brain_shared = max(0.0, sqrt_repair["repaired_shared_red_reuse"] - 0.22 * total_loss)
        brain_gap = min(1.0, sqrt_repair["repaired_brain_gap"] + 0.24 * total_loss)
        brain_trigger = (
            brain_structure < 0.78
            or brain_route < 0.79
            or brain_shared < 0.70
            or brain_gap > 0.18
        )

        safe_task_count = int(not language_trigger) + int(not brain_trigger)
        ablation_readiness = max(
            0.0,
            min(
                1.0,
                0.22 * (1.0 - lang_forgetting)
                + 0.18 * (1.0 - min(1.0, lang_perplexity / 1200.0))
                + 0.15 * lang_novel_accuracy
                + 0.15 * brain_structure
                + 0.10 * brain_route
                + 0.08 * brain_shared
                + 0.12 * (1.0 - brain_gap),
            ),
        )

        ablations[name] = {
            "language_triggered": language_trigger,
            "brain_triggered": brain_trigger,
            "safe_task_count": safe_task_count,
            "lang_forgetting": lang_forgetting,
            "lang_perplexity_delta": lang_perplexity,
            "lang_novel_accuracy": lang_novel_accuracy,
            "brain_structure": brain_structure,
            "brain_route": brain_route,
            "brain_shared_red_reuse": brain_shared,
            "brain_gap": brain_gap,
            "ablation_readiness": ablation_readiness,
        }

    necessary_components = []
    if ablations["drop_fiber"]["safe_task_count"] < ablations["full_repair"]["safe_task_count"]:
        necessary_components.append("fiber_reuse")
    if ablations["drop_context"]["safe_task_count"] < ablations["full_repair"]["safe_task_count"]:
        necessary_components.append("context_grounding")

    minimum_joint_repair_required = ablations["drop_both"]["safe_task_count"] == 0

    return {
        "headline_metrics": {
            "full_repair_safe_task_count": ablations["full_repair"]["safe_task_count"],
            "drop_fiber_safe_task_count": ablations["drop_fiber"]["safe_task_count"],
            "drop_context_safe_task_count": ablations["drop_context"]["safe_task_count"],
            "drop_both_safe_task_count": ablations["drop_both"]["safe_task_count"],
            "minimum_joint_repair_required": minimum_joint_repair_required,
        },
        "necessary_components": necessary_components,
        "ablation_cases": ablations,
        "project_readout": {
            "summary": "Minimal repair-set ablation fixes sqrt as the current winner, then removes fiber reinforcement, context grounding, or both to see which components are truly necessary to keep language and brain tasks inside the safe region.",
            "next_question": "Use the necessary components list to distinguish core repair terms from temporary helper patches before scaling the current repair law further.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage57 Minimal Repair Set Ablation",
        "",
        f"- full_repair_safe_task_count: {hm['full_repair_safe_task_count']}",
        f"- drop_fiber_safe_task_count: {hm['drop_fiber_safe_task_count']}",
        f"- drop_context_safe_task_count: {hm['drop_context_safe_task_count']}",
        f"- drop_both_safe_task_count: {hm['drop_both_safe_task_count']}",
        f"- minimum_joint_repair_required: {hm['minimum_joint_repair_required']}",
        f"- necessary_components: {', '.join(summary['necessary_components'])}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_minimal_repair_set_ablation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
