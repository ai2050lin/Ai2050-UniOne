from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage58_repair_dependency_reduction_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_first_principles_transition_framework import build_first_principles_transition_framework_summary
from stage56_local_generative_law_emergence import build_local_generative_law_emergence_summary
from stage56_native_variable_candidate_mapping import build_native_variable_candidate_mapping_summary
from stage57_context_native_grounding import build_context_native_grounding_summary
from stage57_fiber_reuse_reinforcement import build_fiber_reuse_reinforcement_summary
from stage57_task_level_repair_comparison import build_task_level_repair_comparison_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_repair_dependency_reduction_summary() -> dict:
    fp = build_first_principles_transition_framework_summary()["headline_metrics"]
    local = build_local_generative_law_emergence_summary()["headline_metrics"]
    native = build_native_variable_candidate_mapping_summary()["headline_metrics"]
    fiber = build_fiber_reuse_reinforcement_summary()["headline_metrics"]
    context = build_context_native_grounding_summary()["headline_metrics"]
    sqrt_repair = build_task_level_repair_comparison_summary()["candidate_repairs"]["sqrt"]

    baseline_penalty = 1.0
    native_bridge = (
        0.40 * native["native_mapping_completeness"]
        + 0.30 * local["derivability_score"]
        + 0.30 * fp["local_law_closure"]
    )
    fiber_patch_strength = (
        0.38 * fiber["fiber_reuse"]
        + 0.32 * fiber["cross_region_share_stability"]
        + 0.30 * fiber["route_fiber_coupling_balance"]
    )
    context_patch_strength = (
        0.45 * context["context_native_readiness"]
        + 0.30 * context["conditional_gate_stability"]
        + 0.25 * context["context_route_alignment"]
    )
    fiber_native_substitute = 0.52 * native_bridge + 0.18 * local["patch_coherence"]
    context_native_substitute = 0.54 * native_bridge + 0.10 * context["context_route_alignment"]

    strategies = {
        "full_patch": {"fiber_explicit": 1.0, "context_explicit": 1.0, "synergy": 0.0},
        "fiber_nativeization": {"fiber_explicit": 0.35, "context_explicit": 1.0, "synergy": 0.0},
        "context_nativeization": {"fiber_explicit": 1.0, "context_explicit": 0.35, "synergy": 0.0},
        "joint_nativeization": {"fiber_explicit": 0.55, "context_explicit": 0.55, "synergy": 0.10},
    }

    results = {}
    for name, config in strategies.items():
        fiber_support = (
            config["fiber_explicit"] * fiber_patch_strength
            + (1.0 - config["fiber_explicit"]) * fiber_native_substitute
            + config["synergy"] * 0.35
        )
        context_support = (
            config["context_explicit"] * context_patch_strength
            + (1.0 - config["context_explicit"]) * context_native_substitute
            + config["synergy"] * 0.35
        )

        fiber_loss = max(0.0, fiber_patch_strength - fiber_support)
        context_loss = max(0.0, context_patch_strength - context_support)

        repaired_long_forgetting = _clip01(
            sqrt_repair["repaired_long_forgetting"]
            + 0.060 * context_loss
            + 0.015 * fiber_loss
            - 0.025 * config["synergy"]
        )
        repaired_base_perplexity_delta = sqrt_repair["repaired_base_perplexity_delta"] * (
            1.0
            + 0.46 * context_loss
            + 0.08 * fiber_loss
            - 0.15 * config["synergy"]
        )
        repaired_novel_accuracy_after = _clip01(
            sqrt_repair["repaired_novel_accuracy_after"]
            - 0.040 * context_loss
            - 0.012 * fiber_loss
            + 0.020 * config["synergy"]
        )
        language_triggered = (
            repaired_long_forgetting > 0.20
            or repaired_base_perplexity_delta > 1000.0
            or repaired_novel_accuracy_after < 0.90
        )

        repaired_direct_structure = _clip01(
            sqrt_repair["repaired_direct_structure"]
            - 0.065 * fiber_loss
            - 0.018 * context_loss
            + 0.080 * config["synergy"]
        )
        repaired_direct_route = _clip01(
            sqrt_repair["repaired_direct_route"]
            - 0.050 * fiber_loss
            - 0.025 * context_loss
            + 0.045 * config["synergy"]
        )
        repaired_shared_red_reuse = _clip01(
            sqrt_repair["repaired_shared_red_reuse"]
            - 0.045 * fiber_loss
            - 0.018 * context_loss
            + 0.040 * config["synergy"]
        )
        repaired_brain_gap = _clip01(
            sqrt_repair["repaired_brain_gap"]
            + 0.070 * fiber_loss
            + 0.055 * context_loss
            - 0.070 * config["synergy"]
        )
        brain_triggered = (
            repaired_direct_structure < 0.78
            or repaired_direct_route < 0.79
            or repaired_shared_red_reuse < 0.70
            or repaired_brain_gap > 0.18
        )

        safe_task_count = int(not language_triggered) + int(not brain_triggered)
        dependency_reduction_gain = (
            0.58 * (1.0 - 0.5 * (config["fiber_explicit"] + config["context_explicit"]))
            + 0.18 * config["synergy"]
            + 0.12 * native_bridge * (1.0 - 0.5 * (config["fiber_explicit"] + config["context_explicit"]))
        )
        reduced_dependency_penalty = _clip01(baseline_penalty - dependency_reduction_gain)
        reduced_repair_readiness = _clip01(
            0.22 * (1.0 - repaired_long_forgetting)
            + 0.18 * (1.0 - min(1.0, repaired_base_perplexity_delta / 1200.0))
            + 0.15 * repaired_novel_accuracy_after
            + 0.15 * repaired_direct_structure
            + 0.10 * repaired_direct_route
            + 0.08 * repaired_shared_red_reuse
            + 0.12 * (1.0 - repaired_brain_gap)
        )

        results[name] = {
            "fiber_support": fiber_support,
            "context_support": context_support,
            "fiber_loss": fiber_loss,
            "context_loss": context_loss,
            "language_triggered": language_triggered,
            "brain_triggered": brain_triggered,
            "safe_task_count": safe_task_count,
            "repaired_long_forgetting": repaired_long_forgetting,
            "repaired_base_perplexity_delta": repaired_base_perplexity_delta,
            "repaired_novel_accuracy_after": repaired_novel_accuracy_after,
            "repaired_direct_structure": repaired_direct_structure,
            "repaired_direct_route": repaired_direct_route,
            "repaired_shared_red_reuse": repaired_shared_red_reuse,
            "repaired_brain_gap": repaired_brain_gap,
            "reduced_dependency_penalty": reduced_dependency_penalty,
            "dependency_reduction_gain": dependency_reduction_gain,
            "reduced_repair_readiness": reduced_repair_readiness,
        }

    reduction_candidates = {k: v for k, v in results.items() if k != "full_patch"}
    best_name, best_metrics = max(
        reduction_candidates.items(),
        key=lambda item: (item[1]["safe_task_count"], -item[1]["reduced_dependency_penalty"], item[1]["reduced_repair_readiness"]),
    )

    return {
        "headline_metrics": {
            "baseline_dependency_penalty": baseline_penalty,
            "best_strategy_name": best_name,
            "best_safe_task_count": best_metrics["safe_task_count"],
            "reduced_dependency_penalty": best_metrics["reduced_dependency_penalty"],
            "dependency_reduction_gain": best_metrics["dependency_reduction_gain"],
            "best_reduced_repair_readiness": best_metrics["reduced_repair_readiness"],
        },
        "native_bridge_metrics": {
            "native_bridge": native_bridge,
            "fiber_patch_strength": fiber_patch_strength,
            "context_patch_strength": context_patch_strength,
            "fiber_native_substitute": fiber_native_substitute,
            "context_native_substitute": context_native_substitute,
        },
        "strategy_results": results,
        "project_readout": {
            "summary": "修复依赖削减轮把纤维补强和上下文补强从显式补丁部分压回原生变量支持，比较哪些降补丁策略还能保住语言任务与脑编码桥接任务。",
            "next_question": "下一步要继续压缩 joint nativeization 的显式依赖，看 reduced dependency penalty 是否还能在不丢任务安全的前提下继续下降。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage58 Repair Dependency Reduction",
        "",
        f"- best_strategy_name: {hm['best_strategy_name']}",
        f"- best_safe_task_count: {hm['best_safe_task_count']}",
        f"- baseline_dependency_penalty: {hm['baseline_dependency_penalty']:.6f}",
        f"- reduced_dependency_penalty: {hm['reduced_dependency_penalty']:.6f}",
        f"- dependency_reduction_gain: {hm['dependency_reduction_gain']:.6f}",
        f"- best_reduced_repair_readiness: {hm['best_reduced_repair_readiness']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_repair_dependency_reduction_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
