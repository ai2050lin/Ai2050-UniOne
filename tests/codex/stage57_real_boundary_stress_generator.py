from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage57_real_boundary_stress_generator_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_context_native_grounding import build_context_native_grounding_summary
from stage57_failure_boundary_trigger import _fiber_failure, _kernel_failure, _patch_failure, _route_failure
from stage57_fiber_reuse_reinforcement import build_fiber_reuse_reinforcement_summary
from stage57_kernel_feedback_reintegration import build_kernel_feedback_reintegration_summary

try:
    from stage56_large_scale_long_context_online_validation import build_large_scale_long_context_online_validation_summary
except Exception:  # pragma: no cover - fallback only matters when upstream files are missing
    build_large_scale_long_context_online_validation_summary = None


def _scale_metrics() -> tuple[str, dict]:
    fallback = {
        "scale_language_keep": 0.74,
        "scale_structure_keep": 0.70,
        "long_context_generalization": 0.69,
        "scale_forgetting_penalty": 0.24,
        "scale_collapse_risk": 0.28,
        "scale_readiness": 0.71,
    }
    if build_large_scale_long_context_online_validation_summary is None:
        return "fallback", fallback
    try:
        summary = build_large_scale_long_context_online_validation_summary()
        return "stage56_large_scale_long_context_online_validation", summary["headline_metrics"]
    except Exception:
        return "fallback", fallback


def build_real_boundary_stress_generator_summary() -> dict:
    reintegrated = build_kernel_feedback_reintegration_summary()
    fiber = build_fiber_reuse_reinforcement_summary()["headline_metrics"]
    context = build_context_native_grounding_summary()["headline_metrics"]
    scale_source, scale = _scale_metrics()

    best_name = reintegrated["headline_metrics"]["best_reintegrated_candidate_name"]
    best = reintegrated["reintegrated_candidates"][best_name]

    context_overload = 0.14 + 0.30 * scale["scale_collapse_risk"] + 0.20 * (1.0 - scale["long_context_generalization"])
    fiber_congestion = 0.12 + 0.35 * scale["scale_collapse_risk"] + 0.20 * scale["scale_forgetting_penalty"]
    kernel_rebound = 0.14 + 0.55 * scale["scale_forgetting_penalty"] + 0.20 * (1.0 - scale["scale_readiness"])
    patch_erosion = 0.11 + 0.30 * (1.0 - scale["scale_structure_keep"]) + 0.25 * scale["scale_collapse_risk"]

    scenarios = {
        "context_overload": {
            "context_route_alignment": max(0.0, context["context_route_alignment"] - context_overload),
            "pressure_under_reuse": max(0.0, fiber["pressure_under_reuse"] - 0.70 * context_overload),
        },
        "fiber_congestion_wave": {
            "fiber_reuse": max(0.0, fiber["fiber_reuse"] - 0.70 * fiber_congestion),
            "route_fiber_coupling_balance": max(0.0, fiber["route_fiber_coupling_balance"] - 0.45 * fiber_congestion),
        },
        "kernel_domination_rebound": {
            "domination_penalty": min(1.0, best["domination_penalty"] + 0.55 * kernel_rebound),
        },
        "coupled_patch_erosion": {
            "reintegrated_structure_anchor": max(0.0, best["reintegrated_structure_anchor"] - 0.78 * patch_erosion),
            "reintegrated_local_compatibility": max(0.0, best["reintegrated_local_compatibility"] - 0.46 * patch_erosion),
        },
    }

    scenario_results = {
        "context_overload": {
            "triggered": _route_failure(
                scenarios["context_overload"]["context_route_alignment"],
                scenarios["context_overload"]["pressure_under_reuse"],
            ),
            **scenarios["context_overload"],
        },
        "fiber_congestion_wave": {
            "triggered": _fiber_failure(
                scenarios["fiber_congestion_wave"]["fiber_reuse"],
                scenarios["fiber_congestion_wave"]["route_fiber_coupling_balance"],
            ),
            **scenarios["fiber_congestion_wave"],
        },
        "kernel_domination_rebound": {
            "triggered": _kernel_failure(scenarios["kernel_domination_rebound"]["domination_penalty"]),
            **scenarios["kernel_domination_rebound"],
        },
        "coupled_patch_erosion": {
            "triggered": _patch_failure(
                scenarios["coupled_patch_erosion"]["reintegrated_structure_anchor"],
                scenarios["coupled_patch_erosion"]["reintegrated_local_compatibility"],
            ),
            **scenarios["coupled_patch_erosion"],
        },
    }

    triggered_count = sum(1 for item in scenario_results.values() if item["triggered"])
    real_trigger_rate = triggered_count / len(scenario_results)
    scale_bridge_factor = max(
        0.0,
        min(
            1.0,
            0.35 * scale["scale_readiness"]
            + 0.35 * (1.0 - scale["scale_collapse_risk"])
            + 0.30 * scale["long_context_generalization"],
        ),
    )
    stress_generator_readiness = max(
        0.0,
        min(1.0, 0.45 * real_trigger_rate + 0.30 * scale_bridge_factor + 0.25 * (1.0 - scale["scale_forgetting_penalty"])),
    )

    return {
        "headline_metrics": {
            "real_trigger_rate": real_trigger_rate,
            "triggered_case_count": triggered_count,
            "scale_bridge_factor": scale_bridge_factor,
            "stress_generator_readiness": stress_generator_readiness,
        },
        "scale_source": scale_source,
        "scale_metrics": scale,
        "scenario_results": scenario_results,
        "project_readout": {
            "summary": "Real boundary stress generation derives failure scenarios from the current reintegrated kernel plus long-context and scale metrics, then checks whether patch, fiber, route, and kernel boundaries can be broken by structured perturbations instead of hand-set numbers.",
            "next_question": "Move from structured synthetic stress to task-level stress, so each failure case is produced by an actual language or brain-side workload instead of a metric-level perturbation.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage57 Real Boundary Stress Generator",
        "",
        f"- scale_source: {summary['scale_source']}",
        f"- real_trigger_rate: {hm['real_trigger_rate']:.6f}",
        f"- triggered_case_count: {hm['triggered_case_count']}",
        f"- scale_bridge_factor: {hm['scale_bridge_factor']:.6f}",
        f"- stress_generator_readiness: {hm['stress_generator_readiness']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_real_boundary_stress_generator_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
