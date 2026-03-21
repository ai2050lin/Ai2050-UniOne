from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage57_brain_bridge_boundary_trigger_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_real_boundary_stress_generator import build_real_boundary_stress_generator_summary


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_brain_bridge_boundary_trigger_summary() -> dict:
    brain = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v39_20260321" / "summary.json"
    )["headline_metrics"]
    proto = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_object_attribute_structure_prototype_20260320" / "summary.json"
    )["headline_metrics"]
    stress = build_real_boundary_stress_generator_summary()

    fiber_wave = stress["scenario_results"]["fiber_congestion_wave"]
    patch_erosion = stress["scenario_results"]["coupled_patch_erosion"]
    rebound = stress["scenario_results"]["kernel_domination_rebound"]
    scale = stress["scale_metrics"]

    stressed_direct_structure = max(
        0.0,
        brain["direct_structure_measure_v39"]
        - 0.18 * (0.45 - fiber_wave["fiber_reuse"] if fiber_wave["fiber_reuse"] < 0.45 else 0.0)
        - 0.35 * (brain["direct_structure_measure_v39"] - patch_erosion["reintegrated_structure_anchor"]),
    )
    stressed_direct_route = max(
        0.0,
        brain["direct_route_measure_v39"]
        - 0.10 * (1.0 - fiber_wave["route_fiber_coupling_balance"])
        - 0.12 * scale["scale_collapse_risk"],
    )
    stressed_shared_red_reuse = max(
        0.0,
        proto["shared_red_reuse"]
        - 0.40 * (proto["context_route_split"])
        - 0.12 * rebound["domination_penalty"],
    )
    stressed_brain_gap = min(
        1.0,
        1.0 - (0.28 * brain["direct_origin_measure_v39"] + 0.24 * brain["direct_feature_measure_v39"] + 0.24 * stressed_direct_structure + 0.24 * stressed_direct_route),
    )

    trigger = (
        stressed_direct_structure < 0.78
        or stressed_direct_route < 0.79
        or stressed_shared_red_reuse < 0.70
        or stressed_brain_gap > 0.18
    )

    bridge_boundary_readiness = max(
        0.0,
        min(
            1.0,
            0.28 * stressed_direct_structure
            + 0.24 * stressed_direct_route
            + 0.22 * stressed_shared_red_reuse
            + 0.26 * (1.0 - stressed_brain_gap),
        ),
    )

    return {
        "headline_metrics": {
            "stressed_direct_structure": stressed_direct_structure,
            "stressed_direct_route": stressed_direct_route,
            "stressed_shared_red_reuse": stressed_shared_red_reuse,
            "stressed_brain_gap": stressed_brain_gap,
            "bridge_boundary_readiness": bridge_boundary_readiness,
        },
        "bridge_trigger": {
            "triggered": trigger,
            "reason": "brain bridge fails when structural or route direct measures fall, shared attribute reuse collapses, or the aggregate brain gap reopens.",
        },
        "project_readout": {
            "summary": "Brain bridge boundary trigger injects fiber congestion, patch erosion, and kernel rebound into the current brain-encoding bridge to check whether a real bridge-level failure appears instead of only a metric-layer warning.",
            "next_question": "Use the triggered bridge failure to decide whether fiber reinforcement, context grounding, or the bounded learning law is the first repair target on the brain side.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage57 Brain Bridge Boundary Trigger",
        "",
        f"- stressed_direct_structure: {hm['stressed_direct_structure']:.6f}",
        f"- stressed_direct_route: {hm['stressed_direct_route']:.6f}",
        f"- stressed_shared_red_reuse: {hm['stressed_shared_red_reuse']:.6f}",
        f"- stressed_brain_gap: {hm['stressed_brain_gap']:.6f}",
        f"- bridge_boundary_readiness: {hm['bridge_boundary_readiness']:.6f}",
        f"- triggered: {summary['bridge_trigger']['triggered']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_bridge_boundary_trigger_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
