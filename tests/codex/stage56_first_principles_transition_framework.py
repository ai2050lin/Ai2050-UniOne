from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_first_principles_transition_framework_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_first_principles_transition_framework_summary() -> dict:
    bounded = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_term_boundedization_20260321" / "summary.json"
    )["headline_metrics"]
    native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_native_variable_candidate_mapping_20260321" / "summary.json"
    )["headline_metrics"]
    brain_v39 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v39_20260321" / "summary.json"
    )["headline_metrics"]
    bridge_v45 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v45_20260321" / "summary.json"
    )["headline_metrics"]

    primitive_transition_readiness = min(
        1.0,
        0.28 * bounded["bounded_readiness"]
        + 0.24 * native["primitive_set_readiness"]
        + 0.24 * brain_v39["direct_brain_measure_v39"]
        + 0.24 * (1.0 - bridge_v45["topology_training_gap_v45"]),
    )
    local_law_closure = min(
        1.0,
        0.45 * native["native_mapping_completeness"]
        + 0.30 * bounded["bounded_stability_score"]
        + 0.25 * brain_v39["direct_structure_measure_v39"],
    )
    falsifiability_upgrade = min(
        1.0,
        0.35 * native["native_mapping_completeness"]
        + 0.25 * (1.0 - bounded["raw_domination_penalty"])
        + 0.40 * (1.0 - bridge_v45["topology_training_gap_v45"]),
    )
    first_principles_transition_score = (
        primitive_transition_readiness * 0.4
        + local_law_closure * 0.35
        + falsifiability_upgrade * 0.25
    )

    return {
        "headline_metrics": {
            "primitive_transition_readiness": primitive_transition_readiness,
            "local_law_closure": local_law_closure,
            "falsifiability_upgrade": falsifiability_upgrade,
            "first_principles_transition_score": first_principles_transition_score,
        },
        "framework_equation": {
            "primitive_transition": "T_fp = mix(B_bounded, M_native, D_brain, 1 - G_train)",
            "local_law": "L_fp = mix(M_native_complete, B_stable, D_structure)",
            "falsifiability": "F_fp = mix(M_native_complete, 1 - P_dominate, 1 - G_train)",
            "system": "S_fp = 0.4 * T_fp + 0.35 * L_fp + 0.25 * F_fp",
        },
        "project_readout": {
            "summary": "first-principles transition framework estimates whether the project has enough bounded learning, native primitive mapping, and direct-brain alignment to leave the phenomenological regime.",
            "next_question": "next instantiate candidate local update laws and verify whether patches, fibers, and routes become derivable instead of manually named.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# First-Principles Transition Framework Report",
        "",
        f"- primitive_transition_readiness: {hm['primitive_transition_readiness']:.6f}",
        f"- local_law_closure: {hm['local_law_closure']:.6f}",
        f"- falsifiability_upgrade: {hm['falsifiability_upgrade']:.6f}",
        f"- first_principles_transition_score: {hm['first_principles_transition_score']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_first_principles_transition_framework_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
