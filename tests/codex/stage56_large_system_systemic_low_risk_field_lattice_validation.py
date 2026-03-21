from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_lattice_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_systemic_low_risk_field_lattice_validation_summary() -> dict:
    crystal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_crystallization_validation_20260321" / "summary.json"
    )
    brain_v36 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v36_20260321" / "summary.json"
    )
    bridge_v42 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v42_20260321" / "summary.json"
    )
    v98 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v98_20260321" / "summary.json"
    )

    hs = crystal["headline_metrics"]
    hb = brain_v36["headline_metrics"]
    ht = bridge_v42["headline_metrics"]
    hm = v98["headline_metrics"]

    systemic_low_risk_field_lattice = _clip01(
        hs["systemic_low_risk_field_crystallization_score"] * 0.31
        + hs["systemic_low_risk_field_crystallization_readiness"] * 0.19
        + (1.0 - hs["systemic_low_risk_field_crystallization_penalty"]) * 0.16
        + hb["direct_brain_measure_v36"] * 0.16
        + ht["topology_training_readiness_v42"] * 0.18
    )
    systemic_low_risk_field_structure_lattice = _clip01(
        hs["systemic_low_risk_field_structure_crystallization"] * 0.38
        + hs["systemic_low_risk_field_route_crystallization"] * 0.08
        + hb["direct_structure_measure_v36"] * 0.24
        + ht["structure_rule_alignment_v42"] * 0.30
    )
    systemic_low_risk_field_route_lattice = _clip01(
        hs["systemic_low_risk_field_route_crystallization"] * 0.38
        + hs["systemic_low_risk_field_structure_crystallization"] * 0.08
        + hb["direct_route_measure_v36"] * 0.24
        + ht["systemic_low_risk_field_crystallization_guard_v42"] * 0.30
    )
    systemic_low_risk_field_learning_lattice = _clip01(
        hs["systemic_low_risk_field_learning_crystallization"] * 0.28
        + hs["systemic_low_risk_field_crystallization_score"] * 0.12
        + hb["direct_feature_measure_v36"] * 0.18
        + ht["plasticity_rule_alignment_v42"] * 0.20
        + ht["topology_training_readiness_v42"] * 0.22
    )
    systemic_low_risk_field_lattice_penalty = _clip01(
        hs["systemic_low_risk_field_crystallization_penalty"] * 0.35
        + hb["direct_brain_gap_v36"] * 0.21
        + ht["topology_training_gap_v42"] * 0.21
        + (1.0 - hs["systemic_low_risk_field_crystallization_score"]) * 0.05
        + hm["pressure_term_v98"] * 1e-3 * 0.18
    )
    systemic_low_risk_field_lattice_readiness = _clip01(
        (
            systemic_low_risk_field_lattice
            + systemic_low_risk_field_structure_lattice
            + systemic_low_risk_field_route_lattice
            + systemic_low_risk_field_learning_lattice
            + (1.0 - systemic_low_risk_field_lattice_penalty)
        )
        / 5.0
    )
    systemic_low_risk_field_lattice_score = _clip01(
        (
            systemic_low_risk_field_lattice_readiness
            + systemic_low_risk_field_lattice
            + systemic_low_risk_field_route_lattice
            + systemic_low_risk_field_learning_lattice
            + (1.0 - systemic_low_risk_field_lattice_penalty)
        )
        / 5.0
    )
    systemic_low_risk_field_lattice_margin = (
        systemic_low_risk_field_lattice
        + systemic_low_risk_field_structure_lattice
        + systemic_low_risk_field_route_lattice
        + systemic_low_risk_field_learning_lattice
        + systemic_low_risk_field_lattice_readiness
        + systemic_low_risk_field_lattice_score
        - systemic_low_risk_field_lattice_penalty
        + hm["encoding_margin_v98"] * 1e-15
    )

    return {
        "headline_metrics": {
            "systemic_low_risk_field_lattice": systemic_low_risk_field_lattice,
            "systemic_low_risk_field_structure_lattice": systemic_low_risk_field_structure_lattice,
            "systemic_low_risk_field_route_lattice": systemic_low_risk_field_route_lattice,
            "systemic_low_risk_field_learning_lattice": systemic_low_risk_field_learning_lattice,
            "systemic_low_risk_field_lattice_penalty": systemic_low_risk_field_lattice_penalty,
            "systemic_low_risk_field_lattice_readiness": systemic_low_risk_field_lattice_readiness,
            "systemic_low_risk_field_lattice_score": systemic_low_risk_field_lattice_score,
            "systemic_low_risk_field_lattice_margin": systemic_low_risk_field_lattice_margin,
        },
        "systemic_low_risk_field_lattice_equation": {
            "strength_term": "A_sys_field_lattice = mix(S_sys_field_crystal_score, R_sys_field_crystal, 1 - P_sys_field_crystal, M_brain_direct_v36, R_train_v42)",
            "structure_term": "S_sys_field_lattice = mix(S_sys_field_crystal, R_sys_field_crystal, D_structure_v36, B_struct_v42)",
            "route_term": "R_sys_field_lattice = mix(R_sys_field_crystal, S_sys_field_crystal, D_route_v36, H_sys_field_crystal_v42)",
            "learning_term": "L_sys_field_lattice = mix(L_sys_field_crystal, S_sys_field_crystal_score, D_feature_v36, B_plastic_v42, R_train_v42)",
            "system_term": "M_sys_field_lattice = A_sys_field_lattice + S_sys_field_lattice + R_sys_field_lattice + L_sys_field_lattice - P_sys_field_crystal",
        },
        "project_readout": {
            "summary": "systemic low-risk field lattice validation checks whether the crystallized low-risk field begins turning into a more regular and persistent low-risk field lattice.",
            "next_question": "next verify whether this lattice-like low-risk field still survives after it is folded into the next brain direct chain and training bridge.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Systemic Low-Risk Field Lattice Validation Report",
        "",
        f"- systemic_low_risk_field_lattice: {hm['systemic_low_risk_field_lattice']:.6f}",
        f"- systemic_low_risk_field_structure_lattice: {hm['systemic_low_risk_field_structure_lattice']:.6f}",
        f"- systemic_low_risk_field_route_lattice: {hm['systemic_low_risk_field_route_lattice']:.6f}",
        f"- systemic_low_risk_field_learning_lattice: {hm['systemic_low_risk_field_learning_lattice']:.6f}",
        f"- systemic_low_risk_field_lattice_penalty: {hm['systemic_low_risk_field_lattice_penalty']:.6f}",
        f"- systemic_low_risk_field_lattice_readiness: {hm['systemic_low_risk_field_lattice_readiness']:.6f}",
        f"- systemic_low_risk_field_lattice_score: {hm['systemic_low_risk_field_lattice_score']:.6f}",
        f"- systemic_low_risk_field_lattice_margin: {hm['systemic_low_risk_field_lattice_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_systemic_low_risk_field_lattice_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
