from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_mesh_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_systemic_low_risk_field_mesh_validation_summary() -> dict:
    lattice = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_lattice_validation_20260321" / "summary.json"
    )
    brain_v37 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v37_20260321" / "summary.json"
    )
    bridge_v43 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v43_20260321" / "summary.json"
    )
    v99 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v99_20260321" / "summary.json"
    )

    hs = lattice["headline_metrics"]
    hb = brain_v37["headline_metrics"]
    ht = bridge_v43["headline_metrics"]
    hm = v99["headline_metrics"]

    systemic_low_risk_field_mesh = _clip01(
        hs["systemic_low_risk_field_lattice_score"] * 0.31
        + hs["systemic_low_risk_field_lattice_readiness"] * 0.19
        + (1.0 - hs["systemic_low_risk_field_lattice_penalty"]) * 0.16
        + hb["direct_brain_measure_v37"] * 0.16
        + ht["topology_training_readiness_v43"] * 0.18
    )
    systemic_low_risk_field_structure_mesh = _clip01(
        hs["systemic_low_risk_field_structure_lattice"] * 0.38
        + hs["systemic_low_risk_field_route_lattice"] * 0.08
        + hb["direct_structure_measure_v37"] * 0.24
        + ht["structure_rule_alignment_v43"] * 0.30
    )
    systemic_low_risk_field_route_mesh = _clip01(
        hs["systemic_low_risk_field_route_lattice"] * 0.38
        + hs["systemic_low_risk_field_structure_lattice"] * 0.08
        + hb["direct_route_measure_v37"] * 0.24
        + ht["systemic_low_risk_field_lattice_guard_v43"] * 0.30
    )
    systemic_low_risk_field_learning_mesh = _clip01(
        hs["systemic_low_risk_field_learning_lattice"] * 0.28
        + hs["systemic_low_risk_field_lattice_score"] * 0.12
        + hb["direct_feature_measure_v37"] * 0.18
        + ht["plasticity_rule_alignment_v43"] * 0.20
        + ht["topology_training_readiness_v43"] * 0.22
    )
    systemic_low_risk_field_mesh_penalty = _clip01(
        hs["systemic_low_risk_field_lattice_penalty"] * 0.35
        + hb["direct_brain_gap_v37"] * 0.21
        + ht["topology_training_gap_v43"] * 0.21
        + (1.0 - hs["systemic_low_risk_field_lattice_score"]) * 0.05
        + hm["pressure_term_v99"] * 1e-3 * 0.18
    )
    systemic_low_risk_field_mesh_readiness = _clip01(
        (
            systemic_low_risk_field_mesh
            + systemic_low_risk_field_structure_mesh
            + systemic_low_risk_field_route_mesh
            + systemic_low_risk_field_learning_mesh
            + (1.0 - systemic_low_risk_field_mesh_penalty)
        )
        / 5.0
    )
    systemic_low_risk_field_mesh_score = _clip01(
        (
            systemic_low_risk_field_mesh_readiness
            + systemic_low_risk_field_mesh
            + systemic_low_risk_field_route_mesh
            + systemic_low_risk_field_learning_mesh
            + (1.0 - systemic_low_risk_field_mesh_penalty)
        )
        / 5.0
    )
    systemic_low_risk_field_mesh_margin = (
        systemic_low_risk_field_mesh
        + systemic_low_risk_field_structure_mesh
        + systemic_low_risk_field_route_mesh
        + systemic_low_risk_field_learning_mesh
        + systemic_low_risk_field_mesh_readiness
        + systemic_low_risk_field_mesh_score
        - systemic_low_risk_field_mesh_penalty
        + hm["encoding_margin_v99"] * 1e-15
    )

    return {
        "headline_metrics": {
            "systemic_low_risk_field_mesh": systemic_low_risk_field_mesh,
            "systemic_low_risk_field_structure_mesh": systemic_low_risk_field_structure_mesh,
            "systemic_low_risk_field_route_mesh": systemic_low_risk_field_route_mesh,
            "systemic_low_risk_field_learning_mesh": systemic_low_risk_field_learning_mesh,
            "systemic_low_risk_field_mesh_penalty": systemic_low_risk_field_mesh_penalty,
            "systemic_low_risk_field_mesh_readiness": systemic_low_risk_field_mesh_readiness,
            "systemic_low_risk_field_mesh_score": systemic_low_risk_field_mesh_score,
            "systemic_low_risk_field_mesh_margin": systemic_low_risk_field_mesh_margin,
        },
        "systemic_low_risk_field_mesh_equation": {
            "strength_term": "A_sys_field_mesh = mix(S_sys_field_lattice_score, R_sys_field_lattice, 1 - P_sys_field_lattice, M_brain_direct_v37, R_train_v43)",
            "structure_term": "S_sys_field_mesh = mix(S_sys_field_lattice, R_sys_field_lattice, D_structure_v37, B_struct_v43)",
            "route_term": "R_sys_field_mesh = mix(R_sys_field_lattice, S_sys_field_lattice, D_route_v37, H_sys_field_lattice_v43)",
            "learning_term": "L_sys_field_mesh = mix(L_sys_field_lattice, S_sys_field_lattice_score, D_feature_v37, B_plastic_v43, R_train_v43)",
            "system_term": "M_sys_field_mesh = A_sys_field_mesh + S_sys_field_mesh + R_sys_field_mesh + L_sys_field_mesh - P_sys_field_lattice",
        },
        "project_readout": {
            "summary": "systemic low-risk field mesh validation checks whether the lattice-like low-risk field begins turning into a more regular and resilient low-risk field mesh.",
            "next_question": "next verify whether this mesh-like low-risk field still survives after it is folded into the next brain direct chain and training bridge.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Systemic Low-Risk Field Mesh Validation Report",
        "",
        f"- systemic_low_risk_field_mesh: {hm['systemic_low_risk_field_mesh']:.6f}",
        f"- systemic_low_risk_field_structure_mesh: {hm['systemic_low_risk_field_structure_mesh']:.6f}",
        f"- systemic_low_risk_field_route_mesh: {hm['systemic_low_risk_field_route_mesh']:.6f}",
        f"- systemic_low_risk_field_learning_mesh: {hm['systemic_low_risk_field_learning_mesh']:.6f}",
        f"- systemic_low_risk_field_mesh_penalty: {hm['systemic_low_risk_field_mesh_penalty']:.6f}",
        f"- systemic_low_risk_field_mesh_readiness: {hm['systemic_low_risk_field_mesh_readiness']:.6f}",
        f"- systemic_low_risk_field_mesh_score: {hm['systemic_low_risk_field_mesh_score']:.6f}",
        f"- systemic_low_risk_field_mesh_margin: {hm['systemic_low_risk_field_mesh_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_systemic_low_risk_field_mesh_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
