from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_fabric_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_systemic_low_risk_field_fabric_validation_summary() -> dict:
    mesh = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_mesh_validation_20260321" / "summary.json"
    )
    brain_v38 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v38_20260321" / "summary.json"
    )
    bridge_v44 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v44_20260321" / "summary.json"
    )
    v100 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v100_20260321" / "summary.json"
    )

    hs = mesh["headline_metrics"]
    hb = brain_v38["headline_metrics"]
    ht = bridge_v44["headline_metrics"]
    hm = v100["headline_metrics"]

    systemic_low_risk_field_fabric = _clip01(
        hs["systemic_low_risk_field_mesh_score"] * 0.31
        + hs["systemic_low_risk_field_mesh_readiness"] * 0.19
        + (1.0 - hs["systemic_low_risk_field_mesh_penalty"]) * 0.16
        + hb["direct_brain_measure_v38"] * 0.16
        + ht["topology_training_readiness_v44"] * 0.18
    )
    systemic_low_risk_field_structure_fabric = _clip01(
        hs["systemic_low_risk_field_structure_mesh"] * 0.38
        + hs["systemic_low_risk_field_route_mesh"] * 0.08
        + hb["direct_structure_measure_v38"] * 0.24
        + ht["structure_rule_alignment_v44"] * 0.30
    )
    systemic_low_risk_field_route_fabric = _clip01(
        hs["systemic_low_risk_field_route_mesh"] * 0.38
        + hs["systemic_low_risk_field_structure_mesh"] * 0.08
        + hb["direct_route_measure_v38"] * 0.24
        + ht["systemic_low_risk_field_mesh_guard_v44"] * 0.30
    )
    systemic_low_risk_field_learning_fabric = _clip01(
        hs["systemic_low_risk_field_learning_mesh"] * 0.28
        + hs["systemic_low_risk_field_mesh_score"] * 0.12
        + hb["direct_feature_measure_v38"] * 0.18
        + ht["plasticity_rule_alignment_v44"] * 0.20
        + ht["topology_training_readiness_v44"] * 0.22
    )
    systemic_low_risk_field_fabric_penalty = _clip01(
        hs["systemic_low_risk_field_mesh_penalty"] * 0.35
        + hb["direct_brain_gap_v38"] * 0.21
        + ht["topology_training_gap_v44"] * 0.21
        + (1.0 - hs["systemic_low_risk_field_mesh_score"]) * 0.05
        + hm["pressure_term_v100"] * 1e-3 * 0.18
    )
    systemic_low_risk_field_fabric_readiness = _clip01(
        (
            systemic_low_risk_field_fabric
            + systemic_low_risk_field_structure_fabric
            + systemic_low_risk_field_route_fabric
            + systemic_low_risk_field_learning_fabric
            + (1.0 - systemic_low_risk_field_fabric_penalty)
        )
        / 5.0
    )
    systemic_low_risk_field_fabric_score = _clip01(
        (
            systemic_low_risk_field_fabric_readiness
            + systemic_low_risk_field_fabric
            + systemic_low_risk_field_route_fabric
            + systemic_low_risk_field_learning_fabric
            + (1.0 - systemic_low_risk_field_fabric_penalty)
        )
        / 5.0
    )
    systemic_low_risk_field_fabric_margin = (
        systemic_low_risk_field_fabric
        + systemic_low_risk_field_structure_fabric
        + systemic_low_risk_field_route_fabric
        + systemic_low_risk_field_learning_fabric
        + systemic_low_risk_field_fabric_readiness
        + systemic_low_risk_field_fabric_score
        - systemic_low_risk_field_fabric_penalty
        + hm["encoding_margin_v100"] * 1e-15
    )

    return {
        "headline_metrics": {
            "systemic_low_risk_field_fabric": systemic_low_risk_field_fabric,
            "systemic_low_risk_field_structure_fabric": systemic_low_risk_field_structure_fabric,
            "systemic_low_risk_field_route_fabric": systemic_low_risk_field_route_fabric,
            "systemic_low_risk_field_learning_fabric": systemic_low_risk_field_learning_fabric,
            "systemic_low_risk_field_fabric_penalty": systemic_low_risk_field_fabric_penalty,
            "systemic_low_risk_field_fabric_readiness": systemic_low_risk_field_fabric_readiness,
            "systemic_low_risk_field_fabric_score": systemic_low_risk_field_fabric_score,
            "systemic_low_risk_field_fabric_margin": systemic_low_risk_field_fabric_margin,
        },
        "systemic_low_risk_field_fabric_equation": {
            "strength_term": "A_sys_field_fabric = mix(S_sys_field_mesh_score, R_sys_field_mesh, 1 - P_sys_field_mesh, M_brain_direct_v38, R_train_v44)",
            "structure_term": "S_sys_field_fabric = mix(S_sys_field_mesh, R_sys_field_mesh, D_structure_v38, B_struct_v44)",
            "route_term": "R_sys_field_fabric = mix(R_sys_field_mesh, S_sys_field_mesh, D_route_v38, H_sys_field_mesh_v44)",
            "learning_term": "L_sys_field_fabric = mix(L_sys_field_mesh, S_sys_field_mesh_score, D_feature_v38, B_plastic_v44, R_train_v44)",
            "system_term": "M_sys_field_fabric = A_sys_field_fabric + S_sys_field_fabric + R_sys_field_fabric + L_sys_field_fabric - P_sys_field_mesh",
        },
        "project_readout": {
            "summary": "systemic low-risk field fabric validation checks whether the mesh-like low-risk field begins turning into a more interlocked and persistent low-risk field fabric.",
            "next_question": "next verify whether this fabric-like low-risk field still survives after it is folded into the next brain direct chain and training bridge.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Systemic Low-Risk Field Fabric Validation Report",
        "",
        f"- systemic_low_risk_field_fabric: {hm['systemic_low_risk_field_fabric']:.6f}",
        f"- systemic_low_risk_field_structure_fabric: {hm['systemic_low_risk_field_structure_fabric']:.6f}",
        f"- systemic_low_risk_field_route_fabric: {hm['systemic_low_risk_field_route_fabric']:.6f}",
        f"- systemic_low_risk_field_learning_fabric: {hm['systemic_low_risk_field_learning_fabric']:.6f}",
        f"- systemic_low_risk_field_fabric_penalty: {hm['systemic_low_risk_field_fabric_penalty']:.6f}",
        f"- systemic_low_risk_field_fabric_readiness: {hm['systemic_low_risk_field_fabric_readiness']:.6f}",
        f"- systemic_low_risk_field_fabric_score: {hm['systemic_low_risk_field_fabric_score']:.6f}",
        f"- systemic_low_risk_field_fabric_margin: {hm['systemic_low_risk_field_fabric_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_systemic_low_risk_field_fabric_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
