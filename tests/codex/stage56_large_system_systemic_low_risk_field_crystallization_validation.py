from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_crystallization_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_systemic_low_risk_field_crystallization_validation_summary() -> dict:
    solid = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_solidification_validation_20260321" / "summary.json"
    )
    brain_v35 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v35_20260321" / "summary.json"
    )
    bridge_v41 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v41_20260321" / "summary.json"
    )
    v97 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v97_20260321" / "summary.json"
    )

    hs = solid["headline_metrics"]
    hb = brain_v35["headline_metrics"]
    ht = bridge_v41["headline_metrics"]
    hm = v97["headline_metrics"]

    systemic_low_risk_field_crystallization = _clip01(
        hs["systemic_low_risk_field_solidification_score"] * 0.31
        + hs["systemic_low_risk_field_solidification_readiness"] * 0.19
        + (1.0 - hs["systemic_low_risk_field_solidification_penalty"]) * 0.16
        + hb["direct_brain_measure_v35"] * 0.16
        + ht["topology_training_readiness_v41"] * 0.18
    )
    systemic_low_risk_field_structure_crystallization = _clip01(
        hs["systemic_low_risk_field_structure_solidification"] * 0.38
        + hs["systemic_low_risk_field_route_solidification"] * 0.08
        + hb["direct_structure_measure_v35"] * 0.24
        + ht["structure_rule_alignment_v41"] * 0.30
    )
    systemic_low_risk_field_route_crystallization = _clip01(
        hs["systemic_low_risk_field_route_solidification"] * 0.38
        + hs["systemic_low_risk_field_structure_solidification"] * 0.08
        + hb["direct_route_measure_v35"] * 0.24
        + ht["systemic_low_risk_field_solidification_guard_v41"] * 0.30
    )
    systemic_low_risk_field_learning_crystallization = _clip01(
        hs["systemic_low_risk_field_learning_solidification"] * 0.28
        + hs["systemic_low_risk_field_solidification_score"] * 0.12
        + hb["direct_feature_measure_v35"] * 0.18
        + ht["plasticity_rule_alignment_v41"] * 0.20
        + ht["topology_training_readiness_v41"] * 0.22
    )
    systemic_low_risk_field_crystallization_penalty = _clip01(
        hs["systemic_low_risk_field_solidification_penalty"] * 0.35
        + hb["direct_brain_gap_v35"] * 0.21
        + ht["topology_training_gap_v41"] * 0.21
        + (1.0 - hs["systemic_low_risk_field_solidification_score"]) * 0.05
        + hm["pressure_term_v97"] * 1e-3 * 0.18
    )
    systemic_low_risk_field_crystallization_readiness = _clip01(
        (
            systemic_low_risk_field_crystallization
            + systemic_low_risk_field_structure_crystallization
            + systemic_low_risk_field_route_crystallization
            + systemic_low_risk_field_learning_crystallization
            + (1.0 - systemic_low_risk_field_crystallization_penalty)
        )
        / 5.0
    )
    systemic_low_risk_field_crystallization_score = _clip01(
        (
            systemic_low_risk_field_crystallization_readiness
            + systemic_low_risk_field_crystallization
            + systemic_low_risk_field_route_crystallization
            + systemic_low_risk_field_learning_crystallization
            + (1.0 - systemic_low_risk_field_crystallization_penalty)
        )
        / 5.0
    )
    systemic_low_risk_field_crystallization_margin = (
        systemic_low_risk_field_crystallization
        + systemic_low_risk_field_structure_crystallization
        + systemic_low_risk_field_route_crystallization
        + systemic_low_risk_field_learning_crystallization
        + systemic_low_risk_field_crystallization_readiness
        + systemic_low_risk_field_crystallization_score
        - systemic_low_risk_field_crystallization_penalty
        + hm["encoding_margin_v97"] * 1e-15
    )

    return {
        "headline_metrics": {
            "systemic_low_risk_field_crystallization": systemic_low_risk_field_crystallization,
            "systemic_low_risk_field_structure_crystallization": systemic_low_risk_field_structure_crystallization,
            "systemic_low_risk_field_route_crystallization": systemic_low_risk_field_route_crystallization,
            "systemic_low_risk_field_learning_crystallization": systemic_low_risk_field_learning_crystallization,
            "systemic_low_risk_field_crystallization_penalty": systemic_low_risk_field_crystallization_penalty,
            "systemic_low_risk_field_crystallization_readiness": systemic_low_risk_field_crystallization_readiness,
            "systemic_low_risk_field_crystallization_score": systemic_low_risk_field_crystallization_score,
            "systemic_low_risk_field_crystallization_margin": systemic_low_risk_field_crystallization_margin,
        },
        "systemic_low_risk_field_crystallization_equation": {
            "strength_term": "A_sys_field_crystal = mix(S_sys_field_solid_score, R_sys_field_solid, 1 - P_sys_field_solid, M_brain_direct_v35, R_train_v41)",
            "structure_term": "S_sys_field_crystal = mix(S_sys_field_solid, R_sys_field_solid, D_structure_v35, B_struct_v41)",
            "route_term": "R_sys_field_crystal = mix(R_sys_field_solid, S_sys_field_solid, D_route_v35, H_sys_field_solid_v41)",
            "learning_term": "L_sys_field_crystal = mix(L_sys_field_solid, S_sys_field_solid_score, D_feature_v35, B_plastic_v41, R_train_v41)",
            "system_term": "M_sys_field_crystal = A_sys_field_crystal + S_sys_field_crystal + R_sys_field_crystal + L_sys_field_crystal - P_sys_field_solid",
        },
        "project_readout": {
            "summary": "systemic low-risk field crystallization validation checks whether the solidified low-risk field begins crystallizing into a more persistent field structure.",
            "next_question": "next verify whether this crystallized low-risk field still survives after it is folded into the next brain direct chain and training bridge.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Systemic Low-Risk Field Crystallization Validation Report",
        "",
        f"- systemic_low_risk_field_crystallization: {hm['systemic_low_risk_field_crystallization']:.6f}",
        f"- systemic_low_risk_field_structure_crystallization: {hm['systemic_low_risk_field_structure_crystallization']:.6f}",
        f"- systemic_low_risk_field_route_crystallization: {hm['systemic_low_risk_field_route_crystallization']:.6f}",
        f"- systemic_low_risk_field_learning_crystallization: {hm['systemic_low_risk_field_learning_crystallization']:.6f}",
        f"- systemic_low_risk_field_crystallization_penalty: {hm['systemic_low_risk_field_crystallization_penalty']:.6f}",
        f"- systemic_low_risk_field_crystallization_readiness: {hm['systemic_low_risk_field_crystallization_readiness']:.6f}",
        f"- systemic_low_risk_field_crystallization_score: {hm['systemic_low_risk_field_crystallization_score']:.6f}",
        f"- systemic_low_risk_field_crystallization_margin: {hm['systemic_low_risk_field_crystallization_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_systemic_low_risk_field_crystallization_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
