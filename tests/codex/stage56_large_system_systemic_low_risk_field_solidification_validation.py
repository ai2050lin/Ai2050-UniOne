from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_solidification_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_systemic_low_risk_field_solidification_validation_summary() -> dict:
    cons = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_consolidation_validation_20260321" / "summary.json"
    )
    brain_v34 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v34_20260321" / "summary.json"
    )
    bridge_v40 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v40_20260321" / "summary.json"
    )
    v96 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v96_20260321" / "summary.json"
    )

    hs = cons["headline_metrics"]
    hb = brain_v34["headline_metrics"]
    ht = bridge_v40["headline_metrics"]
    hm = v96["headline_metrics"]

    systemic_low_risk_field_solidification = _clip01(
        hs["systemic_low_risk_field_consolidation_score"] * 0.31
        + hs["systemic_low_risk_field_consolidation_readiness"] * 0.19
        + (1.0 - hs["systemic_low_risk_field_consolidation_penalty"]) * 0.16
        + hb["direct_brain_measure_v34"] * 0.16
        + ht["topology_training_readiness_v40"] * 0.18
    )
    systemic_low_risk_field_structure_solidification = _clip01(
        hs["systemic_low_risk_field_structure_consolidation"] * 0.38
        + hs["systemic_low_risk_field_route_consolidation"] * 0.08
        + hb["direct_structure_measure_v34"] * 0.24
        + ht["structure_rule_alignment_v40"] * 0.30
    )
    systemic_low_risk_field_route_solidification = _clip01(
        hs["systemic_low_risk_field_route_consolidation"] * 0.38
        + hs["systemic_low_risk_field_structure_consolidation"] * 0.08
        + hb["direct_route_measure_v34"] * 0.24
        + ht["systemic_low_risk_field_consolidation_guard_v40"] * 0.30
    )
    systemic_low_risk_field_learning_solidification = _clip01(
        hs["systemic_low_risk_field_learning_consolidation"] * 0.28
        + hs["systemic_low_risk_field_consolidation_score"] * 0.12
        + hb["direct_feature_measure_v34"] * 0.18
        + ht["plasticity_rule_alignment_v40"] * 0.20
        + ht["topology_training_readiness_v40"] * 0.22
    )
    systemic_low_risk_field_solidification_penalty = _clip01(
        hs["systemic_low_risk_field_consolidation_penalty"] * 0.35
        + hb["direct_brain_gap_v34"] * 0.21
        + ht["topology_training_gap_v40"] * 0.21
        + (1.0 - hs["systemic_low_risk_field_consolidation_score"]) * 0.05
        + hm["pressure_term_v96"] * 1e-3 * 0.18
    )
    systemic_low_risk_field_solidification_readiness = _clip01(
        (
            systemic_low_risk_field_solidification
            + systemic_low_risk_field_structure_solidification
            + systemic_low_risk_field_route_solidification
            + systemic_low_risk_field_learning_solidification
            + (1.0 - systemic_low_risk_field_solidification_penalty)
        )
        / 5.0
    )
    systemic_low_risk_field_solidification_score = _clip01(
        (
            systemic_low_risk_field_solidification_readiness
            + systemic_low_risk_field_solidification
            + systemic_low_risk_field_route_solidification
            + systemic_low_risk_field_learning_solidification
            + (1.0 - systemic_low_risk_field_solidification_penalty)
        )
        / 5.0
    )
    systemic_low_risk_field_solidification_margin = (
        systemic_low_risk_field_solidification
        + systemic_low_risk_field_structure_solidification
        + systemic_low_risk_field_route_solidification
        + systemic_low_risk_field_learning_solidification
        + systemic_low_risk_field_solidification_readiness
        + systemic_low_risk_field_solidification_score
        - systemic_low_risk_field_solidification_penalty
        + hm["encoding_margin_v96"] * 1e-15
    )

    return {
        "headline_metrics": {
            "systemic_low_risk_field_solidification": systemic_low_risk_field_solidification,
            "systemic_low_risk_field_structure_solidification": systemic_low_risk_field_structure_solidification,
            "systemic_low_risk_field_route_solidification": systemic_low_risk_field_route_solidification,
            "systemic_low_risk_field_learning_solidification": systemic_low_risk_field_learning_solidification,
            "systemic_low_risk_field_solidification_penalty": systemic_low_risk_field_solidification_penalty,
            "systemic_low_risk_field_solidification_readiness": systemic_low_risk_field_solidification_readiness,
            "systemic_low_risk_field_solidification_score": systemic_low_risk_field_solidification_score,
            "systemic_low_risk_field_solidification_margin": systemic_low_risk_field_solidification_margin,
        },
        "systemic_low_risk_field_solidification_equation": {
            "strength_term": "A_sys_field_solid = mix(S_sys_field_cons_score, R_sys_field_cons, 1 - P_sys_field_cons, M_brain_direct_v34, R_train_v40)",
            "structure_term": "S_sys_field_solid = mix(S_sys_field_cons, R_sys_field_cons, D_structure_v34, B_struct_v40)",
            "route_term": "R_sys_field_solid = mix(R_sys_field_cons, S_sys_field_cons, D_route_v34, H_sys_field_cons_v40)",
            "learning_term": "L_sys_field_solid = mix(L_sys_field_cons, S_sys_field_cons_score, D_feature_v34, B_plastic_v40, R_train_v40)",
            "system_term": "M_sys_field_solid = A_sys_field_solid + S_sys_field_solid + R_sys_field_solid + L_sys_field_solid - P_sys_field_cons",
        },
        "project_readout": {
            "summary": "systemic low-risk field solidification validation checks whether the consolidated low-risk field begins solidifying instead of only staying reinforced by short-horizon support.",
            "next_question": "next verify whether this solidified low-risk field still survives after it is folded into the next brain direct chain and training bridge.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Systemic Low-Risk Field Solidification Validation Report",
        "",
        f"- systemic_low_risk_field_solidification: {hm['systemic_low_risk_field_solidification']:.6f}",
        f"- systemic_low_risk_field_structure_solidification: {hm['systemic_low_risk_field_structure_solidification']:.6f}",
        f"- systemic_low_risk_field_route_solidification: {hm['systemic_low_risk_field_route_solidification']:.6f}",
        f"- systemic_low_risk_field_learning_solidification: {hm['systemic_low_risk_field_learning_solidification']:.6f}",
        f"- systemic_low_risk_field_solidification_penalty: {hm['systemic_low_risk_field_solidification_penalty']:.6f}",
        f"- systemic_low_risk_field_solidification_readiness: {hm['systemic_low_risk_field_solidification_readiness']:.6f}",
        f"- systemic_low_risk_field_solidification_score: {hm['systemic_low_risk_field_solidification_score']:.6f}",
        f"- systemic_low_risk_field_solidification_margin: {hm['systemic_low_risk_field_solidification_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_systemic_low_risk_field_solidification_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
