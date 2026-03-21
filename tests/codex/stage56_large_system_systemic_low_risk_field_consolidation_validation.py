from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_consolidation_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_systemic_low_risk_field_consolidation_validation_summary() -> dict:
    stable = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_stabilization_validation_20260321" / "summary.json"
    )
    brain_v33 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v33_20260321" / "summary.json"
    )
    bridge_v39 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v39_20260321" / "summary.json"
    )
    v95 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v95_20260321" / "summary.json"
    )

    hs = stable["headline_metrics"]
    hb = brain_v33["headline_metrics"]
    ht = bridge_v39["headline_metrics"]
    hm = v95["headline_metrics"]

    systemic_low_risk_field_consolidation = _clip01(
        hs["systemic_low_risk_field_stability_score"] * 0.31
        + hs["systemic_low_risk_field_stability_readiness"] * 0.19
        + (1.0 - hs["systemic_low_risk_field_residual_penalty"]) * 0.16
        + hb["direct_brain_measure_v33"] * 0.16
        + ht["topology_training_readiness_v39"] * 0.18
    )
    systemic_low_risk_field_structure_consolidation = _clip01(
        hs["systemic_low_risk_field_structure_stability"] * 0.38
        + hs["systemic_low_risk_field_route_stability"] * 0.08
        + hb["direct_structure_measure_v33"] * 0.24
        + ht["structure_rule_alignment_v39"] * 0.30
    )
    systemic_low_risk_field_route_consolidation = _clip01(
        hs["systemic_low_risk_field_route_stability"] * 0.38
        + hs["systemic_low_risk_field_structure_stability"] * 0.08
        + hb["direct_route_measure_v33"] * 0.24
        + ht["systemic_low_risk_field_stability_guard_v39"] * 0.30
    )
    systemic_low_risk_field_learning_consolidation = _clip01(
        hs["systemic_low_risk_field_learning_stability"] * 0.28
        + hs["systemic_low_risk_field_stability_score"] * 0.12
        + hb["direct_feature_measure_v33"] * 0.18
        + ht["plasticity_rule_alignment_v39"] * 0.20
        + ht["topology_training_readiness_v39"] * 0.22
    )
    systemic_low_risk_field_consolidation_penalty = _clip01(
        hs["systemic_low_risk_field_residual_penalty"] * 0.35
        + hb["direct_brain_gap_v33"] * 0.21
        + ht["topology_training_gap_v39"] * 0.21
        + (1.0 - hs["systemic_low_risk_field_stability_score"]) * 0.05
        + hm["pressure_term_v95"] * 1e-3 * 0.18
    )
    systemic_low_risk_field_consolidation_readiness = _clip01(
        (
            systemic_low_risk_field_consolidation
            + systemic_low_risk_field_structure_consolidation
            + systemic_low_risk_field_route_consolidation
            + systemic_low_risk_field_learning_consolidation
            + (1.0 - systemic_low_risk_field_consolidation_penalty)
        )
        / 5.0
    )
    systemic_low_risk_field_consolidation_score = _clip01(
        (
            systemic_low_risk_field_consolidation_readiness
            + systemic_low_risk_field_consolidation
            + systemic_low_risk_field_route_consolidation
            + systemic_low_risk_field_learning_consolidation
            + (1.0 - systemic_low_risk_field_consolidation_penalty)
        )
        / 5.0
    )
    systemic_low_risk_field_consolidation_margin = (
        systemic_low_risk_field_consolidation
        + systemic_low_risk_field_structure_consolidation
        + systemic_low_risk_field_route_consolidation
        + systemic_low_risk_field_learning_consolidation
        + systemic_low_risk_field_consolidation_readiness
        + systemic_low_risk_field_consolidation_score
        - systemic_low_risk_field_consolidation_penalty
        + hm["encoding_margin_v95"] * 1e-15
    )

    return {
        "headline_metrics": {
            "systemic_low_risk_field_consolidation": systemic_low_risk_field_consolidation,
            "systemic_low_risk_field_structure_consolidation": systemic_low_risk_field_structure_consolidation,
            "systemic_low_risk_field_route_consolidation": systemic_low_risk_field_route_consolidation,
            "systemic_low_risk_field_learning_consolidation": systemic_low_risk_field_learning_consolidation,
            "systemic_low_risk_field_consolidation_penalty": systemic_low_risk_field_consolidation_penalty,
            "systemic_low_risk_field_consolidation_readiness": systemic_low_risk_field_consolidation_readiness,
            "systemic_low_risk_field_consolidation_score": systemic_low_risk_field_consolidation_score,
            "systemic_low_risk_field_consolidation_margin": systemic_low_risk_field_consolidation_margin,
        },
        "systemic_low_risk_field_consolidation_equation": {
            "strength_term": "A_sys_field_cons = mix(S_sys_field_stability_score, R_sys_field_stable, 1 - P_sys_field_stable, M_brain_direct_v33, R_train_v39)",
            "structure_term": "S_sys_field_cons = mix(S_sys_field_stable, R_sys_field_stable, D_structure_v33, B_struct_v39)",
            "route_term": "R_sys_field_cons = mix(R_sys_field_stable, S_sys_field_stable, D_route_v33, H_sys_field_stable_v39)",
            "learning_term": "L_sys_field_cons = mix(L_sys_field_stable, S_sys_field_stability_score, D_feature_v33, B_plastic_v39, R_train_v39)",
            "system_term": "M_sys_field_cons = A_sys_field_cons + S_sys_field_cons + R_sys_field_cons + L_sys_field_cons - P_sys_field_stable",
        },
        "project_readout": {
            "summary": "systemic low-risk field consolidation validation checks whether the stabilized low-risk field begins consolidating instead of only staying temporarily stable.",
            "next_question": "next verify whether this consolidated low-risk field still survives after it is folded into the next brain direct chain and training bridge.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Systemic Low-Risk Field Consolidation Validation Report",
        "",
        f"- systemic_low_risk_field_consolidation: {hm['systemic_low_risk_field_consolidation']:.6f}",
        f"- systemic_low_risk_field_structure_consolidation: {hm['systemic_low_risk_field_structure_consolidation']:.6f}",
        f"- systemic_low_risk_field_route_consolidation: {hm['systemic_low_risk_field_route_consolidation']:.6f}",
        f"- systemic_low_risk_field_learning_consolidation: {hm['systemic_low_risk_field_learning_consolidation']:.6f}",
        f"- systemic_low_risk_field_consolidation_penalty: {hm['systemic_low_risk_field_consolidation_penalty']:.6f}",
        f"- systemic_low_risk_field_consolidation_readiness: {hm['systemic_low_risk_field_consolidation_readiness']:.6f}",
        f"- systemic_low_risk_field_consolidation_score: {hm['systemic_low_risk_field_consolidation_score']:.6f}",
        f"- systemic_low_risk_field_consolidation_margin: {hm['systemic_low_risk_field_consolidation_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_systemic_low_risk_field_consolidation_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
