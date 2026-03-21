from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_stabilization_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_systemic_low_risk_field_stabilization_validation_summary() -> dict:
    field = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_extension_validation_20260321" / "summary.json"
    )
    brain_v32 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v32_20260321" / "summary.json"
    )
    bridge_v38 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v38_20260321" / "summary.json"
    )
    v94 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v94_20260321" / "summary.json"
    )

    hs = field["headline_metrics"]
    hb = brain_v32["headline_metrics"]
    ht = bridge_v38["headline_metrics"]
    hm = v94["headline_metrics"]

    systemic_low_risk_field_stability = _clip01(
        hs["systemic_low_risk_field_score"] * 0.31
        + hs["systemic_low_risk_field_readiness"] * 0.19
        + (1.0 - hs["systemic_low_risk_field_penalty"]) * 0.16
        + hb["direct_brain_measure_v32"] * 0.16
        + ht["topology_training_readiness_v38"] * 0.18
    )
    systemic_low_risk_field_structure_stability = _clip01(
        hs["systemic_low_risk_field_structure"] * 0.38
        + hs["systemic_low_risk_field_route"] * 0.08
        + hb["direct_structure_measure_v32"] * 0.24
        + ht["structure_rule_alignment_v38"] * 0.30
    )
    systemic_low_risk_field_route_stability = _clip01(
        hs["systemic_low_risk_field_route"] * 0.38
        + hs["systemic_low_risk_field_structure"] * 0.08
        + hb["direct_route_measure_v32"] * 0.24
        + ht["systemic_low_risk_field_guard_v38"] * 0.30
    )
    systemic_low_risk_field_learning_stability = _clip01(
        hs["systemic_low_risk_field_learning"] * 0.28
        + hs["systemic_low_risk_field_score"] * 0.12
        + hb["direct_feature_measure_v32"] * 0.18
        + ht["plasticity_rule_alignment_v38"] * 0.20
        + ht["topology_training_readiness_v38"] * 0.22
    )
    systemic_low_risk_field_residual_penalty = _clip01(
        hs["systemic_low_risk_field_penalty"] * 0.35
        + hb["direct_brain_gap_v32"] * 0.21
        + ht["topology_training_gap_v38"] * 0.21
        + (1.0 - hs["systemic_low_risk_field_score"]) * 0.05
        + hm["pressure_term_v94"] * 1e-3 * 0.18
    )
    systemic_low_risk_field_stability_readiness = _clip01(
        (
            systemic_low_risk_field_stability
            + systemic_low_risk_field_structure_stability
            + systemic_low_risk_field_route_stability
            + systemic_low_risk_field_learning_stability
            + (1.0 - systemic_low_risk_field_residual_penalty)
        )
        / 5.0
    )
    systemic_low_risk_field_stability_score = _clip01(
        (
            systemic_low_risk_field_stability_readiness
            + systemic_low_risk_field_stability
            + systemic_low_risk_field_route_stability
            + systemic_low_risk_field_learning_stability
            + (1.0 - systemic_low_risk_field_residual_penalty)
        )
        / 5.0
    )
    systemic_low_risk_field_stability_margin = (
        systemic_low_risk_field_stability
        + systemic_low_risk_field_structure_stability
        + systemic_low_risk_field_route_stability
        + systemic_low_risk_field_learning_stability
        + systemic_low_risk_field_stability_readiness
        + systemic_low_risk_field_stability_score
        - systemic_low_risk_field_residual_penalty
        + hm["encoding_margin_v94"] * 1e-15
    )

    return {
        "headline_metrics": {
            "systemic_low_risk_field_stability": systemic_low_risk_field_stability,
            "systemic_low_risk_field_structure_stability": systemic_low_risk_field_structure_stability,
            "systemic_low_risk_field_route_stability": systemic_low_risk_field_route_stability,
            "systemic_low_risk_field_learning_stability": systemic_low_risk_field_learning_stability,
            "systemic_low_risk_field_residual_penalty": systemic_low_risk_field_residual_penalty,
            "systemic_low_risk_field_stability_readiness": systemic_low_risk_field_stability_readiness,
            "systemic_low_risk_field_stability_score": systemic_low_risk_field_stability_score,
            "systemic_low_risk_field_stability_margin": systemic_low_risk_field_stability_margin,
        },
        "systemic_low_risk_field_stability_equation": {
            "strength_term": "A_sys_field_stable = mix(S_sys_field_score, R_sys_field, 1 - P_sys_field, M_brain_direct_v32, R_train_v38)",
            "structure_term": "S_sys_field_stable = mix(S_sys_field, R_sys_field, D_structure_v32, B_struct_v38)",
            "route_term": "R_sys_field_stable = mix(R_sys_field, S_sys_field, D_route_v32, H_sys_field_v38)",
            "learning_term": "L_sys_field_stable = mix(L_sys_field, S_sys_field_score, D_feature_v32, B_plastic_v38, R_train_v38)",
            "system_term": "M_sys_field_stable = A_sys_field_stable + S_sys_field_stable + R_sys_field_stable + L_sys_field_stable - P_sys_field",
        },
        "project_readout": {
            "summary": "systemic low-risk field stabilization validation checks whether the connected low-risk field begins stabilizing instead of only continuing to extend.",
            "next_question": "next verify whether this stabilized low-risk field still survives after it is folded into the next brain direct chain and training bridge.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Systemic Low-Risk Field Stabilization Validation Report",
        "",
        f"- systemic_low_risk_field_stability: {hm['systemic_low_risk_field_stability']:.6f}",
        f"- systemic_low_risk_field_structure_stability: {hm['systemic_low_risk_field_structure_stability']:.6f}",
        f"- systemic_low_risk_field_route_stability: {hm['systemic_low_risk_field_route_stability']:.6f}",
        f"- systemic_low_risk_field_learning_stability: {hm['systemic_low_risk_field_learning_stability']:.6f}",
        f"- systemic_low_risk_field_residual_penalty: {hm['systemic_low_risk_field_residual_penalty']:.6f}",
        f"- systemic_low_risk_field_stability_readiness: {hm['systemic_low_risk_field_stability_readiness']:.6f}",
        f"- systemic_low_risk_field_stability_score: {hm['systemic_low_risk_field_stability_score']:.6f}",
        f"- systemic_low_risk_field_stability_margin: {hm['systemic_low_risk_field_stability_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_systemic_low_risk_field_stabilization_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
