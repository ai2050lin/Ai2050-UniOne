from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v39_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v39_summary() -> dict:
    v38 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v38_20260321" / "summary.json"
    )
    systemic_field_stable = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_stabilization_validation_20260321" / "summary.json"
    )
    brain_v33 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v33_20260321" / "summary.json"
    )

    hv = v38["headline_metrics"]
    hs = systemic_field_stable["headline_metrics"]
    hb = brain_v33["headline_metrics"]

    plasticity_rule_alignment_v39 = _clip01(
        hv["plasticity_rule_alignment_v38"] * 0.28
        + hs["systemic_low_risk_field_learning_stability"] * 0.24
        + (1.0 - hs["systemic_low_risk_field_residual_penalty"]) * 0.14
        + hb["direct_feature_measure_v33"] * 0.14
        + (1.0 - hb["direct_brain_gap_v33"]) * 0.20
    )
    structure_rule_alignment_v39 = _clip01(
        hv["structure_rule_alignment_v38"] * 0.28
        + hs["systemic_low_risk_field_structure_stability"] * 0.24
        + hs["systemic_low_risk_field_route_stability"] * 0.14
        + (1.0 - hs["systemic_low_risk_field_residual_penalty"]) * 0.10
        + hb["direct_structure_measure_v33"] * 0.24
    )
    topology_training_readiness_v39 = _clip01(
        hv["topology_training_readiness_v38"] * 0.30
        + plasticity_rule_alignment_v39 * 0.15
        + structure_rule_alignment_v39 * 0.15
        + hs["systemic_low_risk_field_stability_readiness"] * 0.15
        + hb["direct_systemic_field_stability_alignment_v33"] * 0.15
        + (1.0 - hs["systemic_low_risk_field_residual_penalty"]) * 0.10
    )
    topology_training_gap_v39 = max(0.0, 1.0 - topology_training_readiness_v39)
    systemic_low_risk_field_stability_guard_v39 = _clip01(
        (
            hs["systemic_low_risk_field_structure_stability"]
            + hs["systemic_low_risk_field_route_stability"]
            + hs["systemic_low_risk_field_stability"]
            + topology_training_readiness_v39
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v39": plasticity_rule_alignment_v39,
            "structure_rule_alignment_v39": structure_rule_alignment_v39,
            "topology_training_readiness_v39": topology_training_readiness_v39,
            "topology_training_gap_v39": topology_training_gap_v39,
            "systemic_low_risk_field_stability_guard_v39": systemic_low_risk_field_stability_guard_v39,
        },
        "bridge_equation_v39": {
            "plasticity_term": "B_plastic_v39 = mix(B_plastic_v38, L_sys_field_stable, 1 - P_sys_field_stable, D_feature_v33, 1 - G_brain_v33)",
            "structure_term": "B_struct_v39 = mix(B_struct_v38, S_sys_field_stable, R_sys_field_stable, 1 - P_sys_field_stable, D_structure_v33)",
            "readiness_term": "R_train_v39 = mix(R_train_v38, B_plastic_v39, B_struct_v39, R_sys_field_stable, D_align_v33, 1 - P_sys_field_stable)",
            "gap_term": "G_train_v39 = 1 - R_train_v39",
            "guard_term": "H_sys_field_stable_v39 = mean(S_sys_field_stable, R_sys_field_stable, A_sys_field_stable, R_train_v39)",
        },
        "project_readout": {
            "summary": "training bridge v39 checks whether systemic low-risk field stabilization begins to reduce rule-layer risk in a more stable and field-like way.",
            "next_question": "next verify whether this stabilized low-risk field still survives after it is folded into the next closed-form core.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Training Terminal Bridge V39 Report",
        "",
        f"- plasticity_rule_alignment_v39: {hm['plasticity_rule_alignment_v39']:.6f}",
        f"- structure_rule_alignment_v39: {hm['structure_rule_alignment_v39']:.6f}",
        f"- topology_training_readiness_v39: {hm['topology_training_readiness_v39']:.6f}",
        f"- topology_training_gap_v39: {hm['topology_training_gap_v39']:.6f}",
        f"- systemic_low_risk_field_stability_guard_v39: {hm['systemic_low_risk_field_stability_guard_v39']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v39_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
