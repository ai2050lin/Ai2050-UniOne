from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v41_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v41_summary() -> dict:
    v40 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v40_20260321" / "summary.json"
    )
    systemic_field_solid = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_solidification_validation_20260321" / "summary.json"
    )
    brain_v35 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v35_20260321" / "summary.json"
    )

    hv = v40["headline_metrics"]
    hs = systemic_field_solid["headline_metrics"]
    hb = brain_v35["headline_metrics"]

    plasticity_rule_alignment_v41 = _clip01(
        hv["plasticity_rule_alignment_v40"] * 0.28
        + hs["systemic_low_risk_field_learning_solidification"] * 0.24
        + (1.0 - hs["systemic_low_risk_field_solidification_penalty"]) * 0.14
        + hb["direct_feature_measure_v35"] * 0.14
        + (1.0 - hb["direct_brain_gap_v35"]) * 0.20
    )
    structure_rule_alignment_v41 = _clip01(
        hv["structure_rule_alignment_v40"] * 0.28
        + hs["systemic_low_risk_field_structure_solidification"] * 0.24
        + hs["systemic_low_risk_field_route_solidification"] * 0.14
        + (1.0 - hs["systemic_low_risk_field_solidification_penalty"]) * 0.10
        + hb["direct_structure_measure_v35"] * 0.24
    )
    topology_training_readiness_v41 = _clip01(
        hv["topology_training_readiness_v40"] * 0.30
        + plasticity_rule_alignment_v41 * 0.15
        + structure_rule_alignment_v41 * 0.15
        + hs["systemic_low_risk_field_solidification_readiness"] * 0.15
        + hb["direct_systemic_field_solidification_alignment_v35"] * 0.15
        + (1.0 - hs["systemic_low_risk_field_solidification_penalty"]) * 0.10
    )
    topology_training_gap_v41 = max(0.0, 1.0 - topology_training_readiness_v41)
    systemic_low_risk_field_solidification_guard_v41 = _clip01(
        (
            hs["systemic_low_risk_field_structure_solidification"]
            + hs["systemic_low_risk_field_route_solidification"]
            + hs["systemic_low_risk_field_solidification"]
            + topology_training_readiness_v41
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v41": plasticity_rule_alignment_v41,
            "structure_rule_alignment_v41": structure_rule_alignment_v41,
            "topology_training_readiness_v41": topology_training_readiness_v41,
            "topology_training_gap_v41": topology_training_gap_v41,
            "systemic_low_risk_field_solidification_guard_v41": systemic_low_risk_field_solidification_guard_v41,
        },
        "bridge_equation_v41": {
            "plasticity_term": "B_plastic_v41 = mix(B_plastic_v40, L_sys_field_solid, 1 - P_sys_field_solid, D_feature_v35, 1 - G_brain_v35)",
            "structure_term": "B_struct_v41 = mix(B_struct_v40, S_sys_field_solid, R_sys_field_solid, 1 - P_sys_field_solid, D_structure_v35)",
            "readiness_term": "R_train_v41 = mix(R_train_v40, B_plastic_v41, B_struct_v41, R_sys_field_solid, D_align_v35, 1 - P_sys_field_solid)",
            "gap_term": "G_train_v41 = 1 - R_train_v41",
            "guard_term": "H_sys_field_solid_v41 = mean(S_sys_field_solid, R_sys_field_solid, A_sys_field_solid, R_train_v41)",
        },
        "project_readout": {
            "summary": "training bridge v41 checks whether systemic low-risk field solidification begins to reduce rule-layer risk in a more solidified and field-like way.",
            "next_question": "next verify whether this solidified low-risk field still survives after it is folded into the next closed-form core.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Training Terminal Bridge V41 Report",
        "",
        f"- plasticity_rule_alignment_v41: {hm['plasticity_rule_alignment_v41']:.6f}",
        f"- structure_rule_alignment_v41: {hm['structure_rule_alignment_v41']:.6f}",
        f"- topology_training_readiness_v41: {hm['topology_training_readiness_v41']:.6f}",
        f"- topology_training_gap_v41: {hm['topology_training_gap_v41']:.6f}",
        f"- systemic_low_risk_field_solidification_guard_v41: {hm['systemic_low_risk_field_solidification_guard_v41']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v41_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
