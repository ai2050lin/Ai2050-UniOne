from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v43_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v43_summary() -> dict:
    v42 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v42_20260321" / "summary.json"
    )
    field_lattice = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_lattice_validation_20260321" / "summary.json"
    )
    brain_v37 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v37_20260321" / "summary.json"
    )

    hv = v42["headline_metrics"]
    hs = field_lattice["headline_metrics"]
    hb = brain_v37["headline_metrics"]

    plasticity_rule_alignment_v43 = _clip01(
        hv["plasticity_rule_alignment_v42"] * 0.28
        + hs["systemic_low_risk_field_learning_lattice"] * 0.24
        + (1.0 - hs["systemic_low_risk_field_lattice_penalty"]) * 0.14
        + hb["direct_feature_measure_v37"] * 0.14
        + (1.0 - hb["direct_brain_gap_v37"]) * 0.20
    )
    structure_rule_alignment_v43 = _clip01(
        hv["structure_rule_alignment_v42"] * 0.28
        + hs["systemic_low_risk_field_structure_lattice"] * 0.24
        + hs["systemic_low_risk_field_route_lattice"] * 0.14
        + (1.0 - hs["systemic_low_risk_field_lattice_penalty"]) * 0.10
        + hb["direct_structure_measure_v37"] * 0.24
    )
    topology_training_readiness_v43 = _clip01(
        hv["topology_training_readiness_v42"] * 0.30
        + plasticity_rule_alignment_v43 * 0.15
        + structure_rule_alignment_v43 * 0.15
        + hs["systemic_low_risk_field_lattice_readiness"] * 0.15
        + hb["direct_systemic_field_lattice_alignment_v37"] * 0.15
        + (1.0 - hs["systemic_low_risk_field_lattice_penalty"]) * 0.10
    )
    topology_training_gap_v43 = max(0.0, 1.0 - topology_training_readiness_v43)
    systemic_low_risk_field_lattice_guard_v43 = _clip01(
        (
            hs["systemic_low_risk_field_structure_lattice"]
            + hs["systemic_low_risk_field_route_lattice"]
            + hs["systemic_low_risk_field_lattice"]
            + topology_training_readiness_v43
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v43": plasticity_rule_alignment_v43,
            "structure_rule_alignment_v43": structure_rule_alignment_v43,
            "topology_training_readiness_v43": topology_training_readiness_v43,
            "topology_training_gap_v43": topology_training_gap_v43,
            "systemic_low_risk_field_lattice_guard_v43": systemic_low_risk_field_lattice_guard_v43,
        },
        "bridge_equation_v43": {
            "plasticity_term": "B_plastic_v43 = mix(B_plastic_v42, L_sys_field_lattice, 1 - P_sys_field_lattice, D_feature_v37, 1 - G_brain_v37)",
            "structure_term": "B_struct_v43 = mix(B_struct_v42, S_sys_field_lattice, R_sys_field_lattice, 1 - P_sys_field_lattice, D_structure_v37)",
            "readiness_term": "R_train_v43 = mix(R_train_v42, B_plastic_v43, B_struct_v43, R_sys_field_lattice, D_align_v37, 1 - P_sys_field_lattice)",
            "gap_term": "G_train_v43 = 1 - R_train_v43",
            "guard_term": "H_sys_field_lattice_v43 = mean(S_sys_field_lattice, R_sys_field_lattice, A_sys_field_lattice, R_train_v43)",
        },
        "project_readout": {
            "summary": "training bridge v43 checks whether systemic low-risk field lattice begins to reduce rule-layer risk in a more regular field-like way.",
            "next_question": "next verify whether this lattice-like low-risk field still survives after it is folded into the next closed-form core.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Training Terminal Bridge V43 Report",
        "",
        f"- plasticity_rule_alignment_v43: {hm['plasticity_rule_alignment_v43']:.6f}",
        f"- structure_rule_alignment_v43: {hm['structure_rule_alignment_v43']:.6f}",
        f"- topology_training_readiness_v43: {hm['topology_training_readiness_v43']:.6f}",
        f"- topology_training_gap_v43: {hm['topology_training_gap_v43']:.6f}",
        f"- systemic_low_risk_field_lattice_guard_v43: {hm['systemic_low_risk_field_lattice_guard_v43']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v43_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
