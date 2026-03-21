from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v44_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v44_summary() -> dict:
    v43 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v43_20260321" / "summary.json"
    )
    field_mesh = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_mesh_validation_20260321" / "summary.json"
    )
    brain_v38 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v38_20260321" / "summary.json"
    )

    hv = v43["headline_metrics"]
    hs = field_mesh["headline_metrics"]
    hb = brain_v38["headline_metrics"]

    plasticity_rule_alignment_v44 = _clip01(
        hv["plasticity_rule_alignment_v43"] * 0.28
        + hs["systemic_low_risk_field_learning_mesh"] * 0.24
        + (1.0 - hs["systemic_low_risk_field_mesh_penalty"]) * 0.14
        + hb["direct_feature_measure_v38"] * 0.14
        + (1.0 - hb["direct_brain_gap_v38"]) * 0.20
    )
    structure_rule_alignment_v44 = _clip01(
        hv["structure_rule_alignment_v43"] * 0.28
        + hs["systemic_low_risk_field_structure_mesh"] * 0.24
        + hs["systemic_low_risk_field_route_mesh"] * 0.14
        + (1.0 - hs["systemic_low_risk_field_mesh_penalty"]) * 0.10
        + hb["direct_structure_measure_v38"] * 0.24
    )
    topology_training_readiness_v44 = _clip01(
        hv["topology_training_readiness_v43"] * 0.30
        + plasticity_rule_alignment_v44 * 0.15
        + structure_rule_alignment_v44 * 0.15
        + hs["systemic_low_risk_field_mesh_readiness"] * 0.15
        + hb["direct_systemic_field_mesh_alignment_v38"] * 0.15
        + (1.0 - hs["systemic_low_risk_field_mesh_penalty"]) * 0.10
    )
    topology_training_gap_v44 = max(0.0, 1.0 - topology_training_readiness_v44)
    systemic_low_risk_field_mesh_guard_v44 = _clip01(
        (
            hs["systemic_low_risk_field_structure_mesh"]
            + hs["systemic_low_risk_field_route_mesh"]
            + hs["systemic_low_risk_field_mesh"]
            + topology_training_readiness_v44
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v44": plasticity_rule_alignment_v44,
            "structure_rule_alignment_v44": structure_rule_alignment_v44,
            "topology_training_readiness_v44": topology_training_readiness_v44,
            "topology_training_gap_v44": topology_training_gap_v44,
            "systemic_low_risk_field_mesh_guard_v44": systemic_low_risk_field_mesh_guard_v44,
        },
        "bridge_equation_v44": {
            "plasticity_term": "B_plastic_v44 = mix(B_plastic_v43, L_sys_field_mesh, 1 - P_sys_field_mesh, D_feature_v38, 1 - G_brain_v38)",
            "structure_term": "B_struct_v44 = mix(B_struct_v43, S_sys_field_mesh, R_sys_field_mesh, 1 - P_sys_field_mesh, D_structure_v38)",
            "readiness_term": "R_train_v44 = mix(R_train_v43, B_plastic_v44, B_struct_v44, R_sys_field_mesh, D_align_v38, 1 - P_sys_field_mesh)",
            "gap_term": "G_train_v44 = 1 - R_train_v44",
            "guard_term": "H_sys_field_mesh_v44 = mean(S_sys_field_mesh, R_sys_field_mesh, A_sys_field_mesh, R_train_v44)",
        },
        "project_readout": {
            "summary": "training bridge v44 checks whether systemic low-risk field mesh begins to reduce rule-layer risk in a more resilient mesh-like way.",
            "next_question": "next verify whether this mesh-like low-risk field still survives after it is folded into the next closed-form core.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Training Terminal Bridge V44 Report",
        "",
        f"- plasticity_rule_alignment_v44: {hm['plasticity_rule_alignment_v44']:.6f}",
        f"- structure_rule_alignment_v44: {hm['structure_rule_alignment_v44']:.6f}",
        f"- topology_training_readiness_v44: {hm['topology_training_readiness_v44']:.6f}",
        f"- topology_training_gap_v44: {hm['topology_training_gap_v44']:.6f}",
        f"- systemic_low_risk_field_mesh_guard_v44: {hm['systemic_low_risk_field_mesh_guard_v44']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v44_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
