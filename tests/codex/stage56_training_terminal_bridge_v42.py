from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v42_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v42_summary() -> dict:
    v41 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v41_20260321" / "summary.json"
    )
    field_crystal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_crystallization_validation_20260321" / "summary.json"
    )
    brain_v36 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v36_20260321" / "summary.json"
    )

    hv = v41["headline_metrics"]
    hs = field_crystal["headline_metrics"]
    hb = brain_v36["headline_metrics"]

    plasticity_rule_alignment_v42 = _clip01(
        hv["plasticity_rule_alignment_v41"] * 0.28
        + hs["systemic_low_risk_field_learning_crystallization"] * 0.24
        + (1.0 - hs["systemic_low_risk_field_crystallization_penalty"]) * 0.14
        + hb["direct_feature_measure_v36"] * 0.14
        + (1.0 - hb["direct_brain_gap_v36"]) * 0.20
    )
    structure_rule_alignment_v42 = _clip01(
        hv["structure_rule_alignment_v41"] * 0.28
        + hs["systemic_low_risk_field_structure_crystallization"] * 0.24
        + hs["systemic_low_risk_field_route_crystallization"] * 0.14
        + (1.0 - hs["systemic_low_risk_field_crystallization_penalty"]) * 0.10
        + hb["direct_structure_measure_v36"] * 0.24
    )
    topology_training_readiness_v42 = _clip01(
        hv["topology_training_readiness_v41"] * 0.30
        + plasticity_rule_alignment_v42 * 0.15
        + structure_rule_alignment_v42 * 0.15
        + hs["systemic_low_risk_field_crystallization_readiness"] * 0.15
        + hb["direct_systemic_field_crystallization_alignment_v36"] * 0.15
        + (1.0 - hs["systemic_low_risk_field_crystallization_penalty"]) * 0.10
    )
    topology_training_gap_v42 = max(0.0, 1.0 - topology_training_readiness_v42)
    systemic_low_risk_field_crystallization_guard_v42 = _clip01(
        (
            hs["systemic_low_risk_field_structure_crystallization"]
            + hs["systemic_low_risk_field_route_crystallization"]
            + hs["systemic_low_risk_field_crystallization"]
            + topology_training_readiness_v42
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v42": plasticity_rule_alignment_v42,
            "structure_rule_alignment_v42": structure_rule_alignment_v42,
            "topology_training_readiness_v42": topology_training_readiness_v42,
            "topology_training_gap_v42": topology_training_gap_v42,
            "systemic_low_risk_field_crystallization_guard_v42": systemic_low_risk_field_crystallization_guard_v42,
        },
        "bridge_equation_v42": {
            "plasticity_term": "B_plastic_v42 = mix(B_plastic_v41, L_sys_field_crystal, 1 - P_sys_field_crystal, D_feature_v36, 1 - G_brain_v36)",
            "structure_term": "B_struct_v42 = mix(B_struct_v41, S_sys_field_crystal, R_sys_field_crystal, 1 - P_sys_field_crystal, D_structure_v36)",
            "readiness_term": "R_train_v42 = mix(R_train_v41, B_plastic_v42, B_struct_v42, R_sys_field_crystal, D_align_v36, 1 - P_sys_field_crystal)",
            "gap_term": "G_train_v42 = 1 - R_train_v42",
            "guard_term": "H_sys_field_crystal_v42 = mean(S_sys_field_crystal, R_sys_field_crystal, A_sys_field_crystal, R_train_v42)",
        },
        "project_readout": {
            "summary": "training bridge v42 checks whether systemic low-risk field crystallization begins to reduce rule-layer risk in a more persistent field-like way.",
            "next_question": "next verify whether this crystallized low-risk field still survives after it is folded into the next closed-form core.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Training Terminal Bridge V42 Report",
        "",
        f"- plasticity_rule_alignment_v42: {hm['plasticity_rule_alignment_v42']:.6f}",
        f"- structure_rule_alignment_v42: {hm['structure_rule_alignment_v42']:.6f}",
        f"- topology_training_readiness_v42: {hm['topology_training_readiness_v42']:.6f}",
        f"- topology_training_gap_v42: {hm['topology_training_gap_v42']:.6f}",
        f"- systemic_low_risk_field_crystallization_guard_v42: {hm['systemic_low_risk_field_crystallization_guard_v42']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v42_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
