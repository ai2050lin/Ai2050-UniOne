from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v40_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v40_summary() -> dict:
    v39 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v39_20260321" / "summary.json"
    )
    systemic_field_cons = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_consolidation_validation_20260321" / "summary.json"
    )
    brain_v34 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v34_20260321" / "summary.json"
    )

    hv = v39["headline_metrics"]
    hs = systemic_field_cons["headline_metrics"]
    hb = brain_v34["headline_metrics"]

    plasticity_rule_alignment_v40 = _clip01(
        hv["plasticity_rule_alignment_v39"] * 0.28
        + hs["systemic_low_risk_field_learning_consolidation"] * 0.24
        + (1.0 - hs["systemic_low_risk_field_consolidation_penalty"]) * 0.14
        + hb["direct_feature_measure_v34"] * 0.14
        + (1.0 - hb["direct_brain_gap_v34"]) * 0.20
    )
    structure_rule_alignment_v40 = _clip01(
        hv["structure_rule_alignment_v39"] * 0.28
        + hs["systemic_low_risk_field_structure_consolidation"] * 0.24
        + hs["systemic_low_risk_field_route_consolidation"] * 0.14
        + (1.0 - hs["systemic_low_risk_field_consolidation_penalty"]) * 0.10
        + hb["direct_structure_measure_v34"] * 0.24
    )
    topology_training_readiness_v40 = _clip01(
        hv["topology_training_readiness_v39"] * 0.30
        + plasticity_rule_alignment_v40 * 0.15
        + structure_rule_alignment_v40 * 0.15
        + hs["systemic_low_risk_field_consolidation_readiness"] * 0.15
        + hb["direct_systemic_field_consolidation_alignment_v34"] * 0.15
        + (1.0 - hs["systemic_low_risk_field_consolidation_penalty"]) * 0.10
    )
    topology_training_gap_v40 = max(0.0, 1.0 - topology_training_readiness_v40)
    systemic_low_risk_field_consolidation_guard_v40 = _clip01(
        (
            hs["systemic_low_risk_field_structure_consolidation"]
            + hs["systemic_low_risk_field_route_consolidation"]
            + hs["systemic_low_risk_field_consolidation"]
            + topology_training_readiness_v40
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v40": plasticity_rule_alignment_v40,
            "structure_rule_alignment_v40": structure_rule_alignment_v40,
            "topology_training_readiness_v40": topology_training_readiness_v40,
            "topology_training_gap_v40": topology_training_gap_v40,
            "systemic_low_risk_field_consolidation_guard_v40": systemic_low_risk_field_consolidation_guard_v40,
        },
        "bridge_equation_v40": {
            "plasticity_term": "B_plastic_v40 = mix(B_plastic_v39, L_sys_field_cons, 1 - P_sys_field_cons, D_feature_v34, 1 - G_brain_v34)",
            "structure_term": "B_struct_v40 = mix(B_struct_v39, S_sys_field_cons, R_sys_field_cons, 1 - P_sys_field_cons, D_structure_v34)",
            "readiness_term": "R_train_v40 = mix(R_train_v39, B_plastic_v40, B_struct_v40, R_sys_field_cons, D_align_v34, 1 - P_sys_field_cons)",
            "gap_term": "G_train_v40 = 1 - R_train_v40",
            "guard_term": "H_sys_field_cons_v40 = mean(S_sys_field_cons, R_sys_field_cons, A_sys_field_cons, R_train_v40)",
        },
        "project_readout": {
            "summary": "training bridge v40 checks whether systemic low-risk field consolidation begins to reduce rule-layer risk in a more consolidated and field-like way.",
            "next_question": "next verify whether this consolidated low-risk field still survives after it is folded into the next closed-form core.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Training Terminal Bridge V40 Report",
        "",
        f"- plasticity_rule_alignment_v40: {hm['plasticity_rule_alignment_v40']:.6f}",
        f"- structure_rule_alignment_v40: {hm['structure_rule_alignment_v40']:.6f}",
        f"- topology_training_readiness_v40: {hm['topology_training_readiness_v40']:.6f}",
        f"- topology_training_gap_v40: {hm['topology_training_gap_v40']:.6f}",
        f"- systemic_low_risk_field_consolidation_guard_v40: {hm['systemic_low_risk_field_consolidation_guard_v40']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v40_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
