from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v38_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v38_summary() -> dict:
    v37 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v37_20260321" / "summary.json"
    )
    systemic_field = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_extension_validation_20260321" / "summary.json"
    )
    brain_v32 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v32_20260321" / "summary.json"
    )

    hv = v37["headline_metrics"]
    hs = systemic_field["headline_metrics"]
    hb = brain_v32["headline_metrics"]

    plasticity_rule_alignment_v38 = _clip01(
        hv["plasticity_rule_alignment_v37"] * 0.28
        + hs["systemic_low_risk_field_learning"] * 0.24
        + (1.0 - hs["systemic_low_risk_field_penalty"]) * 0.14
        + hb["direct_feature_measure_v32"] * 0.14
        + (1.0 - hb["direct_brain_gap_v32"]) * 0.20
    )
    structure_rule_alignment_v38 = _clip01(
        hv["structure_rule_alignment_v37"] * 0.28
        + hs["systemic_low_risk_field_structure"] * 0.24
        + hs["systemic_low_risk_field_route"] * 0.14
        + (1.0 - hs["systemic_low_risk_field_penalty"]) * 0.10
        + hb["direct_structure_measure_v32"] * 0.24
    )
    topology_training_readiness_v38 = _clip01(
        hv["topology_training_readiness_v37"] * 0.30
        + plasticity_rule_alignment_v38 * 0.15
        + structure_rule_alignment_v38 * 0.15
        + hs["systemic_low_risk_field_readiness"] * 0.15
        + hb["direct_systemic_field_alignment_v32"] * 0.15
        + (1.0 - hs["systemic_low_risk_field_penalty"]) * 0.10
    )
    topology_training_gap_v38 = max(0.0, 1.0 - topology_training_readiness_v38)
    systemic_low_risk_field_guard_v38 = _clip01(
        (
            hs["systemic_low_risk_field_structure"]
            + hs["systemic_low_risk_field_route"]
            + hs["systemic_low_risk_field_strength"]
            + topology_training_readiness_v38
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v38": plasticity_rule_alignment_v38,
            "structure_rule_alignment_v38": structure_rule_alignment_v38,
            "topology_training_readiness_v38": topology_training_readiness_v38,
            "topology_training_gap_v38": topology_training_gap_v38,
            "systemic_low_risk_field_guard_v38": systemic_low_risk_field_guard_v38,
        },
        "bridge_equation_v38": {
            "plasticity_term": "B_plastic_v38 = mix(B_plastic_v37, L_sys_field, 1 - P_sys_field, D_feature_v32, 1 - G_brain_v32)",
            "structure_term": "B_struct_v38 = mix(B_struct_v37, S_sys_field, R_sys_field, 1 - P_sys_field, D_structure_v32)",
            "readiness_term": "R_train_v38 = mix(R_train_v37, B_plastic_v38, B_struct_v38, R_sys_field, D_align_v32, 1 - P_sys_field)",
            "gap_term": "G_train_v38 = 1 - R_train_v38",
            "guard_term": "H_sys_field_v38 = mean(S_sys_field, R_sys_field, A_sys_field, R_train_v38)",
        },
        "project_readout": {
            "summary": "training bridge v38 checks whether systemic low-risk field extension begins to reduce rule-layer risk in a more connected and field-like way.",
            "next_question": "next verify whether this low-risk field still survives after it is folded into the next closed-form core.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Training Terminal Bridge V38 Report",
        "",
        f"- plasticity_rule_alignment_v38: {hm['plasticity_rule_alignment_v38']:.6f}",
        f"- structure_rule_alignment_v38: {hm['structure_rule_alignment_v38']:.6f}",
        f"- topology_training_readiness_v38: {hm['topology_training_readiness_v38']:.6f}",
        f"- topology_training_gap_v38: {hm['topology_training_gap_v38']:.6f}",
        f"- systemic_low_risk_field_guard_v38: {hm['systemic_low_risk_field_guard_v38']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v38_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
