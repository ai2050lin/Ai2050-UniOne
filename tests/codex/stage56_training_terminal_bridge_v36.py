from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v36_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v36_summary() -> dict:
    v35 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v35_20260321" / "summary.json"
    )
    systemic_broad = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_zone_broadening_validation_20260321" / "summary.json"
    )
    brain_v30 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v30_20260321" / "summary.json"
    )

    hv = v35["headline_metrics"]
    hs = systemic_broad["headline_metrics"]
    hb = brain_v30["headline_metrics"]

    plasticity_rule_alignment_v36 = _clip01(
        hv["plasticity_rule_alignment_v35"] * 0.28
        + hs["systemic_low_risk_broadening_learning"] * 0.24
        + (1.0 - hs["systemic_low_risk_broadening_penalty"]) * 0.14
        + hb["direct_feature_measure_v30"] * 0.14
        + (1.0 - hb["direct_brain_gap_v30"]) * 0.20
    )
    structure_rule_alignment_v36 = _clip01(
        hv["structure_rule_alignment_v35"] * 0.28
        + hs["systemic_low_risk_broadening_structure"] * 0.24
        + hs["systemic_low_risk_broadening_route"] * 0.14
        + (1.0 - hs["systemic_low_risk_broadening_penalty"]) * 0.10
        + hb["direct_structure_measure_v30"] * 0.24
    )
    topology_training_readiness_v36 = _clip01(
        hv["topology_training_readiness_v35"] * 0.30
        + plasticity_rule_alignment_v36 * 0.15
        + structure_rule_alignment_v36 * 0.15
        + hs["systemic_low_risk_broadening_readiness"] * 0.15
        + hb["direct_systemic_broadening_alignment_v30"] * 0.15
        + (1.0 - hs["systemic_low_risk_broadening_penalty"]) * 0.10
    )
    topology_training_gap_v36 = max(0.0, 1.0 - topology_training_readiness_v36)
    systemic_low_risk_broadening_guard_v36 = _clip01(
        (
            hs["systemic_low_risk_broadening_structure"]
            + hs["systemic_low_risk_broadening_route"]
            + hs["systemic_low_risk_broadening_strength"]
            + topology_training_readiness_v36
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v36": plasticity_rule_alignment_v36,
            "structure_rule_alignment_v36": structure_rule_alignment_v36,
            "topology_training_readiness_v36": topology_training_readiness_v36,
            "topology_training_gap_v36": topology_training_gap_v36,
            "systemic_low_risk_broadening_guard_v36": systemic_low_risk_broadening_guard_v36,
        },
        "bridge_equation_v36": {
            "plasticity_term": "B_plastic_v36 = mix(B_plastic_v35, L_sys_broad, 1 - P_sys_broad, D_feature_v30, 1 - G_brain_v30)",
            "structure_term": "B_struct_v36 = mix(B_struct_v35, S_sys_broad, R_sys_broad, 1 - P_sys_broad, D_structure_v30)",
            "readiness_term": "R_train_v36 = mix(R_train_v35, B_plastic_v36, B_struct_v36, R_sys_broad, D_align_v30, 1 - P_sys_broad)",
            "gap_term": "G_train_v36 = 1 - R_train_v36",
            "guard_term": "H_sys_broad_v36 = mean(S_sys_broad, R_sys_broad, A_sys_broad, R_train_v36)",
        },
        "project_readout": {
            "summary": "training bridge v36 checks whether systemic low-risk broadening begins to reduce rule-layer risk in a more consolidated and wider way.",
            "next_question": "next verify whether this broadening still survives after it is folded into the next closed-form core.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Training Terminal Bridge V36 Report",
        "",
        f"- plasticity_rule_alignment_v36: {hm['plasticity_rule_alignment_v36']:.6f}",
        f"- structure_rule_alignment_v36: {hm['structure_rule_alignment_v36']:.6f}",
        f"- topology_training_readiness_v36: {hm['topology_training_readiness_v36']:.6f}",
        f"- topology_training_gap_v36: {hm['topology_training_gap_v36']:.6f}",
        f"- systemic_low_risk_broadening_guard_v36: {hm['systemic_low_risk_broadening_guard_v36']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v36_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
