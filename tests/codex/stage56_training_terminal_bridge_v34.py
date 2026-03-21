from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v34_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v34_summary() -> dict:
    v33 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v33_20260321" / "summary.json"
    )
    systemic_expand = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_zone_expansion_validation_20260321" / "summary.json"
    )
    brain_v28 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v28_20260321" / "summary.json"
    )

    hv = v33["headline_metrics"]
    hs = systemic_expand["headline_metrics"]
    hb = brain_v28["headline_metrics"]

    plasticity_rule_alignment_v34 = _clip01(
        hv["plasticity_rule_alignment_v33"] * 0.28
        + hs["systemic_low_risk_expansion_learning"] * 0.24
        + (1.0 - hs["systemic_low_risk_expansion_penalty"]) * 0.14
        + hb["direct_feature_measure_v28"] * 0.14
        + (1.0 - hb["direct_brain_gap_v28"]) * 0.20
    )
    structure_rule_alignment_v34 = _clip01(
        hv["structure_rule_alignment_v33"] * 0.28
        + hs["systemic_low_risk_expansion_structure"] * 0.24
        + hs["systemic_low_risk_expansion_route"] * 0.14
        + (1.0 - hs["systemic_low_risk_expansion_penalty"]) * 0.10
        + hb["direct_structure_measure_v28"] * 0.24
    )
    topology_training_readiness_v34 = _clip01(
        hv["topology_training_readiness_v33"] * 0.30
        + plasticity_rule_alignment_v34 * 0.15
        + structure_rule_alignment_v34 * 0.15
        + hs["systemic_low_risk_expansion_readiness"] * 0.15
        + hb["direct_systemic_expansion_alignment_v28"] * 0.15
        + (1.0 - hs["systemic_low_risk_expansion_penalty"]) * 0.10
    )
    topology_training_gap_v34 = max(0.0, 1.0 - topology_training_readiness_v34)
    systemic_low_risk_expansion_guard_v34 = _clip01(
        (
            hs["systemic_low_risk_expansion_structure"]
            + hs["systemic_low_risk_expansion_route"]
            + hs["systemic_low_risk_expansion_strength"]
            + topology_training_readiness_v34
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v34": plasticity_rule_alignment_v34,
            "structure_rule_alignment_v34": structure_rule_alignment_v34,
            "topology_training_readiness_v34": topology_training_readiness_v34,
            "topology_training_gap_v34": topology_training_gap_v34,
            "systemic_low_risk_expansion_guard_v34": systemic_low_risk_expansion_guard_v34,
        },
        "bridge_equation_v34": {
            "plasticity_term": "B_plastic_v34 = mix(B_plastic_v33, L_sys_expand, 1 - P_sys_expand, D_feature_v28, 1 - G_brain_v28)",
            "structure_term": "B_struct_v34 = mix(B_struct_v33, S_sys_expand, R_sys_expand, 1 - P_sys_expand, D_structure_v28)",
            "readiness_term": "R_train_v34 = mix(R_train_v33, B_plastic_v34, B_struct_v34, R_sys_expand, D_align_v28, 1 - P_sys_expand)",
            "gap_term": "G_train_v34 = 1 - R_train_v34",
            "guard_term": "H_sys_expand_v34 = mean(S_sys_expand, R_sys_expand, A_sys_expand, R_train_v34)",
        },
        "project_readout": {
            "summary": "training bridge v34 checks whether systemic low-risk zone expansion begins to reduce rule-layer risk in a more explicit way.",
            "next_question": "next verify whether this systemic expansion still survives after it is folded into the next closed-form core.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Training Terminal Bridge V34 Report",
        "",
        f"- plasticity_rule_alignment_v34: {hm['plasticity_rule_alignment_v34']:.6f}",
        f"- structure_rule_alignment_v34: {hm['structure_rule_alignment_v34']:.6f}",
        f"- topology_training_readiness_v34: {hm['topology_training_readiness_v34']:.6f}",
        f"- topology_training_gap_v34: {hm['topology_training_gap_v34']:.6f}",
        f"- systemic_low_risk_expansion_guard_v34: {hm['systemic_low_risk_expansion_guard_v34']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v34_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
