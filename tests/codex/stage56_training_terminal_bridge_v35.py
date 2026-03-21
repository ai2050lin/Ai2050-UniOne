from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v35_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v35_summary() -> dict:
    v34 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v34_20260321" / "summary.json"
    )
    systemic_enlarge = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_zone_enlargement_validation_20260321" / "summary.json"
    )
    brain_v29 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v29_20260321" / "summary.json"
    )

    hv = v34["headline_metrics"]
    hs = systemic_enlarge["headline_metrics"]
    hb = brain_v29["headline_metrics"]

    plasticity_rule_alignment_v35 = _clip01(
        hv["plasticity_rule_alignment_v34"] * 0.28
        + hs["systemic_low_risk_enlargement_learning"] * 0.24
        + (1.0 - hs["systemic_low_risk_enlargement_penalty"]) * 0.14
        + hb["direct_feature_measure_v29"] * 0.14
        + (1.0 - hb["direct_brain_gap_v29"]) * 0.20
    )
    structure_rule_alignment_v35 = _clip01(
        hv["structure_rule_alignment_v34"] * 0.28
        + hs["systemic_low_risk_enlargement_structure"] * 0.24
        + hs["systemic_low_risk_enlargement_route"] * 0.14
        + (1.0 - hs["systemic_low_risk_enlargement_penalty"]) * 0.10
        + hb["direct_structure_measure_v29"] * 0.24
    )
    topology_training_readiness_v35 = _clip01(
        hv["topology_training_readiness_v34"] * 0.30
        + plasticity_rule_alignment_v35 * 0.15
        + structure_rule_alignment_v35 * 0.15
        + hs["systemic_low_risk_enlargement_readiness"] * 0.15
        + hb["direct_systemic_enlargement_alignment_v29"] * 0.15
        + (1.0 - hs["systemic_low_risk_enlargement_penalty"]) * 0.10
    )
    topology_training_gap_v35 = max(0.0, 1.0 - topology_training_readiness_v35)
    systemic_low_risk_enlargement_guard_v35 = _clip01(
        (
            hs["systemic_low_risk_enlargement_structure"]
            + hs["systemic_low_risk_enlargement_route"]
            + hs["systemic_low_risk_enlargement_strength"]
            + topology_training_readiness_v35
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v35": plasticity_rule_alignment_v35,
            "structure_rule_alignment_v35": structure_rule_alignment_v35,
            "topology_training_readiness_v35": topology_training_readiness_v35,
            "topology_training_gap_v35": topology_training_gap_v35,
            "systemic_low_risk_enlargement_guard_v35": systemic_low_risk_enlargement_guard_v35,
        },
        "bridge_equation_v35": {
            "plasticity_term": "B_plastic_v35 = mix(B_plastic_v34, L_sys_enlarge, 1 - P_sys_enlarge, D_feature_v29, 1 - G_brain_v29)",
            "structure_term": "B_struct_v35 = mix(B_struct_v34, S_sys_enlarge, R_sys_enlarge, 1 - P_sys_enlarge, D_structure_v29)",
            "readiness_term": "R_train_v35 = mix(R_train_v34, B_plastic_v35, B_struct_v35, R_sys_enlarge, D_align_v29, 1 - P_sys_enlarge)",
            "gap_term": "G_train_v35 = 1 - R_train_v35",
            "guard_term": "H_sys_enlarge_v35 = mean(S_sys_enlarge, R_sys_enlarge, A_sys_enlarge, R_train_v35)",
        },
        "project_readout": {
            "summary": "training bridge v35 checks whether systemic low-risk enlargement begins to reduce rule-layer risk in a more consolidated way.",
            "next_question": "next verify whether this systemic enlargement still survives after it is folded into the next closed-form core.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Training Terminal Bridge V35 Report",
        "",
        f"- plasticity_rule_alignment_v35: {hm['plasticity_rule_alignment_v35']:.6f}",
        f"- structure_rule_alignment_v35: {hm['structure_rule_alignment_v35']:.6f}",
        f"- topology_training_readiness_v35: {hm['topology_training_readiness_v35']:.6f}",
        f"- topology_training_gap_v35: {hm['topology_training_gap_v35']:.6f}",
        f"- systemic_low_risk_enlargement_guard_v35: {hm['systemic_low_risk_enlargement_guard_v35']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v35_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
