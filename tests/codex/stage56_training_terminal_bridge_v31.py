from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v31_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v31_summary() -> dict:
    v30 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v30_20260321" / "summary.json"
    )
    low_risk = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_low_risk_steady_amplification_validation_20260321" / "summary.json"
    )
    brain_v25 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v25_20260321" / "summary.json"
    )

    hv = v30["headline_metrics"]
    hs = low_risk["headline_metrics"]
    hb = brain_v25["headline_metrics"]

    plasticity_rule_alignment_v31 = _clip01(
        hv["plasticity_rule_alignment_v30"] * 0.28
        + hs["low_risk_learning"] * 0.24
        + (1.0 - hs["low_risk_penalty"]) * 0.14
        + hb["direct_feature_measure_v25"] * 0.14
        + (1.0 - hb["direct_brain_gap_v25"]) * 0.20
    )
    structure_rule_alignment_v31 = _clip01(
        hv["structure_rule_alignment_v30"] * 0.28
        + hs["low_risk_structure"] * 0.24
        + hs["low_risk_route"] * 0.14
        + (1.0 - hs["low_risk_penalty"]) * 0.10
        + hb["direct_structure_measure_v25"] * 0.24
    )
    topology_training_readiness_v31 = _clip01(
        hv["topology_training_readiness_v30"] * 0.30
        + plasticity_rule_alignment_v31 * 0.15
        + structure_rule_alignment_v31 * 0.15
        + hs["low_risk_readiness"] * 0.15
        + hb["direct_low_risk_alignment_v25"] * 0.15
        + (1.0 - hs["low_risk_penalty"]) * 0.10
    )
    topology_training_gap_v31 = max(0.0, 1.0 - topology_training_readiness_v31)
    low_risk_guard_v31 = _clip01(
        (
            hs["low_risk_structure"]
            + hs["low_risk_route"]
            + hs["low_risk_strength"]
            + topology_training_readiness_v31
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v31": plasticity_rule_alignment_v31,
            "structure_rule_alignment_v31": structure_rule_alignment_v31,
            "topology_training_readiness_v31": topology_training_readiness_v31,
            "topology_training_gap_v31": topology_training_gap_v31,
            "low_risk_guard_v31": low_risk_guard_v31,
        },
        "bridge_equation_v31": {
            "plasticity_term": "B_plastic_v31 = mix(B_plastic_v30, L_low, 1 - P_low, D_feature_v25, 1 - G_brain_v25)",
            "structure_term": "B_struct_v31 = mix(B_struct_v30, S_low, R_low, 1 - P_low, D_structure_v25)",
            "readiness_term": "R_train_v31 = mix(R_train_v30, B_plastic_v31, B_struct_v31, R_low, D_align_v25, 1 - P_low)",
            "gap_term": "G_train_v31 = 1 - R_train_v31",
            "guard_term": "H_low_v31 = mean(S_low, R_low, A_low, R_train_v31)",
        },
        "project_readout": {
            "summary": "training bridge v31 folds the low-risk steady amplification layer into the rule system and checks whether rule risk keeps shrinking.",
            "next_question": "next verify whether the low-risk trend survives once it is absorbed into the next closed-form core.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Training Terminal Bridge V31 Report",
        "",
        f"- plasticity_rule_alignment_v31: {hm['plasticity_rule_alignment_v31']:.6f}",
        f"- structure_rule_alignment_v31: {hm['structure_rule_alignment_v31']:.6f}",
        f"- topology_training_readiness_v31: {hm['topology_training_readiness_v31']:.6f}",
        f"- topology_training_gap_v31: {hm['topology_training_gap_v31']:.6f}",
        f"- low_risk_guard_v31: {hm['low_risk_guard_v31']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v31_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
