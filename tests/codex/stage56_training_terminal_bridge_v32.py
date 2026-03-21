from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v32_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v32_summary() -> dict:
    v31 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v31_20260321" / "summary.json"
    )
    low_risk_zone = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_low_risk_steady_zone_validation_20260321" / "summary.json"
    )
    brain_v26 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v26_20260321" / "summary.json"
    )

    hv = v31["headline_metrics"]
    hs = low_risk_zone["headline_metrics"]
    hb = brain_v26["headline_metrics"]

    plasticity_rule_alignment_v32 = _clip01(
        hv["plasticity_rule_alignment_v31"] * 0.28
        + hs["low_risk_zone_learning"] * 0.24
        + (1.0 - hs["low_risk_zone_penalty"]) * 0.14
        + hb["direct_feature_measure_v26"] * 0.14
        + (1.0 - hb["direct_brain_gap_v26"]) * 0.20
    )
    structure_rule_alignment_v32 = _clip01(
        hv["structure_rule_alignment_v31"] * 0.28
        + hs["low_risk_zone_structure"] * 0.24
        + hs["low_risk_zone_route"] * 0.14
        + (1.0 - hs["low_risk_zone_penalty"]) * 0.10
        + hb["direct_structure_measure_v26"] * 0.24
    )
    topology_training_readiness_v32 = _clip01(
        hv["topology_training_readiness_v31"] * 0.30
        + plasticity_rule_alignment_v32 * 0.15
        + structure_rule_alignment_v32 * 0.15
        + hs["low_risk_zone_readiness"] * 0.15
        + hb["direct_low_risk_zone_alignment_v26"] * 0.15
        + (1.0 - hs["low_risk_zone_penalty"]) * 0.10
    )
    topology_training_gap_v32 = max(0.0, 1.0 - topology_training_readiness_v32)
    low_risk_zone_guard_v32 = _clip01(
        (
            hs["low_risk_zone_structure"]
            + hs["low_risk_zone_route"]
            + hs["low_risk_zone_strength"]
            + topology_training_readiness_v32
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v32": plasticity_rule_alignment_v32,
            "structure_rule_alignment_v32": structure_rule_alignment_v32,
            "topology_training_readiness_v32": topology_training_readiness_v32,
            "topology_training_gap_v32": topology_training_gap_v32,
            "low_risk_zone_guard_v32": low_risk_zone_guard_v32,
        },
        "bridge_equation_v32": {
            "plasticity_term": "B_plastic_v32 = mix(B_plastic_v31, L_zone, 1 - P_zone, D_feature_v26, 1 - G_brain_v26)",
            "structure_term": "B_struct_v32 = mix(B_struct_v31, S_zone, R_zone, 1 - P_zone, D_structure_v26)",
            "readiness_term": "R_train_v32 = mix(R_train_v31, B_plastic_v32, B_struct_v32, R_zone, D_align_v26, 1 - P_zone)",
            "gap_term": "G_train_v32 = 1 - R_train_v32",
            "guard_term": "H_zone_v32 = mean(S_zone, R_zone, A_zone, R_train_v32)",
        },
        "project_readout": {
            "summary": "training bridge v32 checks whether the low-risk steady zone starts shrinking rule-layer risk in a more stable way.",
            "next_question": "next verify whether this low-risk zone still survives after it is folded back into the next closed-form core.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Training Terminal Bridge V32 Report",
        "",
        f"- plasticity_rule_alignment_v32: {hm['plasticity_rule_alignment_v32']:.6f}",
        f"- structure_rule_alignment_v32: {hm['structure_rule_alignment_v32']:.6f}",
        f"- topology_training_readiness_v32: {hm['topology_training_readiness_v32']:.6f}",
        f"- topology_training_gap_v32: {hm['topology_training_gap_v32']:.6f}",
        f"- low_risk_zone_guard_v32: {hm['low_risk_zone_guard_v32']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v32_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
