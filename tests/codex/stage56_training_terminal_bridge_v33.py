from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v33_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v33_summary() -> dict:
    v32 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v32_20260321" / "summary.json"
    )
    expansion = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_low_risk_steady_zone_expansion_validation_20260321" / "summary.json"
    )
    brain_v27 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v27_20260321" / "summary.json"
    )

    hv = v32["headline_metrics"]
    he = expansion["headline_metrics"]
    hb = brain_v27["headline_metrics"]

    plasticity_rule_alignment_v33 = _clip01(
        hv["plasticity_rule_alignment_v32"] * 0.28
        + he["low_risk_expansion_learning"] * 0.24
        + (1.0 - he["low_risk_expansion_penalty"]) * 0.14
        + hb["direct_feature_measure_v27"] * 0.14
        + (1.0 - hb["direct_brain_gap_v27"]) * 0.20
    )
    structure_rule_alignment_v33 = _clip01(
        hv["structure_rule_alignment_v32"] * 0.28
        + he["low_risk_expansion_structure"] * 0.24
        + he["low_risk_expansion_route"] * 0.14
        + (1.0 - he["low_risk_expansion_penalty"]) * 0.10
        + hb["direct_structure_measure_v27"] * 0.24
    )
    topology_training_readiness_v33 = _clip01(
        hv["topology_training_readiness_v32"] * 0.30
        + plasticity_rule_alignment_v33 * 0.15
        + structure_rule_alignment_v33 * 0.15
        + he["low_risk_expansion_readiness"] * 0.15
        + hb["direct_expansion_alignment_v27"] * 0.15
        + (1.0 - he["low_risk_expansion_penalty"]) * 0.10
    )
    topology_training_gap_v33 = max(0.0, 1.0 - topology_training_readiness_v33)
    low_risk_expansion_guard_v33 = _clip01(
        (
            he["low_risk_expansion_structure"]
            + he["low_risk_expansion_route"]
            + he["low_risk_expansion_strength"]
            + topology_training_readiness_v33
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v33": plasticity_rule_alignment_v33,
            "structure_rule_alignment_v33": structure_rule_alignment_v33,
            "topology_training_readiness_v33": topology_training_readiness_v33,
            "topology_training_gap_v33": topology_training_gap_v33,
            "low_risk_expansion_guard_v33": low_risk_expansion_guard_v33,
        },
        "bridge_equation_v33": {
            "plasticity_term": "B_plastic_v33 = mix(B_plastic_v32, L_expand, 1 - P_expand, D_feature_v27, 1 - G_brain_v27)",
            "structure_term": "B_struct_v33 = mix(B_struct_v32, S_expand, R_expand, 1 - P_expand, D_structure_v27)",
            "readiness_term": "R_train_v33 = mix(R_train_v32, B_plastic_v33, B_struct_v33, R_expand, D_align_v27, 1 - P_expand)",
            "gap_term": "G_train_v33 = 1 - R_train_v33",
            "guard_term": "H_expand_v33 = mean(S_expand, R_expand, A_expand, R_train_v33)",
        },
        "project_readout": {
            "summary": "training bridge v33 checks whether low-risk steady expansion begins to reduce rule-layer risk in a sustained way.",
            "next_question": "next verify whether this expansion still survives after it is folded into the next closed-form core.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Training Terminal Bridge V33 Report",
        "",
        f"- plasticity_rule_alignment_v33: {hm['plasticity_rule_alignment_v33']:.6f}",
        f"- structure_rule_alignment_v33: {hm['structure_rule_alignment_v33']:.6f}",
        f"- topology_training_readiness_v33: {hm['topology_training_readiness_v33']:.6f}",
        f"- topology_training_gap_v33: {hm['topology_training_gap_v33']:.6f}",
        f"- low_risk_expansion_guard_v33: {hm['low_risk_expansion_guard_v33']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v33_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
