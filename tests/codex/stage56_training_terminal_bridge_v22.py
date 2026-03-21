from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v22_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v22_summary() -> dict:
    v21 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v21_20260321" / "summary.json"
    )
    sustained = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_sustained_rebound_validation_20260321" / "summary.json"
    )
    brain_v16 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v16_20260321" / "summary.json"
    )

    hv = v21["headline_metrics"]
    hs = sustained["headline_metrics"]
    hb = brain_v16["headline_metrics"]

    plasticity_rule_alignment_v22 = _clip01(
        hv["plasticity_rule_alignment_v21"] * 0.28
        + hs["sustained_learning"] * 0.20
        + (1.0 - hs["sustained_penalty"]) * 0.15
        + hb["direct_feature_measure_v16"] * 0.17
        + (1.0 - hb["direct_brain_gap_v16"]) * 0.20
    )
    structure_rule_alignment_v22 = _clip01(
        hv["structure_rule_alignment_v21"] * 0.28
        + hs["sustained_structure"] * 0.20
        + hs["sustained_route"] * 0.15
        + (1.0 - hs["sustained_penalty"]) * 0.12
        + hb["direct_structure_measure_v16"] * 0.25
    )
    topology_training_readiness_v22 = _clip01(
        hv["topology_training_readiness_v21"] * 0.30
        + plasticity_rule_alignment_v22 * 0.15
        + structure_rule_alignment_v22 * 0.15
        + hs["sustained_readiness"] * 0.15
        + hb["direct_sustained_alignment_v16"] * 0.15
        + (1.0 - hs["sustained_penalty"]) * 0.10
    )
    topology_training_gap_v22 = max(0.0, 1.0 - topology_training_readiness_v22)
    sustained_guard_v22 = _clip01(
        (
            hs["sustained_structure"]
            + hs["sustained_context"]
            + hs["sustained_route"]
            + topology_training_readiness_v22
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v22": plasticity_rule_alignment_v22,
            "structure_rule_alignment_v22": structure_rule_alignment_v22,
            "topology_training_readiness_v22": topology_training_readiness_v22,
            "topology_training_gap_v22": topology_training_gap_v22,
            "sustained_guard_v22": sustained_guard_v22,
        },
        "bridge_equation_v22": {
            "plasticity_term": "B_plastic_v22 = mix(B_plastic_v21, L_sustain, 1 - P_sustain, D_feature_v16, 1 - G_brain_v16)",
            "structure_term": "B_struct_v22 = mix(B_struct_v21, S_sustain, R_sustain, 1 - P_sustain, D_structure_v16)",
            "readiness_term": "R_train_v22 = mix(R_train_v21, B_plastic_v22, B_struct_v22, R_sustain, D_align_v16, 1 - P_sustain)",
            "gap_term": "G_train_v22 = 1 - R_train_v22",
            "guard_term": "H_sustain_v22 = mean(S_sustain, C_sustain, R_sustain, R_train_v22)",
        },
        "project_readout": {
            "summary": "训练终式第二十二桥开始吸收持续回升验证和脑编码第十六版，检查回升是否开始固化为更稳定的规则层护栏。",
            "next_question": "下一步要把第二十二桥并回主核，验证持续化回升能否开始在更大系统里维持，而不是重新掉回衰减区。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第二十二桥报告",
        "",
        f"- plasticity_rule_alignment_v22: {hm['plasticity_rule_alignment_v22']:.6f}",
        f"- structure_rule_alignment_v22: {hm['structure_rule_alignment_v22']:.6f}",
        f"- topology_training_readiness_v22: {hm['topology_training_readiness_v22']:.6f}",
        f"- topology_training_gap_v22: {hm['topology_training_gap_v22']:.6f}",
        f"- sustained_guard_v22: {hm['sustained_guard_v22']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v22_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
