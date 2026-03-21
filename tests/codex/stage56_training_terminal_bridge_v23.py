from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v23_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v23_summary() -> dict:
    v22 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v22_20260321" / "summary.json"
    )
    amplification = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_sustained_amplification_validation_20260321" / "summary.json"
    )
    brain_v17 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v17_20260321" / "summary.json"
    )

    hv = v22["headline_metrics"]
    ha = amplification["headline_metrics"]
    hb = brain_v17["headline_metrics"]

    plasticity_rule_alignment_v23 = _clip01(
        hv["plasticity_rule_alignment_v22"] * 0.28
        + ha["amplification_learning"] * 0.22
        + (1.0 - ha["amplification_penalty"]) * 0.15
        + hb["direct_feature_measure_v17"] * 0.15
        + (1.0 - hb["direct_brain_gap_v17"]) * 0.20
    )
    structure_rule_alignment_v23 = _clip01(
        hv["structure_rule_alignment_v22"] * 0.28
        + ha["amplification_structure"] * 0.22
        + ha["amplification_route"] * 0.15
        + (1.0 - ha["amplification_penalty"]) * 0.10
        + hb["direct_structure_measure_v17"] * 0.25
    )
    topology_training_readiness_v23 = _clip01(
        hv["topology_training_readiness_v22"] * 0.30
        + plasticity_rule_alignment_v23 * 0.15
        + structure_rule_alignment_v23 * 0.15
        + ha["amplification_readiness"] * 0.15
        + hb["direct_amplification_alignment_v17"] * 0.15
        + (1.0 - ha["amplification_penalty"]) * 0.10
    )
    topology_training_gap_v23 = max(0.0, 1.0 - topology_training_readiness_v23)
    amplification_guard_v23 = _clip01(
        (
            ha["amplification_structure"]
            + ha["amplification_context"]
            + ha["amplification_route"]
            + topology_training_readiness_v23
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v23": plasticity_rule_alignment_v23,
            "structure_rule_alignment_v23": structure_rule_alignment_v23,
            "topology_training_readiness_v23": topology_training_readiness_v23,
            "topology_training_gap_v23": topology_training_gap_v23,
            "amplification_guard_v23": amplification_guard_v23,
        },
        "bridge_equation_v23": {
            "plasticity_term": "B_plastic_v23 = mix(B_plastic_v22, L_amp, 1 - P_amp, D_feature_v17, 1 - G_brain_v17)",
            "structure_term": "B_struct_v23 = mix(B_struct_v22, S_amp, R_amp, 1 - P_amp, D_structure_v17)",
            "readiness_term": "R_train_v23 = mix(R_train_v22, B_plastic_v23, B_struct_v23, R_amp, D_align_v17, 1 - P_amp)",
            "gap_term": "G_train_v23 = 1 - R_train_v23",
            "guard_term": "H_amp_v23 = mean(S_amp, C_amp, R_amp, R_train_v23)",
        },
        "project_readout": {
            "summary": "训练终式第二十三桥开始吸收持续放大验证和脑编码第十七版，检查持续回升是否开始从持续化走向放大化。",
            "next_question": "下一步要把第二十三桥并回主核，验证这次放大趋势能否在更大系统里继续推进。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第二十三桥报告",
        "",
        f"- plasticity_rule_alignment_v23: {hm['plasticity_rule_alignment_v23']:.6f}",
        f"- structure_rule_alignment_v23: {hm['structure_rule_alignment_v23']:.6f}",
        f"- topology_training_readiness_v23: {hm['topology_training_readiness_v23']:.6f}",
        f"- topology_training_gap_v23: {hm['topology_training_gap_v23']:.6f}",
        f"- amplification_guard_v23: {hm['amplification_guard_v23']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v23_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
