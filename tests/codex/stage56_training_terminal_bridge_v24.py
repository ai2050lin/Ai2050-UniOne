from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v24_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v24_summary() -> dict:
    v23 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v23_20260321" / "summary.json"
    )
    reinforce = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_sustained_amplification_strengthening_20260321" / "summary.json"
    )
    brain_v18 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v18_20260321" / "summary.json"
    )

    hv = v23["headline_metrics"]
    hr = reinforce["headline_metrics"]
    hb = brain_v18["headline_metrics"]

    plasticity_rule_alignment_v24 = _clip01(
        hv["plasticity_rule_alignment_v23"] * 0.28
        + hr["amplification_learning_lift"] * 0.22
        + (1.0 - hr["amplification_residual_penalty"]) * 0.15
        + hb["direct_feature_measure_v18"] * 0.15
        + (1.0 - hb["direct_brain_gap_v18"]) * 0.20
    )
    structure_rule_alignment_v24 = _clip01(
        hv["structure_rule_alignment_v23"] * 0.28
        + hr["amplification_structure_stability"] * 0.22
        + hr["amplification_route_stability"] * 0.15
        + (1.0 - hr["amplification_residual_penalty"]) * 0.10
        + hb["direct_structure_measure_v18"] * 0.25
    )
    topology_training_readiness_v24 = _clip01(
        hv["topology_training_readiness_v23"] * 0.30
        + plasticity_rule_alignment_v24 * 0.15
        + structure_rule_alignment_v24 * 0.15
        + hr["amplification_reinforced_readiness"] * 0.15
        + hb["direct_reinforced_alignment_v18"] * 0.15
        + (1.0 - hr["amplification_residual_penalty"]) * 0.10
    )
    topology_training_gap_v24 = max(0.0, 1.0 - topology_training_readiness_v24)
    reinforcement_guard_v24 = _clip01(
        (
            hr["amplification_structure_stability"]
            + hr["amplification_route_stability"]
            + hr["amplification_strength"]
            + topology_training_readiness_v24
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v24": plasticity_rule_alignment_v24,
            "structure_rule_alignment_v24": structure_rule_alignment_v24,
            "topology_training_readiness_v24": topology_training_readiness_v24,
            "topology_training_gap_v24": topology_training_gap_v24,
            "reinforcement_guard_v24": reinforcement_guard_v24,
        },
        "bridge_equation_v24": {
            "plasticity_term": "B_plastic_v24 = mix(B_plastic_v23, L_reinforce, 1 - P_reinforce, D_feature_v18, 1 - G_brain_v18)",
            "structure_term": "B_struct_v24 = mix(B_struct_v23, S_reinforce, R_reinforce, 1 - P_reinforce, D_structure_v18)",
            "readiness_term": "R_train_v24 = mix(R_train_v23, B_plastic_v24, B_struct_v24, R_reinforce, D_align_v18, 1 - P_reinforce)",
            "gap_term": "G_train_v24 = 1 - R_train_v24",
            "guard_term": "H_reinforce_v24 = mean(S_reinforce, R_reinforce, A_reinforce, R_train_v24)",
        },
        "project_readout": {
            "summary": "训练终式第二十四桥开始吸收持续放大强化和脑编码第十八版，检查放大趋势是否开始转成更稳的规则层放大。",
            "next_question": "下一步要把第二十四桥并回主核，验证放大是否开始从轻度增强转向稳态增强。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第二十四桥报告",
        "",
        f"- plasticity_rule_alignment_v24: {hm['plasticity_rule_alignment_v24']:.6f}",
        f"- structure_rule_alignment_v24: {hm['structure_rule_alignment_v24']:.6f}",
        f"- topology_training_readiness_v24: {hm['topology_training_readiness_v24']:.6f}",
        f"- topology_training_gap_v24: {hm['topology_training_gap_v24']:.6f}",
        f"- reinforcement_guard_v24: {hm['reinforcement_guard_v24']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v24_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
