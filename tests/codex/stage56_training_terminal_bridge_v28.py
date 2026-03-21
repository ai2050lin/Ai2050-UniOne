from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v28_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v28_summary() -> dict:
    v27 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v27_20260321" / "summary.json"
    )
    stable_plus = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_stable_amplification_strengthening_20260321" / "summary.json"
    )
    brain_v22 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v22_20260321" / "summary.json"
    )

    hv = v27["headline_metrics"]
    hs = stable_plus["headline_metrics"]
    hb = brain_v22["headline_metrics"]

    plasticity_rule_alignment_v28 = _clip01(
        hv["plasticity_rule_alignment_v27"] * 0.28
        + hs["stable_reinforced_learning"] * 0.24
        + (1.0 - hs["stable_reinforced_penalty"]) * 0.14
        + hb["direct_feature_measure_v22"] * 0.14
        + (1.0 - hb["direct_brain_gap_v22"]) * 0.20
    )
    structure_rule_alignment_v28 = _clip01(
        hv["structure_rule_alignment_v27"] * 0.28
        + hs["stable_reinforced_structure"] * 0.24
        + hs["stable_reinforced_route"] * 0.14
        + (1.0 - hs["stable_reinforced_penalty"]) * 0.10
        + hb["direct_structure_measure_v22"] * 0.24
    )
    topology_training_readiness_v28 = _clip01(
        hv["topology_training_readiness_v27"] * 0.30
        + plasticity_rule_alignment_v28 * 0.15
        + structure_rule_alignment_v28 * 0.15
        + hs["stable_reinforced_readiness"] * 0.15
        + hb["direct_stable_alignment_v22"] * 0.15
        + (1.0 - hs["stable_reinforced_penalty"]) * 0.10
    )
    topology_training_gap_v28 = max(0.0, 1.0 - topology_training_readiness_v28)
    stable_guard_v28 = _clip01(
        (
            hs["stable_reinforced_structure"]
            + hs["stable_reinforced_route"]
            + hs["stable_reinforced_strength"]
            + topology_training_readiness_v28
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v28": plasticity_rule_alignment_v28,
            "structure_rule_alignment_v28": structure_rule_alignment_v28,
            "topology_training_readiness_v28": topology_training_readiness_v28,
            "topology_training_gap_v28": topology_training_gap_v28,
            "stable_guard_v28": stable_guard_v28,
        },
        "bridge_equation_v28": {
            "plasticity_term": "B_plastic_v28 = mix(B_plastic_v27, L_stable_plus, 1 - P_stable_plus, D_feature_v22, 1 - G_brain_v22)",
            "structure_term": "B_struct_v28 = mix(B_struct_v27, S_stable_plus, R_stable_plus, 1 - P_stable_plus, D_structure_v22)",
            "readiness_term": "R_train_v28 = mix(R_train_v27, B_plastic_v28, B_struct_v28, R_stable_plus, D_align_v22, 1 - P_stable_plus)",
            "gap_term": "G_train_v28 = 1 - R_train_v28",
            "guard_term": "H_stable_v28 = mean(S_stable_plus, R_stable_plus, A_stable_plus, R_train_v28)",
        },
        "project_readout": {
            "summary": "训练终式第二十八桥开始吸收稳定放大强化和脑编码第二十二版，检查放大趋势是否继续落成更低风险的规则层承接。",
            "next_question": "下一步要把第二十八桥并回主核，验证稳定放大是否继续走向低风险稳态施工区。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第二十八桥报告",
        "",
        f"- plasticity_rule_alignment_v28: {hm['plasticity_rule_alignment_v28']:.6f}",
        f"- structure_rule_alignment_v28: {hm['structure_rule_alignment_v28']:.6f}",
        f"- topology_training_readiness_v28: {hm['topology_training_readiness_v28']:.6f}",
        f"- topology_training_gap_v28: {hm['topology_training_gap_v28']:.6f}",
        f"- stable_guard_v28: {hm['stable_guard_v28']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v28_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
