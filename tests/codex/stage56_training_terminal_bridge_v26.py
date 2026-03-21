from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v26_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v26_summary() -> dict:
    v25 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v25_20260321" / "summary.json"
    )
    steady_plus = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_steady_amplification_reinforcement_20260321" / "summary.json"
    )
    brain_v20 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v20_20260321" / "summary.json"
    )

    hv = v25["headline_metrics"]
    hs = steady_plus["headline_metrics"]
    hb = brain_v20["headline_metrics"]

    plasticity_rule_alignment_v26 = _clip01(
        hv["plasticity_rule_alignment_v25"] * 0.28
        + hs["steady_reinforcement_learning"] * 0.24
        + (1.0 - hs["steady_reinforcement_penalty"]) * 0.14
        + hb["direct_feature_measure_v20"] * 0.14
        + (1.0 - hb["direct_brain_gap_v20"]) * 0.20
    )
    structure_rule_alignment_v26 = _clip01(
        hv["structure_rule_alignment_v25"] * 0.28
        + hs["steady_reinforcement_structure"] * 0.24
        + hs["steady_reinforcement_route"] * 0.14
        + (1.0 - hs["steady_reinforcement_penalty"]) * 0.10
        + hb["direct_structure_measure_v20"] * 0.24
    )
    topology_training_readiness_v26 = _clip01(
        hv["topology_training_readiness_v25"] * 0.30
        + plasticity_rule_alignment_v26 * 0.15
        + structure_rule_alignment_v26 * 0.15
        + hs["steady_reinforcement_readiness"] * 0.15
        + hb["direct_steady_alignment_v20"] * 0.15
        + (1.0 - hs["steady_reinforcement_penalty"]) * 0.10
    )
    topology_training_gap_v26 = max(0.0, 1.0 - topology_training_readiness_v26)
    steady_guard_v26 = _clip01(
        (
            hs["steady_reinforcement_structure"]
            + hs["steady_reinforcement_route"]
            + hs["steady_reinforcement_strength"]
            + topology_training_readiness_v26
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v26": plasticity_rule_alignment_v26,
            "structure_rule_alignment_v26": structure_rule_alignment_v26,
            "topology_training_readiness_v26": topology_training_readiness_v26,
            "topology_training_gap_v26": topology_training_gap_v26,
            "steady_guard_v26": steady_guard_v26,
        },
        "bridge_equation_v26": {
            "plasticity_term": "B_plastic_v26 = mix(B_plastic_v25, L_steady_plus, 1 - P_steady_plus, D_feature_v20, 1 - G_brain_v20)",
            "structure_term": "B_struct_v26 = mix(B_struct_v25, S_steady_plus, R_steady_plus, 1 - P_steady_plus, D_structure_v20)",
            "readiness_term": "R_train_v26 = mix(R_train_v25, B_plastic_v26, B_struct_v26, R_steady_plus, D_align_v20, 1 - P_steady_plus)",
            "gap_term": "G_train_v26 = 1 - R_train_v26",
            "guard_term": "H_steady_v26 = mean(S_steady_plus, R_steady_plus, A_steady_plus, R_train_v26)",
        },
        "project_readout": {
            "summary": "训练终式第二十六桥开始吸收更稳的放大强化和脑编码第二十版，检查放大趋势是否继续落成更稳定的规则层。",
            "next_question": "下一步要把第二十六桥并回主核，验证更稳放大是否继续走向低风险稳态施工区。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第二十六桥报告",
        "",
        f"- plasticity_rule_alignment_v26: {hm['plasticity_rule_alignment_v26']:.6f}",
        f"- structure_rule_alignment_v26: {hm['structure_rule_alignment_v26']:.6f}",
        f"- topology_training_readiness_v26: {hm['topology_training_readiness_v26']:.6f}",
        f"- topology_training_gap_v26: {hm['topology_training_gap_v26']:.6f}",
        f"- steady_guard_v26: {hm['steady_guard_v26']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v26_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
