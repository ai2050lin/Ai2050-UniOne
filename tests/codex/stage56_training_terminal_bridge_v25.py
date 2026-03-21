from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v25_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v25_summary() -> dict:
    v24 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v24_20260321" / "summary.json"
    )
    steady = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_steady_amplification_validation_20260321" / "summary.json"
    )
    brain_v19 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v19_20260321" / "summary.json"
    )

    hv = v24["headline_metrics"]
    hs = steady["headline_metrics"]
    hb = brain_v19["headline_metrics"]

    plasticity_rule_alignment_v25 = _clip01(
        hv["plasticity_rule_alignment_v24"] * 0.28
        + hs["steady_learning_lift"] * 0.24
        + (1.0 - hs["steady_residual_penalty"]) * 0.14
        + hb["direct_feature_measure_v19"] * 0.14
        + (1.0 - hb["direct_brain_gap_v19"]) * 0.20
    )
    structure_rule_alignment_v25 = _clip01(
        hv["structure_rule_alignment_v24"] * 0.28
        + hs["steady_structure_stability"] * 0.24
        + hs["steady_route_stability"] * 0.14
        + (1.0 - hs["steady_residual_penalty"]) * 0.10
        + hb["direct_structure_measure_v19"] * 0.24
    )
    topology_training_readiness_v25 = _clip01(
        hv["topology_training_readiness_v24"] * 0.30
        + plasticity_rule_alignment_v25 * 0.15
        + structure_rule_alignment_v25 * 0.15
        + hs["steady_readiness"] * 0.15
        + hb["direct_steady_alignment_v19"] * 0.15
        + (1.0 - hs["steady_residual_penalty"]) * 0.10
    )
    topology_training_gap_v25 = max(0.0, 1.0 - topology_training_readiness_v25)
    steady_guard_v25 = _clip01(
        (
            hs["steady_structure_stability"]
            + hs["steady_route_stability"]
            + hs["steady_amplification_strength"]
            + topology_training_readiness_v25
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v25": plasticity_rule_alignment_v25,
            "structure_rule_alignment_v25": structure_rule_alignment_v25,
            "topology_training_readiness_v25": topology_training_readiness_v25,
            "topology_training_gap_v25": topology_training_gap_v25,
            "steady_guard_v25": steady_guard_v25,
        },
        "bridge_equation_v25": {
            "plasticity_term": "B_plastic_v25 = mix(B_plastic_v24, L_steady, 1 - P_steady, D_feature_v19, 1 - G_brain_v19)",
            "structure_term": "B_struct_v25 = mix(B_struct_v24, S_steady, R_steady, 1 - P_steady, D_structure_v19)",
            "readiness_term": "R_train_v25 = mix(R_train_v24, B_plastic_v25, B_struct_v25, R_steady, D_align_v19, 1 - P_steady)",
            "gap_term": "G_train_v25 = 1 - R_train_v25",
            "guard_term": "H_steady_v25 = mean(S_steady, R_steady, A_steady, R_train_v25)",
        },
        "project_readout": {
            "summary": "训练终式第二十五桥开始吸收稳态放大验证和脑编码第十九版，检查放大趋势是否开始落成更稳定的规则层稳态。",
            "next_question": "下一步要把第二十五桥并回主核，验证稳态放大是否开始真正成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第二十五桥报告",
        "",
        f"- plasticity_rule_alignment_v25: {hm['plasticity_rule_alignment_v25']:.6f}",
        f"- structure_rule_alignment_v25: {hm['structure_rule_alignment_v25']:.6f}",
        f"- topology_training_readiness_v25: {hm['topology_training_readiness_v25']:.6f}",
        f"- topology_training_gap_v25: {hm['topology_training_gap_v25']:.6f}",
        f"- steady_guard_v25: {hm['steady_guard_v25']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v25_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
