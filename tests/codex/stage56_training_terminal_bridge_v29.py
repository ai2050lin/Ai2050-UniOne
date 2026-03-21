from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v29_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v29_summary() -> dict:
    v28 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v28_20260321" / "summary.json"
    )
    systemic = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_stable_amplification_validation_20260321" / "summary.json"
    )
    brain_v23 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v23_20260321" / "summary.json"
    )

    hv = v28["headline_metrics"]
    hs = systemic["headline_metrics"]
    hb = brain_v23["headline_metrics"]

    plasticity_rule_alignment_v29 = _clip01(
        hv["plasticity_rule_alignment_v28"] * 0.28
        + hs["systemic_learning_lift"] * 0.24
        + (1.0 - hs["systemic_residual_penalty"]) * 0.14
        + hb["direct_feature_measure_v23"] * 0.14
        + (1.0 - hb["direct_brain_gap_v23"]) * 0.20
    )
    structure_rule_alignment_v29 = _clip01(
        hv["structure_rule_alignment_v28"] * 0.28
        + hs["systemic_structure_stability"] * 0.24
        + hs["systemic_route_stability"] * 0.14
        + (1.0 - hs["systemic_residual_penalty"]) * 0.10
        + hb["direct_structure_measure_v23"] * 0.24
    )
    topology_training_readiness_v29 = _clip01(
        hv["topology_training_readiness_v28"] * 0.30
        + plasticity_rule_alignment_v29 * 0.15
        + structure_rule_alignment_v29 * 0.15
        + hs["systemic_readiness"] * 0.15
        + hb["direct_systemic_alignment_v23"] * 0.15
        + (1.0 - hs["systemic_residual_penalty"]) * 0.10
    )
    topology_training_gap_v29 = max(0.0, 1.0 - topology_training_readiness_v29)
    systemic_guard_v29 = _clip01(
        (
            hs["systemic_structure_stability"]
            + hs["systemic_route_stability"]
            + hs["systemic_amplification_strength"]
            + topology_training_readiness_v29
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v29": plasticity_rule_alignment_v29,
            "structure_rule_alignment_v29": structure_rule_alignment_v29,
            "topology_training_readiness_v29": topology_training_readiness_v29,
            "topology_training_gap_v29": topology_training_gap_v29,
            "systemic_guard_v29": systemic_guard_v29,
        },
        "bridge_equation_v29": {
            "plasticity_term": "B_plastic_v29 = mix(B_plastic_v28, L_system, 1 - P_system, D_feature_v23, 1 - G_brain_v23)",
            "structure_term": "B_struct_v29 = mix(B_struct_v28, S_system, R_system, 1 - P_system, D_structure_v23)",
            "readiness_term": "R_train_v29 = mix(R_train_v28, B_plastic_v29, B_struct_v29, R_system, D_align_v23, 1 - P_system)",
            "gap_term": "G_train_v29 = 1 - R_train_v29",
            "guard_term": "H_system_v29 = mean(S_system, R_system, A_system, R_train_v29)",
        },
        "project_readout": {
            "summary": "训练终式第二十九桥开始吸收系统级稳定放大验证和脑编码第二十三版，检查放大趋势是否继续落成更低风险的规则层承接。",
            "next_question": "下一步要把第二十九桥并回主核，验证系统级稳定放大是否继续走向低风险稳态施工区。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第二十九桥报告",
        "",
        f"- plasticity_rule_alignment_v29: {hm['plasticity_rule_alignment_v29']:.6f}",
        f"- structure_rule_alignment_v29: {hm['structure_rule_alignment_v29']:.6f}",
        f"- topology_training_readiness_v29: {hm['topology_training_readiness_v29']:.6f}",
        f"- topology_training_gap_v29: {hm['topology_training_gap_v29']:.6f}",
        f"- systemic_guard_v29: {hm['systemic_guard_v29']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v29_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
