from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v27_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v27_summary() -> dict:
    v26 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v26_20260321" / "summary.json"
    )
    stable = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_stable_amplification_validation_20260321" / "summary.json"
    )
    brain_v21 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v21_20260321" / "summary.json"
    )

    hv = v26["headline_metrics"]
    hs = stable["headline_metrics"]
    hb = brain_v21["headline_metrics"]

    plasticity_rule_alignment_v27 = _clip01(
        hv["plasticity_rule_alignment_v26"] * 0.28
        + hs["stable_learning_lift"] * 0.24
        + (1.0 - hs["stable_residual_penalty"]) * 0.14
        + hb["direct_feature_measure_v21"] * 0.14
        + (1.0 - hb["direct_brain_gap_v21"]) * 0.20
    )
    structure_rule_alignment_v27 = _clip01(
        hv["structure_rule_alignment_v26"] * 0.28
        + hs["stable_structure_stability"] * 0.24
        + hs["stable_route_stability"] * 0.14
        + (1.0 - hs["stable_residual_penalty"]) * 0.10
        + hb["direct_structure_measure_v21"] * 0.24
    )
    topology_training_readiness_v27 = _clip01(
        hv["topology_training_readiness_v26"] * 0.30
        + plasticity_rule_alignment_v27 * 0.15
        + structure_rule_alignment_v27 * 0.15
        + hs["stable_readiness"] * 0.15
        + hb["direct_stable_alignment_v21"] * 0.15
        + (1.0 - hs["stable_residual_penalty"]) * 0.10
    )
    topology_training_gap_v27 = max(0.0, 1.0 - topology_training_readiness_v27)
    stable_guard_v27 = _clip01(
        (
            hs["stable_structure_stability"]
            + hs["stable_route_stability"]
            + hs["stable_amplification_strength"]
            + topology_training_readiness_v27
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v27": plasticity_rule_alignment_v27,
            "structure_rule_alignment_v27": structure_rule_alignment_v27,
            "topology_training_readiness_v27": topology_training_readiness_v27,
            "topology_training_gap_v27": topology_training_gap_v27,
            "stable_guard_v27": stable_guard_v27,
        },
        "bridge_equation_v27": {
            "plasticity_term": "B_plastic_v27 = mix(B_plastic_v26, L_stable, 1 - P_stable, D_feature_v21, 1 - G_brain_v21)",
            "structure_term": "B_struct_v27 = mix(B_struct_v26, S_stable, R_stable, 1 - P_stable, D_structure_v21)",
            "readiness_term": "R_train_v27 = mix(R_train_v26, B_plastic_v27, B_struct_v27, R_stable, D_align_v21, 1 - P_stable)",
            "gap_term": "G_train_v27 = 1 - R_train_v27",
            "guard_term": "H_stable_v27 = mean(S_stable, R_stable, A_stable, R_train_v27)",
        },
        "project_readout": {
            "summary": "训练终式第二十七桥开始吸收稳定放大验证和脑编码第二十一版，检查放大趋势是否继续落成更低风险的规则层承接。",
            "next_question": "下一步要把第二十七桥并回主核，验证稳定放大是否继续走向低风险稳态施工区。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第二十七桥报告",
        "",
        f"- plasticity_rule_alignment_v27: {hm['plasticity_rule_alignment_v27']:.6f}",
        f"- structure_rule_alignment_v27: {hm['structure_rule_alignment_v27']:.6f}",
        f"- topology_training_readiness_v27: {hm['topology_training_readiness_v27']:.6f}",
        f"- topology_training_gap_v27: {hm['topology_training_gap_v27']:.6f}",
        f"- stable_guard_v27: {hm['stable_guard_v27']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v27_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
