from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v23_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v23_summary() -> dict:
    v22 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v22_20260321" / "summary.json"
    )
    systemic = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_stable_amplification_validation_20260321" / "summary.json"
    )
    bridge_v28 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v28_20260321" / "summary.json"
    )

    hv = v22["headline_metrics"]
    hs = systemic["headline_metrics"]
    hb = bridge_v28["headline_metrics"]

    direct_origin_measure_v23 = _clip01(
        hv["direct_origin_measure_v22"] * 0.46
        + hs["systemic_readiness"] * 0.22
        + (1.0 - hs["systemic_residual_penalty"]) * 0.15
        + hb["topology_training_readiness_v28"] * 0.17
    )
    direct_feature_measure_v23 = _clip01(
        hv["direct_feature_measure_v22"] * 0.44
        + hs["systemic_learning_lift"] * 0.26
        + (1.0 - hs["systemic_residual_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v28"] * 0.20
    )
    direct_structure_measure_v23 = _clip01(
        hv["direct_structure_measure_v22"] * 0.42
        + hs["systemic_structure_stability"] * 0.28
        + (1.0 - hs["systemic_residual_penalty"]) * 0.10
        + hb["structure_rule_alignment_v28"] * 0.20
    )
    direct_route_measure_v23 = _clip01(
        hv["direct_route_measure_v22"] * 0.42
        + hs["systemic_route_stability"] * 0.28
        + hs["systemic_structure_stability"] * 0.08
        + (1.0 - hs["systemic_residual_penalty"]) * 0.05
        + hb["stable_guard_v28"] * 0.17
    )
    direct_brain_measure_v23 = (
        direct_origin_measure_v23
        + direct_feature_measure_v23
        + direct_structure_measure_v23
        + direct_route_measure_v23
    ) / 4.0
    direct_brain_gap_v23 = 1.0 - direct_brain_measure_v23
    direct_systemic_alignment_v23 = (
        direct_structure_measure_v23
        + direct_route_measure_v23
        + hs["systemic_readiness"]
        + hb["topology_training_readiness_v28"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v23": direct_origin_measure_v23,
            "direct_feature_measure_v23": direct_feature_measure_v23,
            "direct_structure_measure_v23": direct_structure_measure_v23,
            "direct_route_measure_v23": direct_route_measure_v23,
            "direct_brain_measure_v23": direct_brain_measure_v23,
            "direct_brain_gap_v23": direct_brain_gap_v23,
            "direct_systemic_alignment_v23": direct_systemic_alignment_v23,
        },
        "direct_equation_v23": {
            "origin_term": "D_origin_v23 = 0.46 * D_origin_v22 + 0.22 * R_system + 0.15 * (1 - P_system) + 0.17 * R_train_v28",
            "feature_term": "D_feature_v23 = 0.44 * D_feature_v22 + 0.26 * L_system + 0.10 * (1 - P_system) + 0.20 * B_plastic_v28",
            "structure_term": "D_structure_v23 = 0.42 * D_structure_v22 + 0.28 * S_system + 0.10 * (1 - P_system) + 0.20 * B_struct_v28",
            "route_term": "D_route_v23 = 0.42 * D_route_v22 + 0.28 * R_system + 0.08 * S_system + 0.05 * (1 - P_system) + 0.17 * H_stable_v28",
            "system_term": "M_brain_direct_v23 = mean(D_origin_v23, D_feature_v23, D_structure_v23, D_route_v23)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第二十三版开始把系统级稳定放大验证并回脑编码链，检查放大趋势是否继续转成更系统化的脑编码承接。",
            "next_question": "下一步要把第二十三版直测链并回训练终式和主核，确认系统级稳定放大是否继续在脑编码层增强。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第二十三版报告",
        "",
        f"- direct_origin_measure_v23: {hm['direct_origin_measure_v23']:.6f}",
        f"- direct_feature_measure_v23: {hm['direct_feature_measure_v23']:.6f}",
        f"- direct_structure_measure_v23: {hm['direct_structure_measure_v23']:.6f}",
        f"- direct_route_measure_v23: {hm['direct_route_measure_v23']:.6f}",
        f"- direct_brain_measure_v23: {hm['direct_brain_measure_v23']:.6f}",
        f"- direct_brain_gap_v23: {hm['direct_brain_gap_v23']:.6f}",
        f"- direct_systemic_alignment_v23: {hm['direct_systemic_alignment_v23']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v23_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
