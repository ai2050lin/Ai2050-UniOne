from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v21_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v21_summary() -> dict:
    v20 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v20_20260321" / "summary.json"
    )
    stable = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_stable_amplification_validation_20260321" / "summary.json"
    )
    bridge_v26 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v26_20260321" / "summary.json"
    )

    hv = v20["headline_metrics"]
    hs = stable["headline_metrics"]
    hb = bridge_v26["headline_metrics"]

    direct_origin_measure_v21 = _clip01(
        hv["direct_origin_measure_v20"] * 0.46
        + hs["stable_readiness"] * 0.22
        + (1.0 - hs["stable_residual_penalty"]) * 0.15
        + hb["topology_training_readiness_v26"] * 0.17
    )
    direct_feature_measure_v21 = _clip01(
        hv["direct_feature_measure_v20"] * 0.44
        + hs["stable_learning_lift"] * 0.26
        + (1.0 - hs["stable_residual_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v26"] * 0.20
    )
    direct_structure_measure_v21 = _clip01(
        hv["direct_structure_measure_v20"] * 0.42
        + hs["stable_structure_stability"] * 0.28
        + (1.0 - hs["stable_residual_penalty"]) * 0.10
        + hb["structure_rule_alignment_v26"] * 0.20
    )
    direct_route_measure_v21 = _clip01(
        hv["direct_route_measure_v20"] * 0.42
        + hs["stable_route_stability"] * 0.28
        + hs["stable_structure_stability"] * 0.08
        + (1.0 - hs["stable_residual_penalty"]) * 0.05
        + hb["steady_guard_v26"] * 0.17
    )
    direct_brain_measure_v21 = (
        direct_origin_measure_v21
        + direct_feature_measure_v21
        + direct_structure_measure_v21
        + direct_route_measure_v21
    ) / 4.0
    direct_brain_gap_v21 = 1.0 - direct_brain_measure_v21
    direct_stable_alignment_v21 = (
        direct_structure_measure_v21
        + direct_route_measure_v21
        + hs["stable_readiness"]
        + hb["topology_training_readiness_v26"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v21": direct_origin_measure_v21,
            "direct_feature_measure_v21": direct_feature_measure_v21,
            "direct_structure_measure_v21": direct_structure_measure_v21,
            "direct_route_measure_v21": direct_route_measure_v21,
            "direct_brain_measure_v21": direct_brain_measure_v21,
            "direct_brain_gap_v21": direct_brain_gap_v21,
            "direct_stable_alignment_v21": direct_stable_alignment_v21,
        },
        "direct_equation_v21": {
            "origin_term": "D_origin_v21 = 0.46 * D_origin_v20 + 0.22 * R_stable + 0.15 * (1 - P_stable) + 0.17 * R_train_v26",
            "feature_term": "D_feature_v21 = 0.44 * D_feature_v20 + 0.26 * L_stable + 0.10 * (1 - P_stable) + 0.20 * B_plastic_v26",
            "structure_term": "D_structure_v21 = 0.42 * D_structure_v20 + 0.28 * S_stable + 0.10 * (1 - P_stable) + 0.20 * B_struct_v26",
            "route_term": "D_route_v21 = 0.42 * D_route_v20 + 0.28 * R_stable + 0.08 * S_stable + 0.05 * (1 - P_stable) + 0.17 * H_steady_v26",
            "system_term": "M_brain_direct_v21 = mean(D_origin_v21, D_feature_v21, D_structure_v21, D_route_v21)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第二十一版开始把稳定放大验证并回脑编码链，检查放大趋势是否继续转成更稳定的脑编码承接。",
            "next_question": "下一步要把第二十一版直测链并回训练终式和主核，确认稳定放大是否继续在脑编码层增强。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第二十一版报告",
        "",
        f"- direct_origin_measure_v21: {hm['direct_origin_measure_v21']:.6f}",
        f"- direct_feature_measure_v21: {hm['direct_feature_measure_v21']:.6f}",
        f"- direct_structure_measure_v21: {hm['direct_structure_measure_v21']:.6f}",
        f"- direct_route_measure_v21: {hm['direct_route_measure_v21']:.6f}",
        f"- direct_brain_measure_v21: {hm['direct_brain_measure_v21']:.6f}",
        f"- direct_brain_gap_v21: {hm['direct_brain_gap_v21']:.6f}",
        f"- direct_stable_alignment_v21: {hm['direct_stable_alignment_v21']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v21_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
