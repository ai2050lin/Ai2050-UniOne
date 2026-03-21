from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v17_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v17_summary() -> dict:
    v16 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v16_20260321" / "summary.json"
    )
    amplification = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_sustained_amplification_validation_20260321" / "summary.json"
    )
    bridge_v22 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v22_20260321" / "summary.json"
    )

    hv = v16["headline_metrics"]
    ha = amplification["headline_metrics"]
    hb = bridge_v22["headline_metrics"]

    direct_origin_measure_v17 = _clip01(
        hv["direct_origin_measure_v16"] * 0.50
        + ha["amplification_readiness"] * 0.20
        + (1.0 - ha["amplification_penalty"]) * 0.15
        + hb["topology_training_readiness_v22"] * 0.15
    )
    direct_feature_measure_v17 = _clip01(
        hv["direct_feature_measure_v16"] * 0.48
        + ha["amplification_learning"] * 0.22
        + (1.0 - ha["amplification_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v22"] * 0.20
    )
    direct_structure_measure_v17 = _clip01(
        hv["direct_structure_measure_v16"] * 0.46
        + ha["amplification_structure"] * 0.24
        + (1.0 - ha["amplification_penalty"]) * 0.10
        + hb["structure_rule_alignment_v22"] * 0.20
    )
    direct_route_measure_v17 = _clip01(
        hv["direct_route_measure_v16"] * 0.46
        + ha["amplification_route"] * 0.24
        + ha["amplification_context"] * 0.10
        + (1.0 - ha["amplification_penalty"]) * 0.05
        + hb["sustained_guard_v22"] * 0.15
    )
    direct_brain_measure_v17 = (
        direct_origin_measure_v17
        + direct_feature_measure_v17
        + direct_structure_measure_v17
        + direct_route_measure_v17
    ) / 4.0
    direct_brain_gap_v17 = 1.0 - direct_brain_measure_v17
    direct_amplification_alignment_v17 = (
        direct_structure_measure_v17
        + direct_route_measure_v17
        + ha["amplification_readiness"]
        + hb["topology_training_readiness_v22"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v17": direct_origin_measure_v17,
            "direct_feature_measure_v17": direct_feature_measure_v17,
            "direct_structure_measure_v17": direct_structure_measure_v17,
            "direct_route_measure_v17": direct_route_measure_v17,
            "direct_brain_measure_v17": direct_brain_measure_v17,
            "direct_brain_gap_v17": direct_brain_gap_v17,
            "direct_amplification_alignment_v17": direct_amplification_alignment_v17,
        },
        "direct_equation_v17": {
            "origin_term": "D_origin_v17 = 0.50 * D_origin_v16 + 0.20 * R_amp + 0.15 * (1 - P_amp) + 0.15 * R_train_v22",
            "feature_term": "D_feature_v17 = 0.48 * D_feature_v16 + 0.22 * L_amp + 0.10 * (1 - P_amp) + 0.20 * B_plastic_v22",
            "structure_term": "D_structure_v17 = 0.46 * D_structure_v16 + 0.24 * S_amp + 0.10 * (1 - P_amp) + 0.20 * B_struct_v22",
            "route_term": "D_route_v17 = 0.46 * D_route_v16 + 0.24 * R_amp + 0.10 * C_amp + 0.05 * (1 - P_amp) + 0.15 * H_sustain_v22",
            "system_term": "M_brain_direct_v17 = mean(D_origin_v17, D_feature_v17, D_structure_v17, D_route_v17)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第十七版开始把持续放大验证并回脑编码链，检查持续回升是否开始向系统级放大推进。",
            "next_question": "下一步要把第十七版直测链并回训练终式和主核，确认放大趋势是否真的进入脑编码层。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第十七版报告",
        "",
        f"- direct_origin_measure_v17: {hm['direct_origin_measure_v17']:.6f}",
        f"- direct_feature_measure_v17: {hm['direct_feature_measure_v17']:.6f}",
        f"- direct_structure_measure_v17: {hm['direct_structure_measure_v17']:.6f}",
        f"- direct_route_measure_v17: {hm['direct_route_measure_v17']:.6f}",
        f"- direct_brain_measure_v17: {hm['direct_brain_measure_v17']:.6f}",
        f"- direct_brain_gap_v17: {hm['direct_brain_gap_v17']:.6f}",
        f"- direct_amplification_alignment_v17: {hm['direct_amplification_alignment_v17']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v17_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
