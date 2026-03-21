from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v14_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v14_summary() -> dict:
    v13 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v13_20260321" / "summary.json"
    )
    attenuation = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_propagation_attenuation_probe_20260321" / "summary.json"
    )
    bridge_v19 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v19_20260321" / "summary.json"
    )

    hv = v13["headline_metrics"]
    ha = attenuation["headline_metrics"]
    hb = bridge_v19["headline_metrics"]

    direct_origin_measure_v14 = _clip01(
        hv["direct_origin_measure_v13"] * 0.61
        + ha["anti_attenuation_readiness"] * 0.14
        + (1.0 - ha["attenuation_penalty"]) * 0.15
        + hb["topology_training_readiness_v19"] * 0.10
    )
    direct_feature_measure_v14 = _clip01(
        hv["direct_feature_measure_v13"] * 0.59
        + (1.0 - ha["attenuation_learning"]) * 0.16
        + (1.0 - ha["attenuation_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v19"] * 0.15
    )
    direct_structure_measure_v14 = _clip01(
        hv["direct_structure_measure_v13"] * 0.55
        + (1.0 - ha["attenuation_structure"]) * 0.20
        + (1.0 - ha["attenuation_penalty"]) * 0.10
        + hb["structure_rule_alignment_v19"] * 0.15
    )
    direct_route_measure_v14 = _clip01(
        hv["direct_route_measure_v13"] * 0.55
        + (1.0 - ha["attenuation_route"]) * 0.20
        + (1.0 - ha["attenuation_context"]) * 0.10
        + (1.0 - ha["attenuation_penalty"]) * 0.05
        + hb["scale_guard_v19"] * 0.10
    )
    direct_brain_measure_v14 = (
        direct_origin_measure_v14
        + direct_feature_measure_v14
        + direct_structure_measure_v14
        + direct_route_measure_v14
    ) / 4.0
    direct_brain_gap_v14 = 1.0 - direct_brain_measure_v14
    direct_anti_attenuation_alignment_v14 = (
        direct_structure_measure_v14
        + direct_route_measure_v14
        + ha["anti_attenuation_readiness"]
        + hb["topology_training_readiness_v19"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v14": direct_origin_measure_v14,
            "direct_feature_measure_v14": direct_feature_measure_v14,
            "direct_structure_measure_v14": direct_structure_measure_v14,
            "direct_route_measure_v14": direct_route_measure_v14,
            "direct_brain_measure_v14": direct_brain_measure_v14,
            "direct_brain_gap_v14": direct_brain_gap_v14,
            "direct_anti_attenuation_alignment_v14": direct_anti_attenuation_alignment_v14,
        },
        "direct_equation_v14": {
            "origin_term": "D_origin_v14 = 0.61 * D_origin_v13 + 0.14 * R_anti_att + 0.15 * (1 - P_att) + 0.10 * R_train_v19",
            "feature_term": "D_feature_v14 = 0.59 * D_feature_v13 + 0.16 * (1 - A_learn) + 0.10 * (1 - P_att) + 0.15 * B_plastic_v19",
            "structure_term": "D_structure_v14 = 0.55 * D_structure_v13 + 0.20 * (1 - A_struct) + 0.10 * (1 - P_att) + 0.15 * B_struct_v19",
            "route_term": "D_route_v14 = 0.55 * D_route_v13 + 0.20 * (1 - A_route) + 0.10 * (1 - A_ctx) + 0.05 * (1 - P_att) + 0.10 * H_scale_v19",
            "system_term": "M_brain_direct_v14 = mean(D_origin_v14, D_feature_v14, D_structure_v14, D_route_v14)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第十四版开始直接吸收传播衰减探针，检验系统能否从衰减走向补偿。",
            "next_question": "下一步要把第十四版直测链并回训练终式和主核，检验是否终于出现补偿级传播。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第十四版报告",
        "",
        f"- direct_origin_measure_v14: {hm['direct_origin_measure_v14']:.6f}",
        f"- direct_feature_measure_v14: {hm['direct_feature_measure_v14']:.6f}",
        f"- direct_structure_measure_v14: {hm['direct_structure_measure_v14']:.6f}",
        f"- direct_route_measure_v14: {hm['direct_route_measure_v14']:.6f}",
        f"- direct_brain_measure_v14: {hm['direct_brain_measure_v14']:.6f}",
        f"- direct_brain_gap_v14: {hm['direct_brain_gap_v14']:.6f}",
        f"- direct_anti_attenuation_alignment_v14: {hm['direct_anti_attenuation_alignment_v14']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v14_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
