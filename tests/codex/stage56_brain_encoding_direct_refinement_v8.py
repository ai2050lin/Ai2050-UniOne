from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v8_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v8_summary() -> dict:
    v7 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v7_20260321" / "summary.json"
    )
    coupled = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_route_structure_coupled_validation_20260321" / "summary.json"
    )
    bridge_v13 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v13_20260321" / "summary.json"
    )

    hv = v7["headline_metrics"]
    hc = coupled["headline_metrics"]
    hb = bridge_v13["headline_metrics"]

    direct_origin_measure_v8 = _clip01(
        hv["direct_origin_measure_v7"] * 0.60
        + hc["coupled_readiness"] * 0.15
        + (1.0 - hc["coupled_forgetting_penalty"]) * 0.15
        + hb["topology_training_readiness_v13"] * 0.10
    )
    direct_feature_measure_v8 = _clip01(
        hv["direct_feature_measure_v7"] * 0.60
        + hc["coupled_novel_gain"] * 0.15
        + (1.0 - hc["coupled_forgetting_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v13"] * 0.15
    )
    direct_structure_measure_v8 = _clip01(
        hv["direct_structure_measure_v7"] * 0.55
        + hc["coupled_structure_keep"] * 0.20
        + (1.0 - hc["coupled_failure_risk"]) * 0.10
        + hb["structure_rule_alignment_v13"] * 0.15
    )
    direct_route_measure_v8 = _clip01(
        hv["direct_route_measure_v7"] * 0.55
        + hc["coupled_route_keep"] * 0.20
        + hc["coupled_context_keep"] * 0.10
        + (1.0 - hc["coupled_failure_risk"]) * 0.05
        + hb["route_guard_v13"] * 0.10
    )
    direct_brain_measure_v8 = (
        direct_origin_measure_v8
        + direct_feature_measure_v8
        + direct_structure_measure_v8
        + direct_route_measure_v8
    ) / 4.0
    direct_brain_gap_v8 = 1.0 - direct_brain_measure_v8
    direct_coupled_alignment_v8 = (
        direct_structure_measure_v8
        + direct_route_measure_v8
        + hc["coupled_readiness"]
        + hb["topology_training_readiness_v13"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v8": direct_origin_measure_v8,
            "direct_feature_measure_v8": direct_feature_measure_v8,
            "direct_structure_measure_v8": direct_structure_measure_v8,
            "direct_route_measure_v8": direct_route_measure_v8,
            "direct_brain_measure_v8": direct_brain_measure_v8,
            "direct_brain_gap_v8": direct_brain_gap_v8,
            "direct_coupled_alignment_v8": direct_coupled_alignment_v8,
        },
        "direct_equation_v8": {
            "origin_term": "D_origin_v8 = 0.60 * D_origin_v7 + 0.15 * A_coupled + 0.15 * (1 - P_coupled) + 0.10 * R_train_v13",
            "feature_term": "D_feature_v8 = 0.60 * D_feature_v7 + 0.15 * G_coupled + 0.10 * (1 - P_coupled) + 0.15 * B_plastic_v13",
            "structure_term": "D_structure_v8 = 0.55 * D_structure_v7 + 0.20 * K_struct + 0.10 * (1 - R_fail) + 0.15 * B_struct_v13",
            "route_term": "D_route_v8 = 0.55 * D_route_v7 + 0.20 * K_route + 0.10 * K_ctx + 0.05 * (1 - R_fail) + 0.10 * H_route_v13",
            "system_term": "M_brain_direct_v8 = mean(D_origin_v8, D_feature_v8, D_structure_v8, D_route_v8)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第八版开始显式吸收路由-结构联动退化压力，使脑编码直测链不只面对单项风险，而是开始面对联动退化链。",
            "next_question": "下一步要把第八版直测链并回训练终式和主核，检验主核在联动退化压力下是否还能继续收口。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第八版报告",
        "",
        f"- direct_origin_measure_v8: {hm['direct_origin_measure_v8']:.6f}",
        f"- direct_feature_measure_v8: {hm['direct_feature_measure_v8']:.6f}",
        f"- direct_structure_measure_v8: {hm['direct_structure_measure_v8']:.6f}",
        f"- direct_route_measure_v8: {hm['direct_route_measure_v8']:.6f}",
        f"- direct_brain_measure_v8: {hm['direct_brain_measure_v8']:.6f}",
        f"- direct_brain_gap_v8: {hm['direct_brain_gap_v8']:.6f}",
        f"- direct_coupled_alignment_v8: {hm['direct_coupled_alignment_v8']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v8_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
