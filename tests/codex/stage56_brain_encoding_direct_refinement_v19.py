from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v19_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v19_summary() -> dict:
    v18 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v18_20260321" / "summary.json"
    )
    steady = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_steady_amplification_validation_20260321" / "summary.json"
    )
    bridge_v24 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v24_20260321" / "summary.json"
    )

    hv = v18["headline_metrics"]
    hs = steady["headline_metrics"]
    hb = bridge_v24["headline_metrics"]

    direct_origin_measure_v19 = _clip01(
        hv["direct_origin_measure_v18"] * 0.46
        + hs["steady_readiness"] * 0.22
        + (1.0 - hs["steady_residual_penalty"]) * 0.15
        + hb["topology_training_readiness_v24"] * 0.17
    )
    direct_feature_measure_v19 = _clip01(
        hv["direct_feature_measure_v18"] * 0.44
        + hs["steady_learning_lift"] * 0.26
        + (1.0 - hs["steady_residual_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v24"] * 0.20
    )
    direct_structure_measure_v19 = _clip01(
        hv["direct_structure_measure_v18"] * 0.42
        + hs["steady_structure_stability"] * 0.28
        + (1.0 - hs["steady_residual_penalty"]) * 0.10
        + hb["structure_rule_alignment_v24"] * 0.20
    )
    direct_route_measure_v19 = _clip01(
        hv["direct_route_measure_v18"] * 0.42
        + hs["steady_route_stability"] * 0.28
        + hs["steady_structure_stability"] * 0.08
        + (1.0 - hs["steady_residual_penalty"]) * 0.05
        + hb["reinforcement_guard_v24"] * 0.17
    )
    direct_brain_measure_v19 = (
        direct_origin_measure_v19
        + direct_feature_measure_v19
        + direct_structure_measure_v19
        + direct_route_measure_v19
    ) / 4.0
    direct_brain_gap_v19 = 1.0 - direct_brain_measure_v19
    direct_steady_alignment_v19 = (
        direct_structure_measure_v19
        + direct_route_measure_v19
        + hs["steady_readiness"]
        + hb["topology_training_readiness_v24"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v19": direct_origin_measure_v19,
            "direct_feature_measure_v19": direct_feature_measure_v19,
            "direct_structure_measure_v19": direct_structure_measure_v19,
            "direct_route_measure_v19": direct_route_measure_v19,
            "direct_brain_measure_v19": direct_brain_measure_v19,
            "direct_brain_gap_v19": direct_brain_gap_v19,
            "direct_steady_alignment_v19": direct_steady_alignment_v19,
        },
        "direct_equation_v19": {
            "origin_term": "D_origin_v19 = 0.46 * D_origin_v18 + 0.22 * R_steady + 0.15 * (1 - P_steady) + 0.17 * R_train_v24",
            "feature_term": "D_feature_v19 = 0.44 * D_feature_v18 + 0.26 * L_steady + 0.10 * (1 - P_steady) + 0.20 * B_plastic_v24",
            "structure_term": "D_structure_v19 = 0.42 * D_structure_v18 + 0.28 * S_steady + 0.10 * (1 - P_steady) + 0.20 * B_struct_v24",
            "route_term": "D_route_v19 = 0.42 * D_route_v18 + 0.28 * R_steady + 0.08 * S_steady + 0.05 * (1 - P_steady) + 0.17 * H_reinforce_v24",
            "system_term": "M_brain_direct_v19 = mean(D_origin_v19, D_feature_v19, D_structure_v19, D_route_v19)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第十九版开始把稳态放大验证并回脑编码链，检查放大趋势是否开始稳态化落在脑编码层。",
            "next_question": "下一步要把第十九版直测链并回训练终式和主核，确认稳态放大是否已经开始真正成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第十九版报告",
        "",
        f"- direct_origin_measure_v19: {hm['direct_origin_measure_v19']:.6f}",
        f"- direct_feature_measure_v19: {hm['direct_feature_measure_v19']:.6f}",
        f"- direct_structure_measure_v19: {hm['direct_structure_measure_v19']:.6f}",
        f"- direct_route_measure_v19: {hm['direct_route_measure_v19']:.6f}",
        f"- direct_brain_measure_v19: {hm['direct_brain_measure_v19']:.6f}",
        f"- direct_brain_gap_v19: {hm['direct_brain_gap_v19']:.6f}",
        f"- direct_steady_alignment_v19: {hm['direct_steady_alignment_v19']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v19_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
