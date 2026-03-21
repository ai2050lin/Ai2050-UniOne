from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v10_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v10_summary() -> dict:
    v9 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v9_20260321" / "summary.json"
    )
    coord = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_coordination_stabilization_20260321" / "summary.json"
    )
    bridge_v15 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v15_20260321" / "summary.json"
    )

    hv = v9["headline_metrics"]
    hc = coord["headline_metrics"]
    hb = bridge_v15["headline_metrics"]

    direct_origin_measure_v10 = _clip01(
        hv["direct_origin_measure_v9"] * 0.60
        + hc["coordinated_readiness"] * 0.15
        + (1.0 - hc["coordinated_instability_penalty"]) * 0.15
        + hb["topology_training_readiness_v15"] * 0.10
    )
    direct_feature_measure_v10 = _clip01(
        hv["direct_feature_measure_v9"] * 0.58
        + hc["coordinated_growth_support"] * 0.17
        + (1.0 - hc["coordinated_instability_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v15"] * 0.15
    )
    direct_structure_measure_v10 = _clip01(
        hv["direct_structure_measure_v9"] * 0.54
        + hc["coordinated_structure_guard"] * 0.21
        + (1.0 - hc["coordinated_instability_penalty"]) * 0.10
        + hb["structure_rule_alignment_v15"] * 0.15
    )
    direct_route_measure_v10 = _clip01(
        hv["direct_route_measure_v9"] * 0.54
        + hc["coordinated_route_guard"] * 0.21
        + hc["coordinated_context_guard"] * 0.10
        + (1.0 - hc["coordinated_instability_penalty"]) * 0.05
        + hb["mega_guard_v15"] * 0.10
    )
    direct_brain_measure_v10 = (
        direct_origin_measure_v10
        + direct_feature_measure_v10
        + direct_structure_measure_v10
        + direct_route_measure_v10
    ) / 4.0
    direct_brain_gap_v10 = 1.0 - direct_brain_measure_v10
    direct_coord_alignment_v10 = (
        direct_structure_measure_v10
        + direct_route_measure_v10
        + hc["coordinated_readiness"]
        + hb["topology_training_readiness_v15"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v10": direct_origin_measure_v10,
            "direct_feature_measure_v10": direct_feature_measure_v10,
            "direct_structure_measure_v10": direct_structure_measure_v10,
            "direct_route_measure_v10": direct_route_measure_v10,
            "direct_brain_measure_v10": direct_brain_measure_v10,
            "direct_brain_gap_v10": direct_brain_gap_v10,
            "direct_coord_alignment_v10": direct_coord_alignment_v10,
        },
        "direct_equation_v10": {
            "origin_term": "D_origin_v10 = 0.60 * D_origin_v9 + 0.15 * R_coord + 0.15 * (1 - P_coord) + 0.10 * R_train_v15",
            "feature_term": "D_feature_v10 = 0.58 * D_feature_v9 + 0.17 * G_growth + 0.10 * (1 - P_coord) + 0.15 * B_plastic_v15",
            "structure_term": "D_structure_v10 = 0.54 * D_structure_v9 + 0.21 * G_struct + 0.10 * (1 - P_coord) + 0.15 * B_struct_v15",
            "route_term": "D_route_v10 = 0.54 * D_route_v9 + 0.21 * G_route + 0.10 * G_ctx + 0.05 * (1 - P_coord) + 0.10 * H_mega_v15",
            "system_term": "M_brain_direct_v10 = mean(D_origin_v10, D_feature_v10, D_structure_v10, D_route_v10)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第十版开始吸收协同稳定化护栏，主线开始从单项补强转向结构、上下文和路由的协同补强。",
            "next_question": "下一步要把第十版直测链并回训练终式和主核，检验主核是否能突破当前的平台期。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第十版报告",
        "",
        f"- direct_origin_measure_v10: {hm['direct_origin_measure_v10']:.6f}",
        f"- direct_feature_measure_v10: {hm['direct_feature_measure_v10']:.6f}",
        f"- direct_structure_measure_v10: {hm['direct_structure_measure_v10']:.6f}",
        f"- direct_route_measure_v10: {hm['direct_route_measure_v10']:.6f}",
        f"- direct_brain_measure_v10: {hm['direct_brain_measure_v10']:.6f}",
        f"- direct_brain_gap_v10: {hm['direct_brain_gap_v10']:.6f}",
        f"- direct_coord_alignment_v10: {hm['direct_coord_alignment_v10']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v10_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
