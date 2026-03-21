from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v11_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v11_summary() -> dict:
    v10 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v10_20260321" / "summary.json"
    )
    plateau = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_plateau_break_probe_20260321" / "summary.json"
    )
    bridge_v16 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v16_20260321" / "summary.json"
    )

    hv = v10["headline_metrics"]
    hp = plateau["headline_metrics"]
    hb = bridge_v16["headline_metrics"]

    direct_origin_measure_v11 = _clip01(
        hv["direct_origin_measure_v10"] * 0.60
        + hp["plateau_break_readiness"] * 0.15
        + (1.0 - hp["plateau_instability_penalty"]) * 0.15
        + hb["topology_training_readiness_v16"] * 0.10
    )
    direct_feature_measure_v11 = _clip01(
        hv["direct_feature_measure_v10"] * 0.58
        + hp["plateau_growth_support"] * 0.17
        + (1.0 - hp["plateau_instability_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v16"] * 0.15
    )
    direct_structure_measure_v11 = _clip01(
        hv["direct_structure_measure_v10"] * 0.54
        + hp["plateau_structure_guard"] * 0.21
        + (1.0 - hp["plateau_instability_penalty"]) * 0.10
        + hb["structure_rule_alignment_v16"] * 0.15
    )
    direct_route_measure_v11 = _clip01(
        hv["direct_route_measure_v10"] * 0.54
        + hp["plateau_route_guard"] * 0.21
        + hp["plateau_context_guard"] * 0.10
        + (1.0 - hp["plateau_instability_penalty"]) * 0.05
        + hb["coordination_guard_v16"] * 0.10
    )
    direct_brain_measure_v11 = (
        direct_origin_measure_v11
        + direct_feature_measure_v11
        + direct_structure_measure_v11
        + direct_route_measure_v11
    ) / 4.0
    direct_brain_gap_v11 = 1.0 - direct_brain_measure_v11
    direct_plateau_alignment_v11 = (
        direct_structure_measure_v11
        + direct_route_measure_v11
        + hp["plateau_break_readiness"]
        + hb["topology_training_readiness_v16"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v11": direct_origin_measure_v11,
            "direct_feature_measure_v11": direct_feature_measure_v11,
            "direct_structure_measure_v11": direct_structure_measure_v11,
            "direct_route_measure_v11": direct_route_measure_v11,
            "direct_brain_measure_v11": direct_brain_measure_v11,
            "direct_brain_gap_v11": direct_brain_gap_v11,
            "direct_plateau_alignment_v11": direct_plateau_alignment_v11,
        },
        "direct_equation_v11": {
            "origin_term": "D_origin_v11 = 0.60 * D_origin_v10 + 0.15 * R_break + 0.15 * (1 - P_break) + 0.10 * R_train_v16",
            "feature_term": "D_feature_v11 = 0.58 * D_feature_v10 + 0.17 * G_growth_break + 0.10 * (1 - P_break) + 0.15 * B_plastic_v16",
            "structure_term": "D_structure_v11 = 0.54 * D_structure_v10 + 0.21 * G_struct_break + 0.10 * (1 - P_break) + 0.15 * B_struct_v16",
            "route_term": "D_route_v11 = 0.54 * D_route_v10 + 0.21 * G_route_break + 0.10 * G_ctx_break + 0.05 * (1 - P_break) + 0.10 * H_coord_v16",
            "system_term": "M_brain_direct_v11 = mean(D_origin_v11, D_feature_v11, D_structure_v11, D_route_v11)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第十一版开始显式吸收破平台探针，开始直接检验协同护栏是否能让脑编码链重新走强。",
            "next_question": "下一步要把第十一版直测链并回训练终式和主核，检验主核是否真的出现破平台迹象。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第十一版报告",
        "",
        f"- direct_origin_measure_v11: {hm['direct_origin_measure_v11']:.6f}",
        f"- direct_feature_measure_v11: {hm['direct_feature_measure_v11']:.6f}",
        f"- direct_structure_measure_v11: {hm['direct_structure_measure_v11']:.6f}",
        f"- direct_route_measure_v11: {hm['direct_route_measure_v11']:.6f}",
        f"- direct_brain_measure_v11: {hm['direct_brain_measure_v11']:.6f}",
        f"- direct_brain_gap_v11: {hm['direct_brain_gap_v11']:.6f}",
        f"- direct_plateau_alignment_v11: {hm['direct_plateau_alignment_v11']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v11_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
