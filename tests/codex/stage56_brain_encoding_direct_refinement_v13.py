from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v13_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v13_summary() -> dict:
    v12 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v12_20260321" / "summary.json"
    )
    scale_prop = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_propagation_validation_20260321" / "summary.json"
    )
    bridge_v18 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v18_20260321" / "summary.json"
    )

    hv = v12["headline_metrics"]
    hs = scale_prop["headline_metrics"]
    hb = bridge_v18["headline_metrics"]

    direct_origin_measure_v13 = _clip01(
        hv["direct_origin_measure_v12"] * 0.60
        + hs["scale_propagation_readiness"] * 0.15
        + (1.0 - hs["scale_propagation_penalty"]) * 0.15
        + hb["topology_training_readiness_v18"] * 0.10
    )
    direct_feature_measure_v13 = _clip01(
        hv["direct_feature_measure_v12"] * 0.58
        + hs["scale_propagation_learning"] * 0.17
        + (1.0 - hs["scale_propagation_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v18"] * 0.15
    )
    direct_structure_measure_v13 = _clip01(
        hv["direct_structure_measure_v12"] * 0.54
        + hs["scale_propagation_structure"] * 0.21
        + (1.0 - hs["scale_propagation_penalty"]) * 0.10
        + hb["structure_rule_alignment_v18"] * 0.15
    )
    direct_route_measure_v13 = _clip01(
        hv["direct_route_measure_v12"] * 0.54
        + hs["scale_propagation_route"] * 0.21
        + hs["scale_propagation_context"] * 0.10
        + (1.0 - hs["scale_propagation_penalty"]) * 0.05
        + hb["propagation_guard_v18"] * 0.10
    )
    direct_brain_measure_v13 = (
        direct_origin_measure_v13
        + direct_feature_measure_v13
        + direct_structure_measure_v13
        + direct_route_measure_v13
    ) / 4.0
    direct_brain_gap_v13 = 1.0 - direct_brain_measure_v13
    direct_scale_alignment_v13 = (
        direct_structure_measure_v13
        + direct_route_measure_v13
        + hs["scale_propagation_readiness"]
        + hb["topology_training_readiness_v18"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v13": direct_origin_measure_v13,
            "direct_feature_measure_v13": direct_feature_measure_v13,
            "direct_structure_measure_v13": direct_structure_measure_v13,
            "direct_route_measure_v13": direct_route_measure_v13,
            "direct_brain_measure_v13": direct_brain_measure_v13,
            "direct_brain_gap_v13": direct_brain_gap_v13,
            "direct_scale_alignment_v13": direct_scale_alignment_v13,
        },
        "direct_equation_v13": {
            "origin_term": "D_origin_v13 = 0.60 * D_origin_v12 + 0.15 * R_prop_scale + 0.15 * (1 - P_prop_scale) + 0.10 * R_train_v18",
            "feature_term": "D_feature_v13 = 0.58 * D_feature_v12 + 0.17 * L_prop_scale + 0.10 * (1 - P_prop_scale) + 0.15 * B_plastic_v18",
            "structure_term": "D_structure_v13 = 0.54 * D_structure_v12 + 0.21 * S_prop_scale + 0.10 * (1 - P_prop_scale) + 0.15 * B_struct_v18",
            "route_term": "D_route_v13 = 0.54 * D_route_v12 + 0.21 * R_prop_scale + 0.10 * C_prop_scale + 0.05 * (1 - P_prop_scale) + 0.10 * H_prop_v18",
            "system_term": "M_brain_direct_v13 = mean(D_origin_v13, D_feature_v13, D_structure_v13, D_route_v13)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第十三版开始直接吸收更大系统传播结果，检验平台期松动能否终于穿透到脑编码链。",
            "next_question": "下一步要把第十三版直测链并回训练终式和主核，检验传播是否开始形成跨层级突破。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第十三版报告",
        "",
        f"- direct_origin_measure_v13: {hm['direct_origin_measure_v13']:.6f}",
        f"- direct_feature_measure_v13: {hm['direct_feature_measure_v13']:.6f}",
        f"- direct_structure_measure_v13: {hm['direct_structure_measure_v13']:.6f}",
        f"- direct_route_measure_v13: {hm['direct_route_measure_v13']:.6f}",
        f"- direct_brain_measure_v13: {hm['direct_brain_measure_v13']:.6f}",
        f"- direct_brain_gap_v13: {hm['direct_brain_gap_v13']:.6f}",
        f"- direct_scale_alignment_v13: {hm['direct_scale_alignment_v13']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v13_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
