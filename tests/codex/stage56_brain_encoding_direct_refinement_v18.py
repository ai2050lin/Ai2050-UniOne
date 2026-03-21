from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v18_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v18_summary() -> dict:
    v17 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v17_20260321" / "summary.json"
    )
    reinforce = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_sustained_amplification_strengthening_20260321" / "summary.json"
    )
    bridge_v23 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v23_20260321" / "summary.json"
    )

    hv = v17["headline_metrics"]
    hr = reinforce["headline_metrics"]
    hb = bridge_v23["headline_metrics"]

    direct_origin_measure_v18 = _clip01(
        hv["direct_origin_measure_v17"] * 0.48
        + hr["amplification_reinforced_readiness"] * 0.20
        + (1.0 - hr["amplification_residual_penalty"]) * 0.15
        + hb["topology_training_readiness_v23"] * 0.17
    )
    direct_feature_measure_v18 = _clip01(
        hv["direct_feature_measure_v17"] * 0.46
        + hr["amplification_learning_lift"] * 0.24
        + (1.0 - hr["amplification_residual_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v23"] * 0.20
    )
    direct_structure_measure_v18 = _clip01(
        hv["direct_structure_measure_v17"] * 0.44
        + hr["amplification_structure_stability"] * 0.26
        + (1.0 - hr["amplification_residual_penalty"]) * 0.10
        + hb["structure_rule_alignment_v23"] * 0.20
    )
    direct_route_measure_v18 = _clip01(
        hv["direct_route_measure_v17"] * 0.44
        + hr["amplification_route_stability"] * 0.26
        + hr["amplification_structure_stability"] * 0.08
        + (1.0 - hr["amplification_residual_penalty"]) * 0.05
        + hb["amplification_guard_v23"] * 0.17
    )
    direct_brain_measure_v18 = (
        direct_origin_measure_v18
        + direct_feature_measure_v18
        + direct_structure_measure_v18
        + direct_route_measure_v18
    ) / 4.0
    direct_brain_gap_v18 = 1.0 - direct_brain_measure_v18
    direct_reinforced_alignment_v18 = (
        direct_structure_measure_v18
        + direct_route_measure_v18
        + hr["amplification_reinforced_readiness"]
        + hb["topology_training_readiness_v23"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v18": direct_origin_measure_v18,
            "direct_feature_measure_v18": direct_feature_measure_v18,
            "direct_structure_measure_v18": direct_structure_measure_v18,
            "direct_route_measure_v18": direct_route_measure_v18,
            "direct_brain_measure_v18": direct_brain_measure_v18,
            "direct_brain_gap_v18": direct_brain_gap_v18,
            "direct_reinforced_alignment_v18": direct_reinforced_alignment_v18,
        },
        "direct_equation_v18": {
            "origin_term": "D_origin_v18 = 0.48 * D_origin_v17 + 0.20 * R_reinforce + 0.15 * (1 - P_reinforce) + 0.17 * R_train_v23",
            "feature_term": "D_feature_v18 = 0.46 * D_feature_v17 + 0.24 * L_reinforce + 0.10 * (1 - P_reinforce) + 0.20 * B_plastic_v23",
            "structure_term": "D_structure_v18 = 0.44 * D_structure_v17 + 0.26 * S_reinforce + 0.10 * (1 - P_reinforce) + 0.20 * B_struct_v23",
            "route_term": "D_route_v18 = 0.44 * D_route_v17 + 0.26 * R_reinforce + 0.08 * S_reinforce + 0.05 * (1 - P_reinforce) + 0.17 * H_amp_v23",
            "system_term": "M_brain_direct_v18 = mean(D_origin_v18, D_feature_v18, D_structure_v18, D_route_v18)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第十八版开始把持续放大强化并回脑编码链，检查放大趋势是否开始稳态化。",
            "next_question": "下一步要把第十八版直测链并回训练终式和主核，确认放大是否开始从轻度放大转向稳态放大。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第十八版报告",
        "",
        f"- direct_origin_measure_v18: {hm['direct_origin_measure_v18']:.6f}",
        f"- direct_feature_measure_v18: {hm['direct_feature_measure_v18']:.6f}",
        f"- direct_structure_measure_v18: {hm['direct_structure_measure_v18']:.6f}",
        f"- direct_route_measure_v18: {hm['direct_route_measure_v18']:.6f}",
        f"- direct_brain_measure_v18: {hm['direct_brain_measure_v18']:.6f}",
        f"- direct_brain_gap_v18: {hm['direct_brain_gap_v18']:.6f}",
        f"- direct_reinforced_alignment_v18: {hm['direct_reinforced_alignment_v18']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v18_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
