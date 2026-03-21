from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v22_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v22_summary() -> dict:
    v21 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v21_20260321" / "summary.json"
    )
    stable_plus = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_stable_amplification_strengthening_20260321" / "summary.json"
    )
    bridge_v27 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v27_20260321" / "summary.json"
    )

    hv = v21["headline_metrics"]
    hs = stable_plus["headline_metrics"]
    hb = bridge_v27["headline_metrics"]

    direct_origin_measure_v22 = _clip01(
        hv["direct_origin_measure_v21"] * 0.46
        + hs["stable_reinforced_readiness"] * 0.22
        + (1.0 - hs["stable_reinforced_penalty"]) * 0.15
        + hb["topology_training_readiness_v27"] * 0.17
    )
    direct_feature_measure_v22 = _clip01(
        hv["direct_feature_measure_v21"] * 0.44
        + hs["stable_reinforced_learning"] * 0.26
        + (1.0 - hs["stable_reinforced_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v27"] * 0.20
    )
    direct_structure_measure_v22 = _clip01(
        hv["direct_structure_measure_v21"] * 0.42
        + hs["stable_reinforced_structure"] * 0.28
        + (1.0 - hs["stable_reinforced_penalty"]) * 0.10
        + hb["structure_rule_alignment_v27"] * 0.20
    )
    direct_route_measure_v22 = _clip01(
        hv["direct_route_measure_v21"] * 0.42
        + hs["stable_reinforced_route"] * 0.28
        + hs["stable_reinforced_structure"] * 0.08
        + (1.0 - hs["stable_reinforced_penalty"]) * 0.05
        + hb["stable_guard_v27"] * 0.17
    )
    direct_brain_measure_v22 = (
        direct_origin_measure_v22
        + direct_feature_measure_v22
        + direct_structure_measure_v22
        + direct_route_measure_v22
    ) / 4.0
    direct_brain_gap_v22 = 1.0 - direct_brain_measure_v22
    direct_stable_alignment_v22 = (
        direct_structure_measure_v22
        + direct_route_measure_v22
        + hs["stable_reinforced_readiness"]
        + hb["topology_training_readiness_v27"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v22": direct_origin_measure_v22,
            "direct_feature_measure_v22": direct_feature_measure_v22,
            "direct_structure_measure_v22": direct_structure_measure_v22,
            "direct_route_measure_v22": direct_route_measure_v22,
            "direct_brain_measure_v22": direct_brain_measure_v22,
            "direct_brain_gap_v22": direct_brain_gap_v22,
            "direct_stable_alignment_v22": direct_stable_alignment_v22,
        },
        "direct_equation_v22": {
            "origin_term": "D_origin_v22 = 0.46 * D_origin_v21 + 0.22 * R_stable_plus + 0.15 * (1 - P_stable_plus) + 0.17 * R_train_v27",
            "feature_term": "D_feature_v22 = 0.44 * D_feature_v21 + 0.26 * L_stable_plus + 0.10 * (1 - P_stable_plus) + 0.20 * B_plastic_v27",
            "structure_term": "D_structure_v22 = 0.42 * D_structure_v21 + 0.28 * S_stable_plus + 0.10 * (1 - P_stable_plus) + 0.20 * B_struct_v27",
            "route_term": "D_route_v22 = 0.42 * D_route_v21 + 0.28 * R_stable_plus + 0.08 * S_stable_plus + 0.05 * (1 - P_stable_plus) + 0.17 * H_stable_v27",
            "system_term": "M_brain_direct_v22 = mean(D_origin_v22, D_feature_v22, D_structure_v22, D_route_v22)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第二十二版开始把稳定放大强化并回脑编码链，检查放大趋势是否继续转成更稳的脑编码承接。",
            "next_question": "下一步要把第二十二版直测链并回训练终式和主核，确认稳定放大是否继续在脑编码层增强。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第二十二版报告",
        "",
        f"- direct_origin_measure_v22: {hm['direct_origin_measure_v22']:.6f}",
        f"- direct_feature_measure_v22: {hm['direct_feature_measure_v22']:.6f}",
        f"- direct_structure_measure_v22: {hm['direct_structure_measure_v22']:.6f}",
        f"- direct_route_measure_v22: {hm['direct_route_measure_v22']:.6f}",
        f"- direct_brain_measure_v22: {hm['direct_brain_measure_v22']:.6f}",
        f"- direct_brain_gap_v22: {hm['direct_brain_gap_v22']:.6f}",
        f"- direct_stable_alignment_v22: {hm['direct_stable_alignment_v22']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v22_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
