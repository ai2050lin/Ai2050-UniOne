from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v16_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v16_summary() -> dict:
    v15 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v15_20260321" / "summary.json"
    )
    sustained = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_sustained_rebound_validation_20260321" / "summary.json"
    )
    bridge_v21 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v21_20260321" / "summary.json"
    )

    hv = v15["headline_metrics"]
    hs = sustained["headline_metrics"]
    hb = bridge_v21["headline_metrics"]

    direct_origin_measure_v16 = _clip01(
        hv["direct_origin_measure_v15"] * 0.52
        + hs["sustained_readiness"] * 0.18
        + (1.0 - hs["sustained_penalty"]) * 0.15
        + hb["topology_training_readiness_v21"] * 0.15
    )
    direct_feature_measure_v16 = _clip01(
        hv["direct_feature_measure_v15"] * 0.50
        + hs["sustained_learning"] * 0.20
        + (1.0 - hs["sustained_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v21"] * 0.20
    )
    direct_structure_measure_v16 = _clip01(
        hv["direct_structure_measure_v15"] * 0.48
        + hs["sustained_structure"] * 0.22
        + (1.0 - hs["sustained_penalty"]) * 0.10
        + hb["structure_rule_alignment_v21"] * 0.20
    )
    direct_route_measure_v16 = _clip01(
        hv["direct_route_measure_v15"] * 0.48
        + hs["sustained_route"] * 0.22
        + hs["sustained_context"] * 0.10
        + (1.0 - hs["sustained_penalty"]) * 0.05
        + hb["persistence_guard_v21"] * 0.15
    )
    direct_brain_measure_v16 = (
        direct_origin_measure_v16
        + direct_feature_measure_v16
        + direct_structure_measure_v16
        + direct_route_measure_v16
    ) / 4.0
    direct_brain_gap_v16 = 1.0 - direct_brain_measure_v16
    direct_sustained_alignment_v16 = (
        direct_structure_measure_v16
        + direct_route_measure_v16
        + hs["sustained_readiness"]
        + hb["topology_training_readiness_v21"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v16": direct_origin_measure_v16,
            "direct_feature_measure_v16": direct_feature_measure_v16,
            "direct_structure_measure_v16": direct_structure_measure_v16,
            "direct_route_measure_v16": direct_route_measure_v16,
            "direct_brain_measure_v16": direct_brain_measure_v16,
            "direct_brain_gap_v16": direct_brain_gap_v16,
            "direct_sustained_alignment_v16": direct_sustained_alignment_v16,
        },
        "direct_equation_v16": {
            "origin_term": "D_origin_v16 = 0.52 * D_origin_v15 + 0.18 * R_sustain + 0.15 * (1 - P_sustain) + 0.15 * R_train_v21",
            "feature_term": "D_feature_v16 = 0.50 * D_feature_v15 + 0.20 * L_sustain + 0.10 * (1 - P_sustain) + 0.20 * B_plastic_v21",
            "structure_term": "D_structure_v16 = 0.48 * D_structure_v15 + 0.22 * S_sustain + 0.10 * (1 - P_sustain) + 0.20 * B_struct_v21",
            "route_term": "D_route_v16 = 0.48 * D_route_v15 + 0.22 * R_sustain + 0.10 * C_sustain + 0.05 * (1 - P_sustain) + 0.15 * H_persist_v21",
            "system_term": "M_brain_direct_v16 = mean(D_origin_v16, D_feature_v16, D_structure_v16, D_route_v16)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第十六版开始把持续回升验证并回脑编码链，检查补偿是否开始变成更稳定的脑编码层增强。",
            "next_question": "下一步要把第十六版直测链并回训练终式和主核，确认持续回升是否已经从局部现象推进到跨层稳定现象。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第十六版报告",
        "",
        f"- direct_origin_measure_v16: {hm['direct_origin_measure_v16']:.6f}",
        f"- direct_feature_measure_v16: {hm['direct_feature_measure_v16']:.6f}",
        f"- direct_structure_measure_v16: {hm['direct_structure_measure_v16']:.6f}",
        f"- direct_route_measure_v16: {hm['direct_route_measure_v16']:.6f}",
        f"- direct_brain_measure_v16: {hm['direct_brain_measure_v16']:.6f}",
        f"- direct_brain_gap_v16: {hm['direct_brain_gap_v16']:.6f}",
        f"- direct_sustained_alignment_v16: {hm['direct_sustained_alignment_v16']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v16_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
