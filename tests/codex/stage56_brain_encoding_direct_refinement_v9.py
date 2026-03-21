from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v9_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v9_summary() -> dict:
    v8 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v8_20260321" / "summary.json"
    )
    mega = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_coupled_degradation_validation_20260321" / "summary.json"
    )
    bridge_v14 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v14_20260321" / "summary.json"
    )

    hv = v8["headline_metrics"]
    hm = mega["headline_metrics"]
    hb = bridge_v14["headline_metrics"]

    direct_origin_measure_v9 = _clip01(
        hv["direct_origin_measure_v8"] * 0.62
        + hm["mega_coupled_readiness"] * 0.18
        + (1.0 - hm["mega_coupled_forgetting_penalty"]) * 0.10
        + hb["topology_training_readiness_v14"] * 0.10
    )
    direct_feature_measure_v9 = _clip01(
        hv["direct_feature_measure_v8"] * 0.58
        + hm["mega_coupled_novel_gain"] * 0.17
        + (1.0 - hm["mega_coupled_forgetting_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v14"] * 0.15
    )
    direct_structure_measure_v9 = _clip01(
        hv["direct_structure_measure_v8"] * 0.55
        + hm["mega_coupled_structure_keep"] * 0.20
        + (1.0 - hm["mega_coupled_collapse_risk"]) * 0.10
        + hb["structure_rule_alignment_v14"] * 0.15
    )
    direct_route_measure_v9 = _clip01(
        hv["direct_route_measure_v8"] * 0.55
        + (1.0 - hm["mega_coupled_route_degradation"]) * 0.20
        + hm["mega_coupled_context_keep"] * 0.10
        + (1.0 - hm["mega_coupled_collapse_risk"]) * 0.05
        + hb["coupled_guard_v14"] * 0.10
    )
    direct_brain_measure_v9 = (
        direct_origin_measure_v9
        + direct_feature_measure_v9
        + direct_structure_measure_v9
        + direct_route_measure_v9
    ) / 4.0
    direct_brain_gap_v9 = 1.0 - direct_brain_measure_v9
    direct_mega_alignment_v9 = (
        direct_structure_measure_v9
        + direct_route_measure_v9
        + hm["mega_coupled_readiness"]
        + hb["topology_training_readiness_v14"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v9": direct_origin_measure_v9,
            "direct_feature_measure_v9": direct_feature_measure_v9,
            "direct_structure_measure_v9": direct_structure_measure_v9,
            "direct_route_measure_v9": direct_route_measure_v9,
            "direct_brain_measure_v9": direct_brain_measure_v9,
            "direct_brain_gap_v9": direct_brain_gap_v9,
            "direct_mega_alignment_v9": direct_mega_alignment_v9,
        },
        "direct_equation_v9": {
            "origin_term": "D_origin_v9 = 0.62 * D_origin_v8 + 0.18 * R_mega + 0.10 * (1 - P_mega) + 0.10 * R_train_v14",
            "feature_term": "D_feature_v9 = 0.58 * D_feature_v8 + 0.17 * G_mega + 0.10 * (1 - P_mega) + 0.15 * B_plastic_v14",
            "structure_term": "D_structure_v9 = 0.55 * D_structure_v8 + 0.20 * S_mega + 0.10 * (1 - R_collapse_mega) + 0.15 * B_struct_v14",
            "route_term": "D_route_v9 = 0.55 * D_route_v8 + 0.20 * (1 - R_route_mega) + 0.10 * C_mega + 0.05 * (1 - R_collapse_mega) + 0.10 * H_coupled_v14",
            "system_term": "M_brain_direct_v9 = mean(D_origin_v9, D_feature_v9, D_structure_v9, D_route_v9)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第九版开始显式吸收更大系统联动退化压力，使脑编码直测链面对的不是局部风险，而是更接近真实规模化系统里的耦合退化。",
            "next_question": "下一步要把第九版直测链并回训练终式和主核，检验主核在更大系统联动退化压力下是否还能继续收口。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第九版报告",
        "",
        f"- direct_origin_measure_v9: {hm['direct_origin_measure_v9']:.6f}",
        f"- direct_feature_measure_v9: {hm['direct_feature_measure_v9']:.6f}",
        f"- direct_structure_measure_v9: {hm['direct_structure_measure_v9']:.6f}",
        f"- direct_route_measure_v9: {hm['direct_route_measure_v9']:.6f}",
        f"- direct_brain_measure_v9: {hm['direct_brain_measure_v9']:.6f}",
        f"- direct_brain_gap_v9: {hm['direct_brain_gap_v9']:.6f}",
        f"- direct_mega_alignment_v9: {hm['direct_mega_alignment_v9']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v9_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
