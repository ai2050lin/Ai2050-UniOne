from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v7_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v7_summary() -> dict:
    v6 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v6_20260321" / "summary.json"
    )
    route_probe = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_route_degradation_probe_20260321" / "summary.json"
    )
    bridge_v12 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v12_20260321" / "summary.json"
    )
    true_scale = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_online_collapse_probe_20260321" / "summary.json"
    )

    hv = v6["headline_metrics"]
    hr = route_probe["headline_metrics"]
    hb = bridge_v12["headline_metrics"]
    ht = true_scale["headline_metrics"]

    direct_origin_measure_v7 = _clip01(
        hv["direct_origin_measure_v6"] * 0.55
        + ht["true_scale_language_keep"] * 0.20
        + (1.0 - ht["true_scale_forgetting_penalty"]) * 0.15
        + hr["true_scale_reinforced_readiness"] * 0.10
    )
    direct_feature_measure_v7 = _clip01(
        hv["direct_feature_measure_v6"] * 0.55
        + ht["true_scale_novel_gain"] * 0.10
        + hr["route_resilience"] * 0.10
        + (1.0 - ht["true_scale_forgetting_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v12"] * 0.15
    )
    direct_structure_measure_v7 = _clip01(
        hv["direct_structure_measure_v6"] * 0.55
        + hr["structure_resilience"] * 0.20
        + (1.0 - hr["structure_phase_shift_risk"]) * 0.10
        + hb["structure_rule_alignment_v12"] * 0.15
    )
    direct_route_measure_v7 = _clip01(
        hv["direct_route_measure_v6"] * 0.50
        + hr["route_resilience"] * 0.20
        + (1.0 - hr["route_degradation_risk"]) * 0.10
        + hb["true_scale_guard_v12"] * 0.10
        + ht["true_scale_context_keep"] * 0.10
    )
    direct_brain_measure_v7 = (
        direct_origin_measure_v7
        + direct_feature_measure_v7
        + direct_structure_measure_v7
        + direct_route_measure_v7
    ) / 4.0
    direct_brain_gap_v7 = 1.0 - direct_brain_measure_v7
    direct_route_alignment_v7 = (
        direct_structure_measure_v7
        + direct_route_measure_v7
        + hr["true_scale_reinforced_readiness"]
        + hb["topology_training_readiness_v12"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v7": direct_origin_measure_v7,
            "direct_feature_measure_v7": direct_feature_measure_v7,
            "direct_structure_measure_v7": direct_structure_measure_v7,
            "direct_route_measure_v7": direct_route_measure_v7,
            "direct_brain_measure_v7": direct_brain_measure_v7,
            "direct_brain_gap_v7": direct_brain_gap_v7,
            "direct_route_alignment_v7": direct_route_alignment_v7,
        },
        "direct_equation_v7": {
            "origin_term": "D_origin_v7 = 0.55 * D_origin_v6 + 0.20 * L_true + 0.15 * (1 - P_true) + 0.10 * A_route",
            "feature_term": "D_feature_v7 = 0.55 * D_feature_v6 + 0.10 * G_true + 0.10 * H_route + 0.10 * (1 - P_true) + 0.15 * B_plastic_v12",
            "structure_term": "D_structure_v7 = 0.55 * D_structure_v6 + 0.20 * H_struct + 0.10 * (1 - R_phase) + 0.15 * B_struct_v12",
            "route_term": "D_route_v7 = 0.50 * D_route_v6 + 0.20 * H_route + 0.10 * (1 - R_route) + 0.10 * H_true_v12 + 0.10 * C_true",
            "system_term": "M_brain_direct_v7 = mean(D_origin_v7, D_feature_v7, D_structure_v7, D_route_v7)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第七版开始显式吸收真正规模化路由退化探针，使脑编码直测链不只面对结构压力，也开始面对路由退化和相变式失稳。",
            "next_question": "下一步要把第七版直测链并回训练终式和主核，检验主核在更真实路由退化压力下是否还能继续收口。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第七版报告",
        "",
        f"- direct_origin_measure_v7: {hm['direct_origin_measure_v7']:.6f}",
        f"- direct_feature_measure_v7: {hm['direct_feature_measure_v7']:.6f}",
        f"- direct_structure_measure_v7: {hm['direct_structure_measure_v7']:.6f}",
        f"- direct_route_measure_v7: {hm['direct_route_measure_v7']:.6f}",
        f"- direct_brain_measure_v7: {hm['direct_brain_measure_v7']:.6f}",
        f"- direct_brain_gap_v7: {hm['direct_brain_gap_v7']:.6f}",
        f"- direct_route_alignment_v7: {hm['direct_route_alignment_v7']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v7_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
