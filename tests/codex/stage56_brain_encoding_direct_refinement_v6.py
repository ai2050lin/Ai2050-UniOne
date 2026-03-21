from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v6_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v6_summary() -> dict:
    v5 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v5_20260321" / "summary.json"
    )
    true_scale = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_online_collapse_probe_20260321" / "summary.json"
    )
    bridge_v11 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v11_20260321" / "summary.json"
    )
    topo_train = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_trainable_prototype_20260321" / "summary.json"
    )

    hv = v5["headline_metrics"]
    ht = true_scale["headline_metrics"]
    hb = bridge_v11["headline_metrics"]
    hp = topo_train["headline_metrics"]

    direct_origin_measure_v6 = _clip01(
        hv["direct_origin_measure_v5"] * 0.50
        + ht["true_scale_language_keep"] * 0.25
        + (1.0 - ht["true_scale_forgetting_penalty"]) * 0.15
        + ht["true_scale_readiness"] * 0.10
    )
    direct_feature_measure_v6 = _clip01(
        hv["direct_feature_measure_v5"] * 0.45
        + hp["path_reuse_score"] * 0.20
        + hp["local_transport_score"] * 0.15
        + ht["true_scale_novel_gain"] * 0.10
        + (1.0 - ht["true_scale_forgetting_penalty"]) * 0.10
    )
    direct_structure_measure_v6 = _clip01(
        hv["direct_structure_measure_v5"] * 0.55
        + ht["true_scale_structure_keep"] * 0.20
        + (1.0 - ht["true_scale_collapse_risk"]) * 0.15
        + hb["structure_rule_alignment_v11"] * 0.10
    )
    direct_route_measure_v6 = _clip01(
        hv["direct_route_measure_v5"] * 0.45
        + ht["true_scale_context_keep"] * 0.25
        + hb["extreme_guard_v11"] * 0.15
        + ht["true_scale_readiness"] * 0.15
    )
    direct_brain_measure_v6 = (
        direct_origin_measure_v6
        + direct_feature_measure_v6
        + direct_structure_measure_v6
        + direct_route_measure_v6
    ) / 4.0
    direct_brain_gap_v6 = 1.0 - direct_brain_measure_v6
    direct_scale_alignment_v6 = (
        direct_structure_measure_v6
        + direct_route_measure_v6
        + ht["true_scale_readiness"]
        + hb["topology_training_readiness_v11"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v6": direct_origin_measure_v6,
            "direct_feature_measure_v6": direct_feature_measure_v6,
            "direct_structure_measure_v6": direct_structure_measure_v6,
            "direct_route_measure_v6": direct_route_measure_v6,
            "direct_brain_measure_v6": direct_brain_measure_v6,
            "direct_brain_gap_v6": direct_brain_gap_v6,
            "direct_scale_alignment_v6": direct_scale_alignment_v6,
        },
        "direct_equation_v6": {
            "origin_term": "D_origin_v6 = 0.50 * D_origin_v5 + 0.25 * L_true + 0.15 * (1 - P_true) + 0.10 * A_true",
            "feature_term": "D_feature_v6 = 0.45 * D_feature_v5 + 0.20 * R_topo + 0.15 * T_local + 0.10 * G_true + 0.10 * (1 - P_true)",
            "structure_term": "D_structure_v6 = 0.55 * D_structure_v5 + 0.20 * S_true + 0.15 * (1 - R_true) + 0.10 * B_struct_v11",
            "route_term": "D_route_v6 = 0.45 * D_route_v5 + 0.25 * C_true + 0.15 * H_ext_v11 + 0.15 * A_true",
            "system_term": "M_brain_direct_v6 = mean(D_origin_v6, D_feature_v6, D_structure_v6, D_route_v6)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第六版开始显式吸收真实规模化压力结果，使脑编码直测链不再只在温和原型里成立，而是在更大规模、更长上下文和更强更新压力下重新校准。",
            "next_question": "下一步要把第六版直测链并回训练终式和主核，检验在更真实规模化条件下主核是否还能继续收口。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第六版报告",
        "",
        f"- direct_origin_measure_v6: {hm['direct_origin_measure_v6']:.6f}",
        f"- direct_feature_measure_v6: {hm['direct_feature_measure_v6']:.6f}",
        f"- direct_structure_measure_v6: {hm['direct_structure_measure_v6']:.6f}",
        f"- direct_route_measure_v6: {hm['direct_route_measure_v6']:.6f}",
        f"- direct_brain_measure_v6: {hm['direct_brain_measure_v6']:.6f}",
        f"- direct_brain_gap_v6: {hm['direct_brain_gap_v6']:.6f}",
        f"- direct_scale_alignment_v6: {hm['direct_scale_alignment_v6']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v6_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
