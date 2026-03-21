from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v69_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v69_summary() -> dict:
    v68 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v68_20260321" / "summary.json"
    )
    route_probe = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_route_degradation_probe_20260321" / "summary.json"
    )
    brain_v7 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v7_20260321" / "summary.json"
    )
    bridge_v13 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v13_20260321" / "summary.json"
    )

    hv = v68["headline_metrics"]
    hr = route_probe["headline_metrics"]
    hb = brain_v7["headline_metrics"]
    ht = bridge_v13["headline_metrics"]

    feature_term_v69 = (
        hv["feature_term_v68"]
        + hv["feature_term_v68"] * hr["true_scale_reinforced_readiness"] * 0.004
        + hv["feature_term_v68"] * ht["plasticity_rule_alignment_v13"] * 0.001
        + hv["feature_term_v68"] * hb["direct_feature_measure_v7"] * 0.001
    )
    structure_term_v69 = (
        hv["structure_term_v68"]
        + hv["structure_term_v68"] * hr["structure_resilience"] * 0.007
        + hv["structure_term_v68"] * ht["structure_rule_alignment_v13"] * 0.004
        + hv["structure_term_v68"] * hb["direct_structure_measure_v7"] * 0.002
    )
    learning_term_v69 = (
        hv["learning_term_v68"]
        + hv["learning_term_v68"] * ht["topology_training_readiness_v13"]
        + hr["route_phase_margin"] * 1000.0
        + hr["true_scale_reinforced_readiness"] * 1000.0
        + hb["direct_brain_measure_v7"] * 1000.0
    )
    pressure_term_v69 = max(
        0.0,
        hv["pressure_term_v68"]
        + ht["topology_training_gap_v13"]
        + hr["route_degradation_risk"]
        + hr["structure_phase_shift_risk"] * 0.2,
    )
    encoding_margin_v69 = feature_term_v69 + structure_term_v69 + learning_term_v69 - pressure_term_v69

    return {
        "headline_metrics": {
            "feature_term_v69": feature_term_v69,
            "structure_term_v69": structure_term_v69,
            "learning_term_v69": learning_term_v69,
            "pressure_term_v69": pressure_term_v69,
            "encoding_margin_v69": encoding_margin_v69,
        },
        "closed_form_equation_v69": {
            "feature_term": "K_f_v69 = K_f_v68 + K_f_v68 * A_route * 0.004 + K_f_v68 * B_plastic_v13 * 0.001 + K_f_v68 * D_feature_v7 * 0.001",
            "structure_term": "K_s_v69 = K_s_v68 + K_s_v68 * H_struct * 0.007 + K_s_v68 * B_struct_v13 * 0.004 + K_s_v68 * D_structure_v7 * 0.002",
            "learning_term": "K_l_v69 = K_l_v68 + K_l_v68 * R_train_v13 + M_route_phase * 1000 + A_route * 1000 + M_brain_direct_v7 * 1000",
            "pressure_term": "P_v69 = P_v68 + G_train_v13 + R_route + 0.2 * R_phase",
            "margin_term": "M_encoding_v69 = K_f_v69 + K_s_v69 + K_l_v69 - P_v69",
        },
        "project_readout": {
            "summary": "第六十九版主核开始把真正规模化路由退化压力、脑编码第七版直测链和训练终式第十三桥一起并回主核，使主核更接近真实大规模在线学习系统在路由退化压力下的受压状态。",
            "next_question": "下一步要继续把这条主核推进到更大的可训练系统里，检验路由退化和结构塌缩是否会一起触发系统性失稳。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第六十九版报告",
        "",
        f"- feature_term_v69: {hm['feature_term_v69']:.6f}",
        f"- structure_term_v69: {hm['structure_term_v69']:.6f}",
        f"- learning_term_v69: {hm['learning_term_v69']:.6f}",
        f"- pressure_term_v69: {hm['pressure_term_v69']:.6f}",
        f"- encoding_margin_v69: {hm['encoding_margin_v69']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v69_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
