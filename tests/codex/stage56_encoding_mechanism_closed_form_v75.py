from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v75_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v75_summary() -> dict:
    v74 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v74_20260321" / "summary.json"
    )
    scale_prop = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_propagation_validation_20260321" / "summary.json"
    )
    brain_v13 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v13_20260321" / "summary.json"
    )
    bridge_v19 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v19_20260321" / "summary.json"
    )

    hv = v74["headline_metrics"]
    hs = scale_prop["headline_metrics"]
    hb = brain_v13["headline_metrics"]
    ht = bridge_v19["headline_metrics"]

    feature_term_v75 = (
        hv["feature_term_v74"]
        + hv["feature_term_v74"] * hs["scale_propagation_score"] * 0.004
        + hv["feature_term_v74"] * ht["plasticity_rule_alignment_v19"] * 0.001
        + hv["feature_term_v74"] * hb["direct_feature_measure_v13"] * 0.001
    )
    structure_term_v75 = (
        hv["structure_term_v74"]
        + hv["structure_term_v74"] * hs["scale_propagation_structure"] * 0.007
        + hv["structure_term_v74"] * ht["structure_rule_alignment_v19"] * 0.004
        + hv["structure_term_v74"] * hb["direct_structure_measure_v13"] * 0.002
    )
    learning_term_v75 = (
        hv["learning_term_v74"]
        + hv["learning_term_v74"] * ht["topology_training_readiness_v19"]
        + hs["scale_propagation_margin"] * 1000.0
        + hs["scale_propagation_score"] * 1000.0
        + hb["direct_brain_measure_v13"] * 1000.0
    )
    pressure_term_v75 = max(
        0.0,
        hv["pressure_term_v74"]
        + ht["topology_training_gap_v19"]
        + hs["scale_propagation_penalty"]
        + (1.0 - hs["scale_propagation_route"]) * 0.2,
    )
    encoding_margin_v75 = feature_term_v75 + structure_term_v75 + learning_term_v75 - pressure_term_v75

    return {
        "headline_metrics": {
            "feature_term_v75": feature_term_v75,
            "structure_term_v75": structure_term_v75,
            "learning_term_v75": learning_term_v75,
            "pressure_term_v75": pressure_term_v75,
            "encoding_margin_v75": encoding_margin_v75,
        },
        "closed_form_equation_v75": {
            "feature_term": "K_f_v75 = K_f_v74 + K_f_v74 * A_scale_prop * 0.004 + K_f_v74 * B_plastic_v19 * 0.001 + K_f_v74 * D_feature_v13 * 0.001",
            "structure_term": "K_s_v75 = K_s_v74 + K_s_v74 * S_prop_scale * 0.007 + K_s_v74 * B_struct_v19 * 0.004 + K_s_v74 * D_structure_v13 * 0.002",
            "learning_term": "K_l_v75 = K_l_v74 + K_l_v74 * R_train_v19 + M_prop_scale * 1000 + A_scale_prop * 1000 + M_brain_direct_v13 * 1000",
            "pressure_term": "P_v75 = P_v74 + G_train_v19 + P_prop_scale + 0.2 * (1 - R_prop_scale)",
            "margin_term": "M_encoding_v75 = K_f_v75 + K_s_v75 + K_l_v75 - P_v75",
        },
        "project_readout": {
            "summary": "第七十五版主核开始把更大系统传播验证、脑编码第十三版直测链和训练终式第十九桥一起并回主核，使主核开始直接回答平台期松动是否能向更大系统持续传播。",
            "next_question": "下一步要把这条主核推进到更大的可训练系统里，检验平台期松动是否会继续扩散成真正的系统级突破。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第七十五版报告",
        "",
        f"- feature_term_v75: {hm['feature_term_v75']:.6f}",
        f"- structure_term_v75: {hm['structure_term_v75']:.6f}",
        f"- learning_term_v75: {hm['learning_term_v75']:.6f}",
        f"- pressure_term_v75: {hm['pressure_term_v75']:.6f}",
        f"- encoding_margin_v75: {hm['encoding_margin_v75']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v75_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
