from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v79_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v79_summary() -> dict:
    v78 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v78_20260321" / "summary.json"
    )
    amplification = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_sustained_amplification_validation_20260321" / "summary.json"
    )
    brain_v17 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v17_20260321" / "summary.json"
    )
    bridge_v23 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v23_20260321" / "summary.json"
    )

    hv = v78["headline_metrics"]
    ha = amplification["headline_metrics"]
    hb = brain_v17["headline_metrics"]
    ht = bridge_v23["headline_metrics"]

    feature_term_v79 = (
        hv["feature_term_v78"]
        + hv["feature_term_v78"] * ha["amplification_score"] * 0.004
        + hv["feature_term_v78"] * ht["plasticity_rule_alignment_v23"] * 0.001
        + hv["feature_term_v78"] * hb["direct_feature_measure_v17"] * 0.001
    )
    structure_term_v79 = (
        hv["structure_term_v78"]
        + hv["structure_term_v78"] * ha["amplification_structure"] * 0.007
        + hv["structure_term_v78"] * ht["structure_rule_alignment_v23"] * 0.004
        + hv["structure_term_v78"] * hb["direct_structure_measure_v17"] * 0.002
    )
    learning_term_v79 = (
        hv["learning_term_v78"]
        + hv["learning_term_v78"] * ht["topology_training_readiness_v23"]
        + ha["amplification_margin"] * 1000.0
        + ha["amplification_score"] * 1000.0
        + hb["direct_brain_measure_v17"] * 1000.0
    )
    pressure_term_v79 = max(
        0.0,
        hv["pressure_term_v78"]
        + ht["topology_training_gap_v23"]
        + ha["amplification_penalty"]
        + (1.0 - ha["amplification_route"]) * 0.2,
    )
    encoding_margin_v79 = feature_term_v79 + structure_term_v79 + learning_term_v79 - pressure_term_v79

    return {
        "headline_metrics": {
            "feature_term_v79": feature_term_v79,
            "structure_term_v79": structure_term_v79,
            "learning_term_v79": learning_term_v79,
            "pressure_term_v79": pressure_term_v79,
            "encoding_margin_v79": encoding_margin_v79,
        },
        "closed_form_equation_v79": {
            "feature_term": "K_f_v79 = K_f_v78 + K_f_v78 * S_amp_score * 0.004 + K_f_v78 * B_plastic_v23 * 0.001 + K_f_v78 * D_feature_v17 * 0.001",
            "structure_term": "K_s_v79 = K_s_v78 + K_s_v78 * S_amp * 0.007 + K_s_v78 * B_struct_v23 * 0.004 + K_s_v78 * D_structure_v17 * 0.002",
            "learning_term": "K_l_v79 = K_l_v78 + K_l_v78 * R_train_v23 + M_amp * 1000 + S_amp_score * 1000 + M_brain_direct_v17 * 1000",
            "pressure_term": "P_v79 = P_v78 + G_train_v23 + P_amp + 0.2 * (1 - R_amp)",
            "margin_term": "M_encoding_v79 = K_f_v79 + K_s_v79 + K_l_v79 - P_v79",
        },
        "project_readout": {
            "summary": "第七十九版主核开始把更大系统持续放大验证、脑编码第十七版和训练终式第二十三桥一起并回主核，直接检测持续回升是否开始变成系统级放大。",
            "next_question": "下一步要把这条主核推进到更大、更长、更高压的系统里，验证放大趋势是否真的开始成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第七十九版报告",
        "",
        f"- feature_term_v79: {hm['feature_term_v79']:.6f}",
        f"- structure_term_v79: {hm['structure_term_v79']:.6f}",
        f"- learning_term_v79: {hm['learning_term_v79']:.6f}",
        f"- pressure_term_v79: {hm['pressure_term_v79']:.6f}",
        f"- encoding_margin_v79: {hm['encoding_margin_v79']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v79_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
