from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v80_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v80_summary() -> dict:
    v79 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v79_20260321" / "summary.json"
    )
    reinforce = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_sustained_amplification_strengthening_20260321" / "summary.json"
    )
    brain_v18 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v18_20260321" / "summary.json"
    )
    bridge_v24 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v24_20260321" / "summary.json"
    )

    hv = v79["headline_metrics"]
    hr = reinforce["headline_metrics"]
    hb = brain_v18["headline_metrics"]
    ht = bridge_v24["headline_metrics"]

    feature_term_v80 = (
        hv["feature_term_v79"]
        + hv["feature_term_v79"] * hr["amplification_reinforced_score"] * 0.004
        + hv["feature_term_v79"] * ht["plasticity_rule_alignment_v24"] * 0.001
        + hv["feature_term_v79"] * hb["direct_feature_measure_v18"] * 0.001
    )
    structure_term_v80 = (
        hv["structure_term_v79"]
        + hv["structure_term_v79"] * hr["amplification_structure_stability"] * 0.007
        + hv["structure_term_v79"] * ht["structure_rule_alignment_v24"] * 0.004
        + hv["structure_term_v79"] * hb["direct_structure_measure_v18"] * 0.002
    )
    learning_term_v80 = (
        hv["learning_term_v79"]
        + hv["learning_term_v79"] * ht["topology_training_readiness_v24"]
        + hr["amplification_reinforced_margin"] * 1000.0
        + hr["amplification_reinforced_score"] * 1000.0
        + hb["direct_brain_measure_v18"] * 1000.0
    )
    pressure_term_v80 = max(
        0.0,
        hv["pressure_term_v79"]
        + ht["topology_training_gap_v24"]
        + hr["amplification_residual_penalty"]
        + (1.0 - hr["amplification_route_stability"]) * 0.2,
    )
    encoding_margin_v80 = feature_term_v80 + structure_term_v80 + learning_term_v80 - pressure_term_v80

    return {
        "headline_metrics": {
            "feature_term_v80": feature_term_v80,
            "structure_term_v80": structure_term_v80,
            "learning_term_v80": learning_term_v80,
            "pressure_term_v80": pressure_term_v80,
            "encoding_margin_v80": encoding_margin_v80,
        },
        "closed_form_equation_v80": {
            "feature_term": "K_f_v80 = K_f_v79 + K_f_v79 * S_reinforce_score * 0.004 + K_f_v79 * B_plastic_v24 * 0.001 + K_f_v79 * D_feature_v18 * 0.001",
            "structure_term": "K_s_v80 = K_s_v79 + K_s_v79 * S_reinforce * 0.007 + K_s_v79 * B_struct_v24 * 0.004 + K_s_v79 * D_structure_v18 * 0.002",
            "learning_term": "K_l_v80 = K_l_v79 + K_l_v79 * R_train_v24 + M_reinforce * 1000 + S_reinforce_score * 1000 + M_brain_direct_v18 * 1000",
            "pressure_term": "P_v80 = P_v79 + G_train_v24 + P_reinforce + 0.2 * (1 - R_reinforce)",
            "margin_term": "M_encoding_v80 = K_f_v80 + K_s_v80 + K_l_v80 - P_v80",
        },
        "project_readout": {
            "summary": "第八十版主核开始把持续放大强化、脑编码第十八版和训练终式第二十四桥一起并回主核，直接检测放大趋势是否开始稳态化。",
            "next_question": "下一步要把这条主核推进到更大、更长、更高压的系统里，验证稳态放大是否真的开始成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第八十版报告",
        "",
        f"- feature_term_v80: {hm['feature_term_v80']:.6f}",
        f"- structure_term_v80: {hm['structure_term_v80']:.6f}",
        f"- learning_term_v80: {hm['learning_term_v80']:.6f}",
        f"- pressure_term_v80: {hm['pressure_term_v80']:.6f}",
        f"- encoding_margin_v80: {hm['encoding_margin_v80']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v80_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
