from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v85_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v85_summary() -> dict:
    v84 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v84_20260321" / "summary.json"
    )
    systemic = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_stable_amplification_validation_20260321" / "summary.json"
    )
    brain_v23 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v23_20260321" / "summary.json"
    )
    bridge_v29 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v29_20260321" / "summary.json"
    )

    hv = v84["headline_metrics"]
    hs = systemic["headline_metrics"]
    hb = brain_v23["headline_metrics"]
    ht = bridge_v29["headline_metrics"]

    feature_term_v85 = (
        hv["feature_term_v84"]
        + hv["feature_term_v84"] * hs["systemic_score"] * 0.004
        + hv["feature_term_v84"] * ht["plasticity_rule_alignment_v29"] * 0.001
        + hv["feature_term_v84"] * hb["direct_feature_measure_v23"] * 0.001
    )
    structure_term_v85 = (
        hv["structure_term_v84"]
        + hv["structure_term_v84"] * hs["systemic_structure_stability"] * 0.007
        + hv["structure_term_v84"] * ht["structure_rule_alignment_v29"] * 0.004
        + hv["structure_term_v84"] * hb["direct_structure_measure_v23"] * 0.002
    )
    learning_term_v85 = (
        hv["learning_term_v84"]
        + hv["learning_term_v84"] * ht["topology_training_readiness_v29"]
        + hs["systemic_margin"] * 1000.0
        + hs["systemic_score"] * 1000.0
        + hb["direct_brain_measure_v23"] * 1000.0
    )
    pressure_term_v85 = max(
        0.0,
        hv["pressure_term_v84"]
        + ht["topology_training_gap_v29"]
        + hs["systemic_residual_penalty"]
        + (1.0 - hs["systemic_route_stability"]) * 0.2,
    )
    encoding_margin_v85 = feature_term_v85 + structure_term_v85 + learning_term_v85 - pressure_term_v85

    return {
        "headline_metrics": {
            "feature_term_v85": feature_term_v85,
            "structure_term_v85": structure_term_v85,
            "learning_term_v85": learning_term_v85,
            "pressure_term_v85": pressure_term_v85,
            "encoding_margin_v85": encoding_margin_v85,
        },
        "closed_form_equation_v85": {
            "feature_term": "K_f_v85 = K_f_v84 + K_f_v84 * S_system_score * 0.004 + K_f_v84 * B_plastic_v29 * 0.001 + K_f_v84 * D_feature_v23 * 0.001",
            "structure_term": "K_s_v85 = K_s_v84 + K_s_v84 * S_system * 0.007 + K_s_v84 * B_struct_v29 * 0.004 + K_s_v84 * D_structure_v23 * 0.002",
            "learning_term": "K_l_v85 = K_l_v84 + K_l_v84 * R_train_v29 + M_system * 1000 + S_system_score * 1000 + M_brain_direct_v23 * 1000",
            "pressure_term": "P_v85 = P_v84 + G_train_v29 + P_system + 0.2 * (1 - R_system)",
            "margin_term": "M_encoding_v85 = K_f_v85 + K_s_v85 + K_l_v85 - P_v85",
        },
        "project_readout": {
            "summary": "第八十五版主核开始把系统级稳定放大验证、脑编码第二十三版和训练终式第二十九桥一起并回主核，直接检验系统级稳定放大是否开始形成。",
            "next_question": "下一步要把这条主核推进到更大、更长、更高压的系统里，验证系统级稳定放大是否真正成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第八十五版报告",
        "",
        f"- feature_term_v85: {hm['feature_term_v85']:.6f}",
        f"- structure_term_v85: {hm['structure_term_v85']:.6f}",
        f"- learning_term_v85: {hm['learning_term_v85']:.6f}",
        f"- pressure_term_v85: {hm['pressure_term_v85']:.6f}",
        f"- encoding_margin_v85: {hm['encoding_margin_v85']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v85_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
