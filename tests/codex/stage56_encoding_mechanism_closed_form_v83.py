from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v83_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v83_summary() -> dict:
    v82 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v82_20260321" / "summary.json"
    )
    stable = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_stable_amplification_validation_20260321" / "summary.json"
    )
    brain_v21 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v21_20260321" / "summary.json"
    )
    bridge_v27 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v27_20260321" / "summary.json"
    )

    hv = v82["headline_metrics"]
    hs = stable["headline_metrics"]
    hb = brain_v21["headline_metrics"]
    ht = bridge_v27["headline_metrics"]

    feature_term_v83 = (
        hv["feature_term_v82"]
        + hv["feature_term_v82"] * hs["stable_score"] * 0.004
        + hv["feature_term_v82"] * ht["plasticity_rule_alignment_v27"] * 0.001
        + hv["feature_term_v82"] * hb["direct_feature_measure_v21"] * 0.001
    )
    structure_term_v83 = (
        hv["structure_term_v82"]
        + hv["structure_term_v82"] * hs["stable_structure_stability"] * 0.007
        + hv["structure_term_v82"] * ht["structure_rule_alignment_v27"] * 0.004
        + hv["structure_term_v82"] * hb["direct_structure_measure_v21"] * 0.002
    )
    learning_term_v83 = (
        hv["learning_term_v82"]
        + hv["learning_term_v82"] * ht["topology_training_readiness_v27"]
        + hs["stable_margin"] * 1000.0
        + hs["stable_score"] * 1000.0
        + hb["direct_brain_measure_v21"] * 1000.0
    )
    pressure_term_v83 = max(
        0.0,
        hv["pressure_term_v82"]
        + ht["topology_training_gap_v27"]
        + hs["stable_residual_penalty"]
        + (1.0 - hs["stable_route_stability"]) * 0.2,
    )
    encoding_margin_v83 = feature_term_v83 + structure_term_v83 + learning_term_v83 - pressure_term_v83

    return {
        "headline_metrics": {
            "feature_term_v83": feature_term_v83,
            "structure_term_v83": structure_term_v83,
            "learning_term_v83": learning_term_v83,
            "pressure_term_v83": pressure_term_v83,
            "encoding_margin_v83": encoding_margin_v83,
        },
        "closed_form_equation_v83": {
            "feature_term": "K_f_v83 = K_f_v82 + K_f_v82 * S_stable_score * 0.004 + K_f_v82 * B_plastic_v27 * 0.001 + K_f_v82 * D_feature_v21 * 0.001",
            "structure_term": "K_s_v83 = K_s_v82 + K_s_v82 * S_stable * 0.007 + K_s_v82 * B_struct_v27 * 0.004 + K_s_v82 * D_structure_v21 * 0.002",
            "learning_term": "K_l_v83 = K_l_v82 + K_l_v82 * R_train_v27 + M_stable * 1000 + S_stable_score * 1000 + M_brain_direct_v21 * 1000",
            "pressure_term": "P_v83 = P_v82 + G_train_v27 + P_stable + 0.2 * (1 - R_stable)",
            "margin_term": "M_encoding_v83 = K_f_v83 + K_s_v83 + K_l_v83 - P_v83",
        },
        "project_readout": {
            "summary": "第八十三版主核开始把稳定放大验证、脑编码第二十一版和训练终式第二十七桥一起并回主核，直接检验稳定放大是否继续增强。",
            "next_question": "下一步要把这条主核推进到更大、更长、更高压的系统里，验证稳定放大能否继续增强成系统级稳态放大。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第八十三版报告",
        "",
        f"- feature_term_v83: {hm['feature_term_v83']:.6f}",
        f"- structure_term_v83: {hm['structure_term_v83']:.6f}",
        f"- learning_term_v83: {hm['learning_term_v83']:.6f}",
        f"- pressure_term_v83: {hm['pressure_term_v83']:.6f}",
        f"- encoding_margin_v83: {hm['encoding_margin_v83']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v83_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
