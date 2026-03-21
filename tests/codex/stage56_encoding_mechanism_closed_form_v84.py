from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v84_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v84_summary() -> dict:
    v83 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v83_20260321" / "summary.json"
    )
    stable_plus = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_stable_amplification_strengthening_20260321" / "summary.json"
    )
    brain_v22 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v22_20260321" / "summary.json"
    )
    bridge_v28 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v28_20260321" / "summary.json"
    )

    hv = v83["headline_metrics"]
    hs = stable_plus["headline_metrics"]
    hb = brain_v22["headline_metrics"]
    ht = bridge_v28["headline_metrics"]

    feature_term_v84 = (
        hv["feature_term_v83"]
        + hv["feature_term_v83"] * hs["stable_reinforced_score"] * 0.004
        + hv["feature_term_v83"] * ht["plasticity_rule_alignment_v28"] * 0.001
        + hv["feature_term_v83"] * hb["direct_feature_measure_v22"] * 0.001
    )
    structure_term_v84 = (
        hv["structure_term_v83"]
        + hv["structure_term_v83"] * hs["stable_reinforced_structure"] * 0.007
        + hv["structure_term_v83"] * ht["structure_rule_alignment_v28"] * 0.004
        + hv["structure_term_v83"] * hb["direct_structure_measure_v22"] * 0.002
    )
    learning_term_v84 = (
        hv["learning_term_v83"]
        + hv["learning_term_v83"] * ht["topology_training_readiness_v28"]
        + hs["stable_reinforced_margin"] * 1000.0
        + hs["stable_reinforced_score"] * 1000.0
        + hb["direct_brain_measure_v22"] * 1000.0
    )
    pressure_term_v84 = max(
        0.0,
        hv["pressure_term_v83"]
        + ht["topology_training_gap_v28"]
        + hs["stable_reinforced_penalty"]
        + (1.0 - hs["stable_reinforced_route"]) * 0.2,
    )
    encoding_margin_v84 = feature_term_v84 + structure_term_v84 + learning_term_v84 - pressure_term_v84

    return {
        "headline_metrics": {
            "feature_term_v84": feature_term_v84,
            "structure_term_v84": structure_term_v84,
            "learning_term_v84": learning_term_v84,
            "pressure_term_v84": pressure_term_v84,
            "encoding_margin_v84": encoding_margin_v84,
        },
        "closed_form_equation_v84": {
            "feature_term": "K_f_v84 = K_f_v83 + K_f_v83 * S_stable_plus_score * 0.004 + K_f_v83 * B_plastic_v28 * 0.001 + K_f_v83 * D_feature_v22 * 0.001",
            "structure_term": "K_s_v84 = K_s_v83 + K_s_v83 * S_stable_plus * 0.007 + K_s_v83 * B_struct_v28 * 0.004 + K_s_v83 * D_structure_v22 * 0.002",
            "learning_term": "K_l_v84 = K_l_v83 + K_l_v83 * R_train_v28 + M_stable_plus * 1000 + S_stable_plus_score * 1000 + M_brain_direct_v22 * 1000",
            "pressure_term": "P_v84 = P_v83 + G_train_v28 + P_stable_plus + 0.2 * (1 - R_stable_plus)",
            "margin_term": "M_encoding_v84 = K_f_v84 + K_s_v84 + K_l_v84 - P_v84",
        },
        "project_readout": {
            "summary": "第八十四版主核开始把稳定放大强化、脑编码第二十二版和训练终式第二十八桥一起并回主核，直接检验稳定放大是否继续增强。",
            "next_question": "下一步要把这条主核推进到更大、更长、更高压的系统里，验证稳定放大能否继续增强成系统级稳态放大。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第八十四版报告",
        "",
        f"- feature_term_v84: {hm['feature_term_v84']:.6f}",
        f"- structure_term_v84: {hm['structure_term_v84']:.6f}",
        f"- learning_term_v84: {hm['learning_term_v84']:.6f}",
        f"- pressure_term_v84: {hm['pressure_term_v84']:.6f}",
        f"- encoding_margin_v84: {hm['encoding_margin_v84']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v84_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
