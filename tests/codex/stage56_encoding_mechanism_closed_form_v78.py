from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v78_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v78_summary() -> dict:
    v77 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v77_20260321" / "summary.json"
    )
    sustained = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_sustained_rebound_validation_20260321" / "summary.json"
    )
    brain_v16 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v16_20260321" / "summary.json"
    )
    bridge_v22 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v22_20260321" / "summary.json"
    )

    hv = v77["headline_metrics"]
    hs = sustained["headline_metrics"]
    hb = brain_v16["headline_metrics"]
    ht = bridge_v22["headline_metrics"]

    feature_term_v78 = (
        hv["feature_term_v77"]
        + hv["feature_term_v77"] * hs["sustained_rebound_score"] * 0.004
        + hv["feature_term_v77"] * ht["plasticity_rule_alignment_v22"] * 0.001
        + hv["feature_term_v77"] * hb["direct_feature_measure_v16"] * 0.001
    )
    structure_term_v78 = (
        hv["structure_term_v77"]
        + hv["structure_term_v77"] * hs["sustained_structure"] * 0.007
        + hv["structure_term_v77"] * ht["structure_rule_alignment_v22"] * 0.004
        + hv["structure_term_v77"] * hb["direct_structure_measure_v16"] * 0.002
    )
    learning_term_v78 = (
        hv["learning_term_v77"]
        + hv["learning_term_v77"] * ht["topology_training_readiness_v22"]
        + hs["sustained_margin"] * 1000.0
        + hs["sustained_rebound_score"] * 1000.0
        + hb["direct_brain_measure_v16"] * 1000.0
    )
    pressure_term_v78 = max(
        0.0,
        hv["pressure_term_v77"]
        + ht["topology_training_gap_v22"]
        + hs["sustained_penalty"]
        + (1.0 - hs["sustained_route"]) * 0.2,
    )
    encoding_margin_v78 = feature_term_v78 + structure_term_v78 + learning_term_v78 - pressure_term_v78

    return {
        "headline_metrics": {
            "feature_term_v78": feature_term_v78,
            "structure_term_v78": structure_term_v78,
            "learning_term_v78": learning_term_v78,
            "pressure_term_v78": pressure_term_v78,
            "encoding_margin_v78": encoding_margin_v78,
        },
        "closed_form_equation_v78": {
            "feature_term": "K_f_v78 = K_f_v77 + K_f_v77 * S_sustain_score * 0.004 + K_f_v77 * B_plastic_v22 * 0.001 + K_f_v77 * D_feature_v16 * 0.001",
            "structure_term": "K_s_v78 = K_s_v77 + K_s_v77 * S_sustain * 0.007 + K_s_v77 * B_struct_v22 * 0.004 + K_s_v77 * D_structure_v16 * 0.002",
            "learning_term": "K_l_v78 = K_l_v77 + K_l_v77 * R_train_v22 + M_sustain * 1000 + S_sustain_score * 1000 + M_brain_direct_v16 * 1000",
            "pressure_term": "P_v78 = P_v77 + G_train_v22 + P_sustain + 0.2 * (1 - R_sustain)",
            "margin_term": "M_encoding_v78 = K_f_v78 + K_s_v78 + K_l_v78 - P_v78",
        },
        "project_readout": {
            "summary": "第七十八版主核开始把更大系统持续回升验证、脑编码第十六版和训练终式第二十二桥一起并回主核，直接回答持续回升能否开始固化。",
            "next_question": "下一步要把这条主核推进到更大系统里，检验持续回升是会继续站住，还是重新掉回平台衰减区。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第七十八版报告",
        "",
        f"- feature_term_v78: {hm['feature_term_v78']:.6f}",
        f"- structure_term_v78: {hm['structure_term_v78']:.6f}",
        f"- learning_term_v78: {hm['learning_term_v78']:.6f}",
        f"- pressure_term_v78: {hm['pressure_term_v78']:.6f}",
        f"- encoding_margin_v78: {hm['encoding_margin_v78']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v78_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
