from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v76_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v76_summary() -> dict:
    v75 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v75_20260321" / "summary.json"
    )
    attenuation = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_propagation_attenuation_probe_20260321" / "summary.json"
    )
    brain_v14 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v14_20260321" / "summary.json"
    )
    bridge_v20 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v20_20260321" / "summary.json"
    )

    hv = v75["headline_metrics"]
    ha = attenuation["headline_metrics"]
    hb = brain_v14["headline_metrics"]
    ht = bridge_v20["headline_metrics"]

    feature_term_v76 = (
        hv["feature_term_v75"]
        + hv["feature_term_v75"] * ha["anti_attenuation_readiness"] * 0.004
        + hv["feature_term_v75"] * ht["plasticity_rule_alignment_v20"] * 0.001
        + hv["feature_term_v75"] * hb["direct_feature_measure_v14"] * 0.001
    )
    structure_term_v76 = (
        hv["structure_term_v75"]
        + hv["structure_term_v75"] * (1.0 - ha["attenuation_structure"]) * 0.007
        + hv["structure_term_v75"] * ht["structure_rule_alignment_v20"] * 0.004
        + hv["structure_term_v75"] * hb["direct_structure_measure_v14"] * 0.002
    )
    learning_term_v76 = (
        hv["learning_term_v75"]
        + hv["learning_term_v75"] * ht["topology_training_readiness_v20"]
        + ha["anti_attenuation_margin"] * 1000.0
        + ha["anti_attenuation_readiness"] * 1000.0
        + hb["direct_brain_measure_v14"] * 1000.0
    )
    pressure_term_v76 = max(
        0.0,
        hv["pressure_term_v75"]
        + ht["topology_training_gap_v20"]
        + ha["attenuation_penalty"]
        + ha["attenuation_gap"] * 0.2,
    )
    encoding_margin_v76 = feature_term_v76 + structure_term_v76 + learning_term_v76 - pressure_term_v76

    return {
        "headline_metrics": {
            "feature_term_v76": feature_term_v76,
            "structure_term_v76": structure_term_v76,
            "learning_term_v76": learning_term_v76,
            "pressure_term_v76": pressure_term_v76,
            "encoding_margin_v76": encoding_margin_v76,
        },
        "closed_form_equation_v76": {
            "feature_term": "K_f_v76 = K_f_v75 + K_f_v75 * R_anti_att * 0.004 + K_f_v75 * B_plastic_v20 * 0.001 + K_f_v75 * D_feature_v14 * 0.001",
            "structure_term": "K_s_v76 = K_s_v75 + K_s_v75 * (1 - A_struct) * 0.007 + K_s_v75 * B_struct_v20 * 0.004 + K_s_v75 * D_structure_v14 * 0.002",
            "learning_term": "K_l_v76 = K_l_v75 + K_l_v75 * R_train_v20 + M_anti_att * 1000 + R_anti_att * 1000 + M_brain_direct_v14 * 1000",
            "pressure_term": "P_v76 = P_v75 + G_train_v20 + P_att + 0.2 * G_att",
            "margin_term": "M_encoding_v76 = K_f_v76 + K_s_v76 + K_l_v76 - P_v76",
        },
        "project_readout": {
            "summary": "第七十六版主核开始把传播衰减探针、脑编码第十四版直测链和训练终式第二十桥一起并回主核，使主核开始直接回答系统能否从传播衰减走向补偿。",
            "next_question": "下一步要把这条主核推进到更大的可训练系统里，检验平台期松动是否能够真正被放大而不再衰减。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第七十六版报告",
        "",
        f"- feature_term_v76: {hm['feature_term_v76']:.6f}",
        f"- structure_term_v76: {hm['structure_term_v76']:.6f}",
        f"- learning_term_v76: {hm['learning_term_v76']:.6f}",
        f"- pressure_term_v76: {hm['pressure_term_v76']:.6f}",
        f"- encoding_margin_v76: {hm['encoding_margin_v76']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v76_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
