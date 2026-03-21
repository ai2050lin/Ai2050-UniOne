from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v86_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v86_summary() -> dict:
    v85 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v85_20260321" / "summary.json"
    )
    systemic_steady = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_steady_amplification_validation_20260321" / "summary.json"
    )
    brain_v24 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v24_20260321" / "summary.json"
    )
    bridge_v30 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v30_20260321" / "summary.json"
    )

    hv = v85["headline_metrics"]
    hs = systemic_steady["headline_metrics"]
    hb = brain_v24["headline_metrics"]
    ht = bridge_v30["headline_metrics"]

    feature_term_v86 = (
        hv["feature_term_v85"]
        + hv["feature_term_v85"] * hs["systemic_steady_score"] * 0.004
        + hv["feature_term_v85"] * ht["plasticity_rule_alignment_v30"] * 0.001
        + hv["feature_term_v85"] * hb["direct_feature_measure_v24"] * 0.001
    )
    structure_term_v86 = (
        hv["structure_term_v85"]
        + hv["structure_term_v85"] * hs["systemic_steady_structure"] * 0.007
        + hv["structure_term_v85"] * ht["structure_rule_alignment_v30"] * 0.004
        + hv["structure_term_v85"] * hb["direct_structure_measure_v24"] * 0.002
    )
    learning_term_v86 = (
        hv["learning_term_v85"]
        + hv["learning_term_v85"] * ht["topology_training_readiness_v30"]
        + hs["systemic_steady_margin"] * 1000.0
        + hs["systemic_steady_score"] * 1000.0
        + hb["direct_brain_measure_v24"] * 1000.0
    )
    pressure_term_v86 = max(
        0.0,
        hv["pressure_term_v85"]
        + ht["topology_training_gap_v30"]
        + hs["systemic_steady_penalty"]
        + (1.0 - hs["systemic_steady_route"]) * 0.2,
    )
    encoding_margin_v86 = feature_term_v86 + structure_term_v86 + learning_term_v86 - pressure_term_v86

    return {
        "headline_metrics": {
            "feature_term_v86": feature_term_v86,
            "structure_term_v86": structure_term_v86,
            "learning_term_v86": learning_term_v86,
            "pressure_term_v86": pressure_term_v86,
            "encoding_margin_v86": encoding_margin_v86,
        },
        "closed_form_equation_v86": {
            "feature_term": "K_f_v86 = K_f_v85 + K_f_v85 * S_system_steady_score * 0.004 + K_f_v85 * B_plastic_v30 * 0.001 + K_f_v85 * D_feature_v24 * 0.001",
            "structure_term": "K_s_v86 = K_s_v85 + K_s_v85 * S_system_steady * 0.007 + K_s_v85 * B_struct_v30 * 0.004 + K_s_v85 * D_structure_v24 * 0.002",
            "learning_term": "K_l_v86 = K_l_v85 + K_l_v85 * R_train_v30 + M_system_steady * 1000 + S_system_steady_score * 1000 + M_brain_direct_v24 * 1000",
            "pressure_term": "P_v86 = P_v85 + G_train_v30 + P_system_steady + 0.2 * (1 - R_system_steady)",
            "margin_term": "M_encoding_v86 = K_f_v86 + K_s_v86 + K_l_v86 - P_v86",
        },
        "project_readout": {
            "summary": "第八十六版主核开始把系统级稳态放大验证、脑编码第二十四版和训练终式第三十桥一起并回主核，直接检验系统级稳态放大是否开始站住。",
            "next_question": "下一步要把这条主核推进到更大、更长、更高压的系统里，验证系统级稳态放大是否继续向低风险区推进。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第八十六版报告",
        "",
        f"- feature_term_v86: {hm['feature_term_v86']:.6f}",
        f"- structure_term_v86: {hm['structure_term_v86']:.6f}",
        f"- learning_term_v86: {hm['learning_term_v86']:.6f}",
        f"- pressure_term_v86: {hm['pressure_term_v86']:.6f}",
        f"- encoding_margin_v86: {hm['encoding_margin_v86']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v86_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
