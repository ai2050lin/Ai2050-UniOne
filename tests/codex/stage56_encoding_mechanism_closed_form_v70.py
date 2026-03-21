from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v70_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v70_summary() -> dict:
    v69 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v69_20260321" / "summary.json"
    )
    coupled = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_route_structure_coupled_validation_20260321" / "summary.json"
    )
    brain_v8 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v8_20260321" / "summary.json"
    )
    bridge_v14 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v14_20260321" / "summary.json"
    )

    hv = v69["headline_metrics"]
    hc = coupled["headline_metrics"]
    hb = brain_v8["headline_metrics"]
    ht = bridge_v14["headline_metrics"]

    feature_term_v70 = (
        hv["feature_term_v69"]
        + hv["feature_term_v69"] * hc["coupled_readiness"] * 0.004
        + hv["feature_term_v69"] * ht["plasticity_rule_alignment_v14"] * 0.001
        + hv["feature_term_v69"] * hb["direct_feature_measure_v8"] * 0.001
    )
    structure_term_v70 = (
        hv["structure_term_v69"]
        + hv["structure_term_v69"] * hc["coupled_structure_keep"] * 0.007
        + hv["structure_term_v69"] * ht["structure_rule_alignment_v14"] * 0.004
        + hv["structure_term_v69"] * hb["direct_structure_measure_v8"] * 0.002
    )
    learning_term_v70 = (
        hv["learning_term_v69"]
        + hv["learning_term_v69"] * ht["topology_training_readiness_v14"]
        + hc["coupled_margin"] * 1000.0
        + hc["coupled_readiness"] * 1000.0
        + hb["direct_brain_measure_v8"] * 1000.0
    )
    pressure_term_v70 = max(
        0.0,
        hv["pressure_term_v69"]
        + ht["topology_training_gap_v14"]
        + hc["coupled_failure_risk"]
        + hc["coupled_forgetting_penalty"] * 0.2,
    )
    encoding_margin_v70 = feature_term_v70 + structure_term_v70 + learning_term_v70 - pressure_term_v70

    return {
        "headline_metrics": {
            "feature_term_v70": feature_term_v70,
            "structure_term_v70": structure_term_v70,
            "learning_term_v70": learning_term_v70,
            "pressure_term_v70": pressure_term_v70,
            "encoding_margin_v70": encoding_margin_v70,
        },
        "closed_form_equation_v70": {
            "feature_term": "K_f_v70 = K_f_v69 + K_f_v69 * A_coupled * 0.004 + K_f_v69 * B_plastic_v14 * 0.001 + K_f_v69 * D_feature_v8 * 0.001",
            "structure_term": "K_s_v70 = K_s_v69 + K_s_v69 * K_struct * 0.007 + K_s_v69 * B_struct_v14 * 0.004 + K_s_v69 * D_structure_v8 * 0.002",
            "learning_term": "K_l_v70 = K_l_v69 + K_l_v69 * R_train_v14 + M_coupled * 1000 + A_coupled * 1000 + M_brain_direct_v8 * 1000",
            "pressure_term": "P_v70 = P_v69 + G_train_v14 + R_fail + 0.2 * P_coupled",
            "margin_term": "M_encoding_v70 = K_f_v70 + K_s_v70 + K_l_v70 - P_v70",
        },
        "project_readout": {
            "summary": "第七十版主核开始把路由-结构联动退化链、脑编码第八版直测链和训练终式第十四桥一起并回主核，使主核更接近真实大规模在线学习系统在联动退化压力下的受压状态。",
            "next_question": "下一步要继续把这条主核推进到更大的可训练系统里，检验路由退化和结构塌缩是否会联动触发系统性失稳。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第七十版报告",
        "",
        f"- feature_term_v70: {hm['feature_term_v70']:.6f}",
        f"- structure_term_v70: {hm['structure_term_v70']:.6f}",
        f"- learning_term_v70: {hm['learning_term_v70']:.6f}",
        f"- pressure_term_v70: {hm['pressure_term_v70']:.6f}",
        f"- encoding_margin_v70: {hm['encoding_margin_v70']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v70_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
