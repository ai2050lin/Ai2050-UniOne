from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v71_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v71_summary() -> dict:
    v70 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v70_20260321" / "summary.json"
    )
    mega = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_coupled_degradation_validation_20260321" / "summary.json"
    )
    brain_v9 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v9_20260321" / "summary.json"
    )
    bridge_v15 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v15_20260321" / "summary.json"
    )

    hv = v70["headline_metrics"]
    hm = mega["headline_metrics"]
    hb = brain_v9["headline_metrics"]
    ht = bridge_v15["headline_metrics"]

    feature_term_v71 = (
        hv["feature_term_v70"]
        + hv["feature_term_v70"] * hm["mega_coupled_readiness"] * 0.004
        + hv["feature_term_v70"] * ht["plasticity_rule_alignment_v15"] * 0.001
        + hv["feature_term_v70"] * hb["direct_feature_measure_v9"] * 0.001
    )
    structure_term_v71 = (
        hv["structure_term_v70"]
        + hv["structure_term_v70"] * hm["mega_coupled_structure_keep"] * 0.007
        + hv["structure_term_v70"] * ht["structure_rule_alignment_v15"] * 0.004
        + hv["structure_term_v70"] * hb["direct_structure_measure_v9"] * 0.002
    )
    learning_term_v71 = (
        hv["learning_term_v70"]
        + hv["learning_term_v70"] * ht["topology_training_readiness_v15"]
        + hm["mega_coupled_margin"] * 1000.0
        + hm["mega_coupled_readiness"] * 1000.0
        + hb["direct_brain_measure_v9"] * 1000.0
    )
    pressure_term_v71 = max(
        0.0,
        hv["pressure_term_v70"]
        + ht["topology_training_gap_v15"]
        + hm["mega_coupled_collapse_risk"]
        + hm["mega_coupled_route_degradation"] * 0.2,
    )
    encoding_margin_v71 = feature_term_v71 + structure_term_v71 + learning_term_v71 - pressure_term_v71

    return {
        "headline_metrics": {
            "feature_term_v71": feature_term_v71,
            "structure_term_v71": structure_term_v71,
            "learning_term_v71": learning_term_v71,
            "pressure_term_v71": pressure_term_v71,
            "encoding_margin_v71": encoding_margin_v71,
        },
        "closed_form_equation_v71": {
            "feature_term": "K_f_v71 = K_f_v70 + K_f_v70 * A_mega * 0.004 + K_f_v70 * B_plastic_v15 * 0.001 + K_f_v70 * D_feature_v9 * 0.001",
            "structure_term": "K_s_v71 = K_s_v70 + K_s_v70 * S_mega * 0.007 + K_s_v70 * B_struct_v15 * 0.004 + K_s_v70 * D_structure_v9 * 0.002",
            "learning_term": "K_l_v71 = K_l_v70 + K_l_v70 * R_train_v15 + M_mega * 1000 + A_mega * 1000 + M_brain_direct_v9 * 1000",
            "pressure_term": "P_v71 = P_v70 + G_train_v15 + R_collapse_mega + 0.2 * R_route_mega",
            "margin_term": "M_encoding_v71 = K_f_v71 + K_s_v71 + K_l_v71 - P_v71",
        },
        "project_readout": {
            "summary": "第七十一版主核开始把更大系统联动退化链、脑编码第九版直测链和训练终式第十五桥一起并回主核，使主核更接近真实大系统在线学习下的受压状态。",
            "next_question": "下一步要把这条主核推进到更大的可训练系统里，检验联动退化是否会触发真正的系统级失稳。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第七十一版报告",
        "",
        f"- feature_term_v71: {hm['feature_term_v71']:.6f}",
        f"- structure_term_v71: {hm['structure_term_v71']:.6f}",
        f"- learning_term_v71: {hm['learning_term_v71']:.6f}",
        f"- pressure_term_v71: {hm['pressure_term_v71']:.6f}",
        f"- encoding_margin_v71: {hm['encoding_margin_v71']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v71_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
