from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v73_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v73_summary() -> dict:
    v72 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v72_20260321" / "summary.json"
    )
    plateau = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_plateau_break_probe_20260321" / "summary.json"
    )
    brain_v11 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v11_20260321" / "summary.json"
    )
    bridge_v17 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v17_20260321" / "summary.json"
    )

    hv = v72["headline_metrics"]
    hp = plateau["headline_metrics"]
    hb = brain_v11["headline_metrics"]
    ht = bridge_v17["headline_metrics"]

    feature_term_v73 = (
        hv["feature_term_v72"]
        + hv["feature_term_v72"] * hp["plateau_break_readiness"] * 0.004
        + hv["feature_term_v72"] * ht["plasticity_rule_alignment_v17"] * 0.001
        + hv["feature_term_v72"] * hb["direct_feature_measure_v11"] * 0.001
    )
    structure_term_v73 = (
        hv["structure_term_v72"]
        + hv["structure_term_v72"] * hp["plateau_structure_guard"] * 0.007
        + hv["structure_term_v72"] * ht["structure_rule_alignment_v17"] * 0.004
        + hv["structure_term_v72"] * hb["direct_structure_measure_v11"] * 0.002
    )
    learning_term_v73 = (
        hv["learning_term_v72"]
        + hv["learning_term_v72"] * ht["topology_training_readiness_v17"]
        + hp["plateau_break_margin"] * 1000.0
        + hp["plateau_break_readiness"] * 1000.0
        + hb["direct_brain_measure_v11"] * 1000.0
    )
    pressure_term_v73 = max(
        0.0,
        hv["pressure_term_v72"]
        + ht["topology_training_gap_v17"]
        + hp["plateau_instability_penalty"]
        + (1.0 - hp["plateau_route_guard"]) * 0.2,
    )
    encoding_margin_v73 = feature_term_v73 + structure_term_v73 + learning_term_v73 - pressure_term_v73

    return {
        "headline_metrics": {
            "feature_term_v73": feature_term_v73,
            "structure_term_v73": structure_term_v73,
            "learning_term_v73": learning_term_v73,
            "pressure_term_v73": pressure_term_v73,
            "encoding_margin_v73": encoding_margin_v73,
        },
        "closed_form_equation_v73": {
            "feature_term": "K_f_v73 = K_f_v72 + K_f_v72 * A_break * 0.004 + K_f_v72 * B_plastic_v17 * 0.001 + K_f_v72 * D_feature_v11 * 0.001",
            "structure_term": "K_s_v73 = K_s_v72 + K_s_v72 * G_struct_break * 0.007 + K_s_v72 * B_struct_v17 * 0.004 + K_s_v72 * D_structure_v11 * 0.002",
            "learning_term": "K_l_v73 = K_l_v72 + K_l_v72 * R_train_v17 + M_break * 1000 + A_break * 1000 + M_brain_direct_v11 * 1000",
            "pressure_term": "P_v73 = P_v72 + G_train_v17 + P_break + 0.2 * (1 - G_route_break)",
            "margin_term": "M_encoding_v73 = K_f_v73 + K_s_v73 + K_l_v73 - P_v73",
        },
        "project_readout": {
            "summary": "第七十三版主核开始把破平台探针、脑编码第十一版直测链和训练终式第十七桥一起并回主核，使主核开始直接回答“平台期会不会松动”。",
            "next_question": "下一步要把这条主核推进到更大的可训练系统里，检验平台期是否真的开始松动。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第七十三版报告",
        "",
        f"- feature_term_v73: {hm['feature_term_v73']:.6f}",
        f"- structure_term_v73: {hm['structure_term_v73']:.6f}",
        f"- learning_term_v73: {hm['learning_term_v73']:.6f}",
        f"- pressure_term_v73: {hm['pressure_term_v73']:.6f}",
        f"- encoding_margin_v73: {hm['encoding_margin_v73']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v73_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
