from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v68_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v68_summary() -> dict:
    v67 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v67_20260321" / "summary.json"
    )
    true_scale = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_online_collapse_probe_20260321" / "summary.json"
    )
    brain_v6 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v6_20260321" / "summary.json"
    )
    bridge_v12 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v12_20260321" / "summary.json"
    )

    hv = v67["headline_metrics"]
    ht = true_scale["headline_metrics"]
    hb = brain_v6["headline_metrics"]
    hr = bridge_v12["headline_metrics"]

    feature_term_v68 = (
        hv["feature_term_v67"]
        + hv["feature_term_v67"] * ht["true_scale_language_keep"] * 0.004
        + hv["feature_term_v67"] * hr["plasticity_rule_alignment_v12"] * 0.001
        + hv["feature_term_v67"] * hb["direct_feature_measure_v6"] * 0.001
    )
    structure_term_v68 = (
        hv["structure_term_v67"]
        + hv["structure_term_v67"] * ht["true_scale_structure_keep"] * 0.007
        + hv["structure_term_v67"] * hr["structure_rule_alignment_v12"] * 0.004
        + hv["structure_term_v67"] * hb["direct_structure_measure_v6"] * 0.002
    )
    learning_term_v68 = (
        hv["learning_term_v67"]
        + hv["learning_term_v67"] * hr["topology_training_readiness_v12"]
        + ht["true_scale_margin"] * 1000.0
        + ht["true_scale_novel_gain"] * 1000.0
        + hb["direct_brain_measure_v6"] * 1000.0
    )
    pressure_term_v68 = max(
        0.0,
        hv["pressure_term_v67"]
        + hr["topology_training_gap_v12"]
        + ht["true_scale_forgetting_penalty"]
        + ht["true_scale_collapse_risk"] * 0.2
        + ht["true_scale_phase_shift_risk"] * 0.2,
    )
    encoding_margin_v68 = feature_term_v68 + structure_term_v68 + learning_term_v68 - pressure_term_v68

    return {
        "headline_metrics": {
            "feature_term_v68": feature_term_v68,
            "structure_term_v68": structure_term_v68,
            "learning_term_v68": learning_term_v68,
            "pressure_term_v68": pressure_term_v68,
            "encoding_margin_v68": encoding_margin_v68,
        },
        "closed_form_equation_v68": {
            "feature_term": "K_f_v68 = K_f_v67 + K_f_v67 * L_true * 0.004 + K_f_v67 * B_plastic_v12 * 0.001 + K_f_v67 * D_feature_v6 * 0.001",
            "structure_term": "K_s_v68 = K_s_v67 + K_s_v67 * S_true * 0.007 + K_s_v67 * B_struct_v12 * 0.004 + K_s_v67 * D_structure_v6 * 0.002",
            "learning_term": "K_l_v68 = K_l_v67 + K_l_v67 * R_train_v12 + M_true * 1000 + G_true * 1000 + M_brain_direct_v6 * 1000",
            "pressure_term": "P_v68 = P_v67 + G_train_v12 + P_true + 0.2 * R_true + 0.2 * Q_true",
            "margin_term": "M_encoding_v68 = K_f_v68 + K_s_v68 + K_l_v68 - P_v68",
        },
        "project_readout": {
            "summary": "第六十八版主核开始把真正规模化压力、脑编码第六版直测链和训练终式第十二桥一起并回主核，使主核更接近真实大规模在线学习系统在高压条件下的受压状态。",
            "next_question": "下一步要继续把这条主核推进到更大的可训练系统里，检验在更真实规模化条件下结构是否会发生相变式塌缩。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第六十八版报告",
        "",
        f"- feature_term_v68: {hm['feature_term_v68']:.6f}",
        f"- structure_term_v68: {hm['structure_term_v68']:.6f}",
        f"- learning_term_v68: {hm['learning_term_v68']:.6f}",
        f"- pressure_term_v68: {hm['pressure_term_v68']:.6f}",
        f"- encoding_margin_v68: {hm['encoding_margin_v68']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v68_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
