from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v64_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v64_summary() -> dict:
    v63 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v63_20260321" / "summary.json"
    )
    high_intensity = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_online_high_intensity_update_20260321" / "summary.json"
    )
    bridge_v8 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v8_20260321" / "summary.json"
    )

    hv = v63["headline_metrics"]
    hh = high_intensity["headline_metrics"]
    hb = bridge_v8["headline_metrics"]

    feature_term_v64 = (
        hv["feature_term_v63"]
        + hv["feature_term_v63"] * hh["high_intensity_language_keep"] * 0.004
        + hv["feature_term_v63"] * hb["plasticity_rule_alignment_v8"] * 0.001
    )
    structure_term_v64 = (
        hv["structure_term_v63"]
        + hv["structure_term_v63"] * hh["high_intensity_structure_keep"] * 0.007
        + hv["structure_term_v63"] * hb["structure_rule_alignment_v8"] * 0.004
    )
    learning_term_v64 = (
        hv["learning_term_v63"]
        + hv["learning_term_v63"] * hb["topology_training_readiness_v8"]
        + hh["high_intensity_margin"] * 1000.0
        + hh["high_intensity_novel_gain"] * 1000.0
    )
    pressure_term_v64 = max(
        0.0,
        hv["pressure_term_v63"]
        + hb["topology_training_gap_v8"]
        + hh["high_intensity_forgetting_penalty"]
        + (1.0 - hh["high_intensity_stability"]) * 0.2,
    )
    encoding_margin_v64 = feature_term_v64 + structure_term_v64 + learning_term_v64 - pressure_term_v64

    return {
        "headline_metrics": {
            "feature_term_v64": feature_term_v64,
            "structure_term_v64": structure_term_v64,
            "learning_term_v64": learning_term_v64,
            "pressure_term_v64": pressure_term_v64,
            "encoding_margin_v64": encoding_margin_v64,
        },
        "closed_form_equation_v64": {
            "feature_term": "K_f_v64 = K_f_v63 + K_f_v63 * L_hi * 0.004 + K_f_v63 * B_plastic_v8 * 0.001",
            "structure_term": "K_s_v64 = K_s_v63 + K_s_v63 * S_hi * 0.007 + K_s_v63 * B_struct_v8 * 0.004",
            "learning_term": "K_l_v64 = K_l_v63 + K_l_v63 * R_train_v8 + M_hi * 1000 + G_hi * 1000",
            "pressure_term": "P_v64 = P_v63 + G_train_v8 + P_hi + 0.2 * (1 - R_hi)",
            "margin_term": "M_encoding_v64 = K_f_v64 + K_s_v64 + K_l_v64 - P_v64",
        },
        "project_readout": {
            "summary": "第六十四版主核把高强度在线更新场景和训练终式第八桥一起并回主式，使主核第一次显式容纳更严苛场景下的语言保持、结构保持、增量学习和遗忘惩罚。",
            "next_question": "下一步要把这条主核放进更长时间尺度的高强度在线原型，验证是否会出现累积性系统失稳。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第六十四版报告",
        "",
        f"- feature_term_v64: {hm['feature_term_v64']:.6f}",
        f"- structure_term_v64: {hm['structure_term_v64']:.6f}",
        f"- learning_term_v64: {hm['learning_term_v64']:.6f}",
        f"- pressure_term_v64: {hm['pressure_term_v64']:.6f}",
        f"- encoding_margin_v64: {hm['encoding_margin_v64']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v64_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
