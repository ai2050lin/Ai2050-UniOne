from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v65_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v65_summary() -> dict:
    v64 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v64_20260321" / "summary.json"
    )
    hi_long = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_online_high_intensity_long_horizon_20260321" / "summary.json"
    )
    bridge_v9 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v9_20260321" / "summary.json"
    )

    hv = v64["headline_metrics"]
    hh = hi_long["headline_metrics"]
    hb = bridge_v9["headline_metrics"]

    feature_term_v65 = (
        hv["feature_term_v64"]
        + hv["feature_term_v64"] * hh["cumulative_language_keep"] * 0.004
        + hv["feature_term_v64"] * hb["plasticity_rule_alignment_v9"] * 0.001
    )
    structure_term_v65 = (
        hv["structure_term_v64"]
        + hv["structure_term_v64"] * hh["cumulative_structure_keep"] * 0.007
        + hv["structure_term_v64"] * hb["structure_rule_alignment_v9"] * 0.004
    )
    learning_term_v65 = (
        hv["learning_term_v64"]
        + hv["learning_term_v64"] * hb["topology_training_readiness_v9"]
        + hh["cumulative_margin"] * 1000.0
        + hh["cumulative_novel_gain"] * 1000.0
    )
    pressure_term_v65 = max(
        0.0,
        hv["pressure_term_v64"]
        + hb["topology_training_gap_v9"]
        + hh["cumulative_forgetting_penalty"]
        + hh["cumulative_instability_risk"] * 0.2,
    )
    encoding_margin_v65 = feature_term_v65 + structure_term_v65 + learning_term_v65 - pressure_term_v65

    return {
        "headline_metrics": {
            "feature_term_v65": feature_term_v65,
            "structure_term_v65": structure_term_v65,
            "learning_term_v65": learning_term_v65,
            "pressure_term_v65": pressure_term_v65,
            "encoding_margin_v65": encoding_margin_v65,
        },
        "closed_form_equation_v65": {
            "feature_term": "K_f_v65 = K_f_v64 + K_f_v64 * L_hi_long * 0.004 + K_f_v64 * B_plastic_v9 * 0.001",
            "structure_term": "K_s_v65 = K_s_v64 + K_s_v64 * S_hi_long * 0.007 + K_s_v64 * B_struct_v9 * 0.004",
            "learning_term": "K_l_v65 = K_l_v64 + K_l_v64 * R_train_v9 + M_hi_long * 1000 + G_hi_long * 1000",
            "pressure_term": "P_v65 = P_v64 + G_train_v9 + P_hi_long + 0.2 * I_hi_long",
            "margin_term": "M_encoding_v65 = K_f_v65 + K_s_v65 + K_l_v65 - P_v65",
        },
        "project_readout": {
            "summary": "第六十五版主核开始显式容纳更长时间尺度高强度更新下的累积性遗忘、结构保持和系统失稳风险，使主核更接近真实长期在线场景。",
            "next_question": "下一步要把这条主核继续放进更大对象集和更长上下文里，检验是否会出现相变式结构塌缩。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第六十五版报告",
        "",
        f"- feature_term_v65: {hm['feature_term_v65']:.6f}",
        f"- structure_term_v65: {hm['structure_term_v65']:.6f}",
        f"- learning_term_v65: {hm['learning_term_v65']:.6f}",
        f"- pressure_term_v65: {hm['pressure_term_v65']:.6f}",
        f"- encoding_margin_v65: {hm['encoding_margin_v65']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v65_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
