from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v66_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v66_summary() -> dict:
    v65 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v65_20260321" / "summary.json"
    )
    scale_ctx = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_scale_long_context_online_validation_20260321" / "summary.json"
    )
    bridge_v10 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v10_20260321" / "summary.json"
    )

    hv = v65["headline_metrics"]
    hs = scale_ctx["headline_metrics"]
    hb = bridge_v10["headline_metrics"]

    feature_term_v66 = (
        hv["feature_term_v65"]
        + hv["feature_term_v65"] * hs["scale_language_keep"] * 0.004
        + hv["feature_term_v65"] * hb["plasticity_rule_alignment_v10"] * 0.001
    )
    structure_term_v66 = (
        hv["structure_term_v65"]
        + hv["structure_term_v65"] * hs["scale_structure_keep"] * 0.007
        + hv["structure_term_v65"] * hb["structure_rule_alignment_v10"] * 0.004
    )
    learning_term_v66 = (
        hv["learning_term_v65"]
        + hv["learning_term_v65"] * hb["topology_training_readiness_v10"]
        + hs["scale_margin"] * 1000.0
        + hs["scale_novel_gain"] * 1000.0
    )
    pressure_term_v66 = max(
        0.0,
        hv["pressure_term_v65"]
        + hb["topology_training_gap_v10"]
        + hs["scale_forgetting_penalty"]
        + hs["scale_collapse_risk"] * 0.2,
    )
    encoding_margin_v66 = feature_term_v66 + structure_term_v66 + learning_term_v66 - pressure_term_v66

    return {
        "headline_metrics": {
            "feature_term_v66": feature_term_v66,
            "structure_term_v66": structure_term_v66,
            "learning_term_v66": learning_term_v66,
            "pressure_term_v66": pressure_term_v66,
            "encoding_margin_v66": encoding_margin_v66,
        },
        "closed_form_equation_v66": {
            "feature_term": "K_f_v66 = K_f_v65 + K_f_v65 * L_scale * 0.004 + K_f_v65 * B_plastic_v10 * 0.001",
            "structure_term": "K_s_v66 = K_s_v65 + K_s_v65 * S_scale * 0.007 + K_s_v65 * B_struct_v10 * 0.004",
            "learning_term": "K_l_v66 = K_l_v65 + K_l_v65 * R_train_v10 + M_scale * 1000 + G_scale * 1000",
            "pressure_term": "P_v66 = P_v65 + G_train_v10 + P_scale + 0.2 * R_scale",
            "margin_term": "M_encoding_v66 = K_f_v66 + K_s_v66 + K_l_v66 - P_v66",
        },
        "project_readout": {
            "summary": "第六十六版主核开始显式吸收更大对象集和长上下文场景下的语言保持、结构保持、长程泛化、遗忘惩罚和系统塌缩风险，使主核更接近规模化在线学习场景。",
            "next_question": "下一步要把这条主核推进到真正更大的对象集与更长上下文原型里，检验是否会出现相变式结构塌缩。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第六十六版报告",
        "",
        f"- feature_term_v66: {hm['feature_term_v66']:.6f}",
        f"- structure_term_v66: {hm['structure_term_v66']:.6f}",
        f"- learning_term_v66: {hm['learning_term_v66']:.6f}",
        f"- pressure_term_v66: {hm['pressure_term_v66']:.6f}",
        f"- encoding_margin_v66: {hm['encoding_margin_v66']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v66_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
