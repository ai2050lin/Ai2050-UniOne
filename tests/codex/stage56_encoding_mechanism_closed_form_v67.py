from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v67_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v67_summary() -> dict:
    v66 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v66_20260321" / "summary.json"
    )
    extreme = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_scale_high_intensity_long_horizon_extreme_20260321" / "summary.json"
    )
    bridge_v11 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v11_20260321" / "summary.json"
    )

    hv = v66["headline_metrics"]
    he = extreme["headline_metrics"]
    hb = bridge_v11["headline_metrics"]

    feature_term_v67 = (
        hv["feature_term_v66"]
        + hv["feature_term_v66"] * he["extreme_language_keep"] * 0.004
        + hv["feature_term_v66"] * hb["plasticity_rule_alignment_v11"] * 0.001
    )
    structure_term_v67 = (
        hv["structure_term_v66"]
        + hv["structure_term_v66"] * he["extreme_structure_keep"] * 0.007
        + hv["structure_term_v66"] * hb["structure_rule_alignment_v11"] * 0.004
    )
    learning_term_v67 = (
        hv["learning_term_v66"]
        + hv["learning_term_v66"] * hb["topology_training_readiness_v11"]
        + he["extreme_margin"] * 1000.0
        + he["extreme_novel_gain"] * 1000.0
    )
    pressure_term_v67 = max(
        0.0,
        hv["pressure_term_v66"]
        + hb["topology_training_gap_v11"]
        + he["extreme_forgetting_penalty"]
        + he["extreme_collapse_risk"] * 0.2,
    )
    encoding_margin_v67 = feature_term_v67 + structure_term_v67 + learning_term_v67 - pressure_term_v67

    return {
        "headline_metrics": {
            "feature_term_v67": feature_term_v67,
            "structure_term_v67": structure_term_v67,
            "learning_term_v67": learning_term_v67,
            "pressure_term_v67": pressure_term_v67,
            "encoding_margin_v67": encoding_margin_v67,
        },
        "closed_form_equation_v67": {
            "feature_term": "K_f_v67 = K_f_v66 + K_f_v66 * L_ext * 0.004 + K_f_v66 * B_plastic_v11 * 0.001",
            "structure_term": "K_s_v67 = K_s_v66 + K_s_v66 * S_ext * 0.007 + K_s_v66 * B_struct_v11 * 0.004",
            "learning_term": "K_l_v67 = K_l_v66 + K_l_v66 * R_train_v11 + M_ext * 1000 + G_ext * 1000",
            "pressure_term": "P_v67 = P_v66 + G_train_v11 + P_ext + 0.2 * R_ext",
            "margin_term": "M_encoding_v67 = K_f_v67 + K_s_v67 + K_l_v67 - P_v67",
        },
        "project_readout": {
            "summary": "第六十七版主核开始显式吸收最严苛的规模化高压长时场景，使主核更接近真实在线学习系统在极端条件下的受压状态。",
            "next_question": "下一步要把这条主核继续推进到真正更大的对象集和更长上下文原型里，检验是否会出现相变式结构塌缩。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第六十七版报告",
        "",
        f"- feature_term_v67: {hm['feature_term_v67']:.6f}",
        f"- structure_term_v67: {hm['structure_term_v67']:.6f}",
        f"- learning_term_v67: {hm['learning_term_v67']:.6f}",
        f"- pressure_term_v67: {hm['pressure_term_v67']:.6f}",
        f"- encoding_margin_v67: {hm['encoding_margin_v67']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v67_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
