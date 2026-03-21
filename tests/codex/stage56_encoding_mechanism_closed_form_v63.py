from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v63_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v63_summary() -> dict:
    v62 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v62_20260321" / "summary.json"
    )
    large_online = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_online_prototype_validation_20260321" / "summary.json"
    )
    bridge_v7 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v7_20260321" / "summary.json"
    )
    brain_v5 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v5_20260321" / "summary.json"
    )

    hv = v62["headline_metrics"]
    hl = large_online["headline_metrics"]
    ht = bridge_v7["headline_metrics"]
    hb = brain_v5["headline_metrics"]

    feature_term_v63 = (
        hv["feature_term_v62"]
        + hv["feature_term_v62"] * hl["large_online_language_keep"] * 0.004
        + hv["feature_term_v62"] * hb["direct_feature_measure_v5"] * 0.002
    )
    structure_term_v63 = (
        hv["structure_term_v62"]
        + hv["structure_term_v62"] * hl["large_online_structure_keep"] * 0.007
        + hv["structure_term_v62"] * ht["structure_rule_alignment_v7"] * 0.004
    )
    learning_term_v63 = (
        hv["learning_term_v62"]
        + hv["learning_term_v62"] * ht["topology_training_readiness_v7"]
        + hl["large_online_margin"] * 1000.0
        + hl["large_online_novel_gain"] * 1000.0
    )
    pressure_term_v63 = max(
        0.0,
        hv["pressure_term_v62"]
        + ht["topology_training_gap_v7"]
        + hl["large_online_forgetting_penalty"]
        + (1.0 - hl["large_online_language_keep"]) * 0.2,
    )
    encoding_margin_v63 = feature_term_v63 + structure_term_v63 + learning_term_v63 - pressure_term_v63

    return {
        "headline_metrics": {
            "feature_term_v63": feature_term_v63,
            "structure_term_v63": structure_term_v63,
            "learning_term_v63": learning_term_v63,
            "pressure_term_v63": pressure_term_v63,
            "encoding_margin_v63": encoding_margin_v63,
        },
        "closed_form_equation_v63": {
            "feature_term": "K_f_v63 = K_f_v62 + K_f_v62 * L_large * 0.004 + K_f_v62 * D_feature_v5 * 0.002",
            "structure_term": "K_s_v63 = K_s_v62 + K_s_v62 * S_large * 0.007 + K_s_v62 * B_struct_v7 * 0.004",
            "learning_term": "K_l_v63 = K_l_v62 + K_l_v62 * R_train_v7 + M_large * 1000 + G_large * 1000",
            "pressure_term": "P_v63 = P_v62 + G_train_v7 + P_large + 0.2 * (1 - L_large)",
            "margin_term": "M_encoding_v63 = K_f_v63 + K_s_v63 + K_l_v63 - P_v63",
        },
        "project_readout": {
            "summary": "第六十三版主核把更大在线原型验证和训练终式第七桥一起并回主式，使主核第一次显式容纳更高更新强度下的语言保持、新知识增益和遗忘惩罚。",
            "next_question": "下一步要把这条主核真正放进更大在线原型训练过程，直接验证更高强度更新下是否会出现系统性失稳。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第六十三版报告",
        "",
        f"- feature_term_v63: {hm['feature_term_v63']:.6f}",
        f"- structure_term_v63: {hm['structure_term_v63']:.6f}",
        f"- learning_term_v63: {hm['learning_term_v63']:.6f}",
        f"- pressure_term_v63: {hm['pressure_term_v63']:.6f}",
        f"- encoding_margin_v63: {hm['encoding_margin_v63']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v63_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
