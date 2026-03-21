from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v61_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v61_summary() -> dict:
    v60 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v60_20260321" / "summary.json"
    )
    reinforcement = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_plasticity_reinforcement_20260321" / "summary.json"
    )
    brain_v4 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v4_20260321" / "summary.json"
    )
    bridge_v5 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v5_20260321" / "summary.json"
    )

    hv = v60["headline_metrics"]
    hr = reinforcement["headline_metrics"]
    hd = brain_v4["headline_metrics"]
    hb = bridge_v5["headline_metrics"]

    feature_term_v61 = (
        hv["feature_term_v60"]
        + hv["feature_term_v60"] * hd["direct_feature_measure_v4"] * 0.005
        + hv["feature_term_v60"] * hr["shared_guard_reinforced"] * 0.002
    )
    structure_term_v61 = (
        hv["structure_term_v60"]
        + hv["structure_term_v60"] * hd["direct_structure_measure_v4"] * 0.008
        + hv["structure_term_v60"] * hb["structure_rule_alignment_v5"] * 0.004
    )
    learning_term_v61 = (
        hv["learning_term_v60"]
        + hv["learning_term_v60"] * hb["topology_training_readiness_v5"]
        + hr["plasticity_reinforced_margin"] * 1000.0
        + hd["direct_brain_measure_v4"] * 1000.0
    )
    pressure_term_v61 = max(
        0.0,
        hv["pressure_term_v60"]
        + hb["topology_training_gap_v5"]
        + max(0.0, 1.0 - hr["plastic_growth_readiness"])
        + hd["direct_brain_gap_v4"] * 0.1,
    )
    encoding_margin_v61 = feature_term_v61 + structure_term_v61 + learning_term_v61 - pressure_term_v61

    return {
        "headline_metrics": {
            "feature_term_v61": feature_term_v61,
            "structure_term_v61": structure_term_v61,
            "learning_term_v61": learning_term_v61,
            "pressure_term_v61": pressure_term_v61,
            "encoding_margin_v61": encoding_margin_v61,
        },
        "closed_form_equation_v61": {
            "feature_term": "K_f_v61 = K_f_v60 + K_f_v60 * D_feature_v4 * 0.005 + K_f_v60 * H_guard_plus * 0.002",
            "structure_term": "K_s_v61 = K_s_v60 + K_s_v60 * D_structure_v4 * 0.008 + K_s_v60 * B_struct_v5 * 0.004",
            "learning_term": "K_l_v61 = K_l_v60 + K_l_v60 * R_train_v5 + M_plasticity_plus * 1000 + M_brain_direct_v4 * 1000",
            "pressure_term": "P_v61 = P_v60 + G_train_v5 + (1 - R_growth_plus) + 0.1 * G_brain_v4",
            "margin_term": "M_encoding_v61 = K_f_v61 + K_s_v61 + K_l_v61 - P_v61",
        },
        "project_readout": {
            "summary": "第六十一版主核把长期可塑性强化、脑编码直测第四版和训练终式第五桥一起并回主式，使主核第一次更明确地同时容纳结构增强、长期增量学习和规模化防塌缩。",
            "next_question": "下一步要把这条更完整的训练桥放进更大的在线原型里，验证它在更高更新强度下是否还能保住结构和共享支路。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第六十一版报告",
        "",
        f"- feature_term_v61: {hm['feature_term_v61']:.6f}",
        f"- structure_term_v61: {hm['structure_term_v61']:.6f}",
        f"- learning_term_v61: {hm['learning_term_v61']:.6f}",
        f"- pressure_term_v61: {hm['pressure_term_v61']:.6f}",
        f"- encoding_margin_v61: {hm['encoding_margin_v61']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v61_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
