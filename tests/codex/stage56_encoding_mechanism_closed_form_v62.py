from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v62_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v62_summary() -> dict:
    v61 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v61_20260321" / "summary.json"
    )
    curriculum = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_plasticity_curriculum_20260321" / "summary.json"
    )
    brain_v5 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v5_20260321" / "summary.json"
    )
    bridge_v6 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v6_20260321" / "summary.json"
    )

    hv = v61["headline_metrics"]
    hc = curriculum["headline_metrics"]
    hb = brain_v5["headline_metrics"]
    ht = bridge_v6["headline_metrics"]

    feature_term_v62 = (
        hv["feature_term_v61"]
        + hv["feature_term_v61"] * hb["direct_feature_measure_v5"] * 0.004
        + hv["feature_term_v61"] * hc["shared_route_guard"] * 0.002
    )
    structure_term_v62 = (
        hv["structure_term_v61"]
        + hv["structure_term_v61"] * hb["direct_structure_measure_v5"] * 0.007
        + hv["structure_term_v61"] * ht["structure_rule_alignment_v6"] * 0.004
    )
    learning_term_v62 = (
        hv["learning_term_v61"]
        + hv["learning_term_v61"] * ht["topology_training_readiness_v6"]
        + hc["plasticity_curriculum_margin"] * 1000.0
        + hb["direct_brain_measure_v5"] * 1000.0
    )
    pressure_term_v62 = max(
        0.0,
        hv["pressure_term_v61"]
        + ht["topology_training_gap_v6"]
        + max(0.0, 1.0 - hc["long_horizon_growth_v2"])
        + (1.0 - hb["direct_topology_alignment_v5"]) * 0.2,
    )
    encoding_margin_v62 = feature_term_v62 + structure_term_v62 + learning_term_v62 - pressure_term_v62

    return {
        "headline_metrics": {
            "feature_term_v62": feature_term_v62,
            "structure_term_v62": structure_term_v62,
            "learning_term_v62": learning_term_v62,
            "pressure_term_v62": pressure_term_v62,
            "encoding_margin_v62": encoding_margin_v62,
        },
        "closed_form_equation_v62": {
            "feature_term": "K_f_v62 = K_f_v61 + K_f_v61 * D_feature_v5 * 0.004 + K_f_v61 * H_curr * 0.002",
            "structure_term": "K_s_v62 = K_s_v61 + K_s_v61 * D_structure_v5 * 0.007 + K_s_v61 * B_struct_v6 * 0.004",
            "learning_term": "K_l_v62 = K_l_v61 + K_l_v61 * R_train_v6 + M_curr * 1000 + M_brain_direct_v5 * 1000",
            "pressure_term": "P_v62 = P_v61 + G_train_v6 + (1 - G_curr) + 0.2 * (1 - A_topo_v5)",
            "margin_term": "M_encoding_v62 = K_f_v62 + K_s_v62 + K_l_v62 - P_v62",
        },
        "project_readout": {
            "summary": "第六十二版主核把课程式可塑性、脑编码直测第五版和训练终式第六桥一起并回主式，使主核更接近“可持续增量学习 + 结构保持 + 规模化防塌缩”的统一表达。",
            "next_question": "下一步要把这条更完整的主核真正放进更大的在线原型，看它能不能在更高强度更新里保持语言能力和结构稳定。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第六十二版报告",
        "",
        f"- feature_term_v62: {hm['feature_term_v62']:.6f}",
        f"- structure_term_v62: {hm['structure_term_v62']:.6f}",
        f"- learning_term_v62: {hm['learning_term_v62']:.6f}",
        f"- pressure_term_v62: {hm['pressure_term_v62']:.6f}",
        f"- encoding_margin_v62: {hm['encoding_margin_v62']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v62_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
