from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v82_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v82_summary() -> dict:
    v81 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v81_20260321" / "summary.json"
    )
    steady_plus = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_steady_amplification_reinforcement_20260321" / "summary.json"
    )
    brain_v20 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v20_20260321" / "summary.json"
    )
    bridge_v26 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v26_20260321" / "summary.json"
    )

    hv = v81["headline_metrics"]
    hs = steady_plus["headline_metrics"]
    hb = brain_v20["headline_metrics"]
    ht = bridge_v26["headline_metrics"]

    feature_term_v82 = (
        hv["feature_term_v81"]
        + hv["feature_term_v81"] * hs["steady_reinforcement_score"] * 0.004
        + hv["feature_term_v81"] * ht["plasticity_rule_alignment_v26"] * 0.001
        + hv["feature_term_v81"] * hb["direct_feature_measure_v20"] * 0.001
    )
    structure_term_v82 = (
        hv["structure_term_v81"]
        + hv["structure_term_v81"] * hs["steady_reinforcement_structure"] * 0.007
        + hv["structure_term_v81"] * ht["structure_rule_alignment_v26"] * 0.004
        + hv["structure_term_v81"] * hb["direct_structure_measure_v20"] * 0.002
    )
    learning_term_v82 = (
        hv["learning_term_v81"]
        + hv["learning_term_v81"] * ht["topology_training_readiness_v26"]
        + hs["steady_reinforcement_margin"] * 1000.0
        + hs["steady_reinforcement_score"] * 1000.0
        + hb["direct_brain_measure_v20"] * 1000.0
    )
    pressure_term_v82 = max(
        0.0,
        hv["pressure_term_v81"]
        + ht["topology_training_gap_v26"]
        + hs["steady_reinforcement_penalty"]
        + (1.0 - hs["steady_reinforcement_route"]) * 0.2,
    )
    encoding_margin_v82 = feature_term_v82 + structure_term_v82 + learning_term_v82 - pressure_term_v82

    return {
        "headline_metrics": {
            "feature_term_v82": feature_term_v82,
            "structure_term_v82": structure_term_v82,
            "learning_term_v82": learning_term_v82,
            "pressure_term_v82": pressure_term_v82,
            "encoding_margin_v82": encoding_margin_v82,
        },
        "closed_form_equation_v82": {
            "feature_term": "K_f_v82 = K_f_v81 + K_f_v81 * S_steady_plus_score * 0.004 + K_f_v81 * B_plastic_v26 * 0.001 + K_f_v81 * D_feature_v20 * 0.001",
            "structure_term": "K_s_v82 = K_s_v81 + K_s_v81 * S_steady_plus * 0.007 + K_s_v81 * B_struct_v26 * 0.004 + K_s_v81 * D_structure_v20 * 0.002",
            "learning_term": "K_l_v82 = K_l_v81 + K_l_v81 * R_train_v26 + M_steady_plus * 1000 + S_steady_plus_score * 1000 + M_brain_direct_v20 * 1000",
            "pressure_term": "P_v82 = P_v81 + G_train_v26 + P_steady_plus + 0.2 * (1 - R_steady_plus)",
            "margin_term": "M_encoding_v82 = K_f_v82 + K_s_v82 + K_l_v82 - P_v82",
        },
        "project_readout": {
            "summary": "第八十二版主核开始把更稳的放大强化、脑编码第二十版和训练终式第二十六桥一起并回主核，直接检验稳态放大是否继续增强。",
            "next_question": "下一步要把这条主核推进到更大、更长、更高压的系统里，验证稳态放大能否继续增强成系统级稳态放大。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第八十二版报告",
        "",
        f"- feature_term_v82: {hm['feature_term_v82']:.6f}",
        f"- structure_term_v82: {hm['structure_term_v82']:.6f}",
        f"- learning_term_v82: {hm['learning_term_v82']:.6f}",
        f"- pressure_term_v82: {hm['pressure_term_v82']:.6f}",
        f"- encoding_margin_v82: {hm['encoding_margin_v82']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v82_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
