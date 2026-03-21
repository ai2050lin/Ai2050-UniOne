from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v81_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v81_summary() -> dict:
    v80 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v80_20260321" / "summary.json"
    )
    steady = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_steady_amplification_validation_20260321" / "summary.json"
    )
    brain_v19 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v19_20260321" / "summary.json"
    )
    bridge_v25 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v25_20260321" / "summary.json"
    )

    hv = v80["headline_metrics"]
    hs = steady["headline_metrics"]
    hb = brain_v19["headline_metrics"]
    ht = bridge_v25["headline_metrics"]

    feature_term_v81 = (
        hv["feature_term_v80"]
        + hv["feature_term_v80"] * hs["steady_score"] * 0.004
        + hv["feature_term_v80"] * ht["plasticity_rule_alignment_v25"] * 0.001
        + hv["feature_term_v80"] * hb["direct_feature_measure_v19"] * 0.001
    )
    structure_term_v81 = (
        hv["structure_term_v80"]
        + hv["structure_term_v80"] * hs["steady_structure_stability"] * 0.007
        + hv["structure_term_v80"] * ht["structure_rule_alignment_v25"] * 0.004
        + hv["structure_term_v80"] * hb["direct_structure_measure_v19"] * 0.002
    )
    learning_term_v81 = (
        hv["learning_term_v80"]
        + hv["learning_term_v80"] * ht["topology_training_readiness_v25"]
        + hs["steady_margin"] * 1000.0
        + hs["steady_score"] * 1000.0
        + hb["direct_brain_measure_v19"] * 1000.0
    )
    pressure_term_v81 = max(
        0.0,
        hv["pressure_term_v80"]
        + ht["topology_training_gap_v25"]
        + hs["steady_residual_penalty"]
        + (1.0 - hs["steady_route_stability"]) * 0.2,
    )
    encoding_margin_v81 = feature_term_v81 + structure_term_v81 + learning_term_v81 - pressure_term_v81

    return {
        "headline_metrics": {
            "feature_term_v81": feature_term_v81,
            "structure_term_v81": structure_term_v81,
            "learning_term_v81": learning_term_v81,
            "pressure_term_v81": pressure_term_v81,
            "encoding_margin_v81": encoding_margin_v81,
        },
        "closed_form_equation_v81": {
            "feature_term": "K_f_v81 = K_f_v80 + K_f_v80 * S_steady_score * 0.004 + K_f_v80 * B_plastic_v25 * 0.001 + K_f_v80 * D_feature_v19 * 0.001",
            "structure_term": "K_s_v81 = K_s_v80 + K_s_v80 * S_steady * 0.007 + K_s_v80 * B_struct_v25 * 0.004 + K_s_v80 * D_structure_v19 * 0.002",
            "learning_term": "K_l_v81 = K_l_v80 + K_l_v80 * R_train_v25 + M_steady * 1000 + S_steady_score * 1000 + M_brain_direct_v19 * 1000",
            "pressure_term": "P_v81 = P_v80 + G_train_v25 + P_steady + 0.2 * (1 - R_steady)",
            "margin_term": "M_encoding_v81 = K_f_v81 + K_s_v81 + K_l_v81 - P_v81",
        },
        "project_readout": {
            "summary": "第八十一版主核开始把稳态放大验证、脑编码第十九版和训练终式第二十五桥一起并回主核，直接检测放大趋势是否开始稳态成立。",
            "next_question": "下一步要把这条主核推进到更大、更长、更高压的系统里，验证稳态放大是否真的开始成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第八十一版报告",
        "",
        f"- feature_term_v81: {hm['feature_term_v81']:.6f}",
        f"- structure_term_v81: {hm['structure_term_v81']:.6f}",
        f"- learning_term_v81: {hm['learning_term_v81']:.6f}",
        f"- pressure_term_v81: {hm['pressure_term_v81']:.6f}",
        f"- encoding_margin_v81: {hm['encoding_margin_v81']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v81_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
