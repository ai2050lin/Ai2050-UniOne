from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v89_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v89_summary() -> dict:
    v88 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v88_20260321" / "summary.json"
    )
    expansion = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_low_risk_steady_zone_expansion_validation_20260321" / "summary.json"
    )
    brain_v27 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v27_20260321" / "summary.json"
    )
    bridge_v33 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v33_20260321" / "summary.json"
    )

    hv = v88["headline_metrics"]
    he = expansion["headline_metrics"]
    hb = brain_v27["headline_metrics"]
    ht = bridge_v33["headline_metrics"]

    feature_term_v89 = (
        hv["feature_term_v88"]
        + hv["feature_term_v88"] * he["low_risk_expansion_score"] * 0.004
        + hv["feature_term_v88"] * ht["plasticity_rule_alignment_v33"] * 0.001
        + hv["feature_term_v88"] * hb["direct_feature_measure_v27"] * 0.001
    )
    structure_term_v89 = (
        hv["structure_term_v88"]
        + hv["structure_term_v88"] * he["low_risk_expansion_structure"] * 0.007
        + hv["structure_term_v88"] * ht["structure_rule_alignment_v33"] * 0.004
        + hv["structure_term_v88"] * hb["direct_structure_measure_v27"] * 0.002
    )
    learning_term_v89 = (
        hv["learning_term_v88"]
        + hv["learning_term_v88"] * ht["topology_training_readiness_v33"]
        + he["low_risk_expansion_margin"] * 1000.0
        + he["low_risk_expansion_score"] * 1000.0
        + hb["direct_brain_measure_v27"] * 1000.0
    )
    pressure_term_v89 = max(
        0.0,
        hv["pressure_term_v88"]
        + ht["topology_training_gap_v33"]
        + he["low_risk_expansion_penalty"]
        + (1.0 - he["low_risk_expansion_route"]) * 0.2,
    )
    encoding_margin_v89 = feature_term_v89 + structure_term_v89 + learning_term_v89 - pressure_term_v89

    return {
        "headline_metrics": {
            "feature_term_v89": feature_term_v89,
            "structure_term_v89": structure_term_v89,
            "learning_term_v89": learning_term_v89,
            "pressure_term_v89": pressure_term_v89,
            "encoding_margin_v89": encoding_margin_v89,
        },
        "closed_form_equation_v89": {
            "feature_term": "K_f_v89 = K_f_v88 + K_f_v88 * S_expand_score * 0.004 + K_f_v88 * B_plastic_v33 * 0.001 + K_f_v88 * D_feature_v27 * 0.001",
            "structure_term": "K_s_v89 = K_s_v88 + K_s_v88 * S_expand * 0.007 + K_s_v88 * B_struct_v33 * 0.004 + K_s_v88 * D_structure_v27 * 0.002",
            "learning_term": "K_l_v89 = K_l_v88 + K_l_v88 * R_train_v33 + M_expand * 1000 + S_expand_score * 1000 + M_brain_direct_v27 * 1000",
            "pressure_term": "P_v89 = P_v88 + G_train_v33 + P_expand + 0.2 * (1 - R_expand)",
            "margin_term": "M_encoding_v89 = K_f_v89 + K_s_v89 + K_l_v89 - P_v89",
        },
        "project_readout": {
            "summary": "v89 checks whether the low-risk steady zone starts expanding into a clearer systemic low-risk steady regime instead of only holding local closure.",
            "next_question": "next push this expanded low-risk regime into larger and longer systems to test whether it keeps enlarging.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Encoding Mechanism Closed Form V89 Report",
        "",
        f"- feature_term_v89: {hm['feature_term_v89']:.6f}",
        f"- structure_term_v89: {hm['structure_term_v89']:.6f}",
        f"- learning_term_v89: {hm['learning_term_v89']:.6f}",
        f"- pressure_term_v89: {hm['pressure_term_v89']:.6f}",
        f"- encoding_margin_v89: {hm['encoding_margin_v89']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v89_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
