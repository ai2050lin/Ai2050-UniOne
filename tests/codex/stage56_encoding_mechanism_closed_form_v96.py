from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v96_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v96_summary() -> dict:
    v95 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v95_20260321" / "summary.json"
    )
    systemic_field_cons = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_consolidation_validation_20260321" / "summary.json"
    )
    brain_v34 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v34_20260321" / "summary.json"
    )
    bridge_v40 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v40_20260321" / "summary.json"
    )

    hv = v95["headline_metrics"]
    hs = systemic_field_cons["headline_metrics"]
    hb = brain_v34["headline_metrics"]
    ht = bridge_v40["headline_metrics"]

    feature_term_v96 = (
        hv["feature_term_v95"]
        + hv["feature_term_v95"] * hs["systemic_low_risk_field_consolidation_score"] * 0.004
        + hv["feature_term_v95"] * ht["plasticity_rule_alignment_v40"] * 0.001
        + hv["feature_term_v95"] * hb["direct_feature_measure_v34"] * 0.001
    )
    structure_term_v96 = (
        hv["structure_term_v95"]
        + hv["structure_term_v95"] * hs["systemic_low_risk_field_structure_consolidation"] * 0.007
        + hv["structure_term_v95"] * ht["structure_rule_alignment_v40"] * 0.004
        + hv["structure_term_v95"] * hb["direct_structure_measure_v34"] * 0.002
    )
    learning_term_v96 = (
        hv["learning_term_v95"]
        + hv["learning_term_v95"] * ht["topology_training_readiness_v40"]
        + hs["systemic_low_risk_field_consolidation_margin"] * 1000.0
        + hs["systemic_low_risk_field_consolidation_score"] * 1000.0
        + hb["direct_brain_measure_v34"] * 1000.0
    )
    pressure_term_v96 = max(
        0.0,
        hv["pressure_term_v95"]
        + ht["topology_training_gap_v40"]
        + hs["systemic_low_risk_field_consolidation_penalty"]
        + (1.0 - hs["systemic_low_risk_field_route_consolidation"]) * 0.2,
    )
    encoding_margin_v96 = feature_term_v96 + structure_term_v96 + learning_term_v96 - pressure_term_v96

    return {
        "headline_metrics": {
            "feature_term_v96": feature_term_v96,
            "structure_term_v96": structure_term_v96,
            "learning_term_v96": learning_term_v96,
            "pressure_term_v96": pressure_term_v96,
            "encoding_margin_v96": encoding_margin_v96,
        },
        "closed_form_equation_v96": {
            "feature_term": "K_f_v96 = K_f_v95 + K_f_v95 * S_sys_field_cons_score * 0.004 + K_f_v95 * B_plastic_v40 * 0.001 + K_f_v95 * D_feature_v34 * 0.001",
            "structure_term": "K_s_v96 = K_s_v95 + K_s_v95 * S_sys_field_cons * 0.007 + K_s_v95 * B_struct_v40 * 0.004 + K_s_v95 * D_structure_v34 * 0.002",
            "learning_term": "K_l_v96 = K_l_v95 + K_l_v95 * R_train_v40 + M_sys_field_cons * 1000 + S_sys_field_cons_score * 1000 + M_brain_direct_v34 * 1000",
            "pressure_term": "P_v96 = P_v95 + G_train_v40 + P_sys_field_cons + 0.2 * (1 - R_sys_field_cons)",
            "margin_term": "M_encoding_v96 = K_f_v96 + K_s_v96 + K_l_v96 - P_v96",
        },
        "project_readout": {
            "summary": "v96 checks whether systemic low-risk field stabilization starts turning into a consolidated low-risk field instead of only stabilizing the previous field.",
            "next_question": "next push this consolidated low-risk field into larger and longer systems to test whether it keeps consolidating toward a wider low-risk steady field.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Encoding Mechanism Closed Form V96 Report",
        "",
        f"- feature_term_v96: {hm['feature_term_v96']:.6f}",
        f"- structure_term_v96: {hm['structure_term_v96']:.6f}",
        f"- learning_term_v96: {hm['learning_term_v96']:.6f}",
        f"- pressure_term_v96: {hm['pressure_term_v96']:.6f}",
        f"- encoding_margin_v96: {hm['encoding_margin_v96']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v96_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
