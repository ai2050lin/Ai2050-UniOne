from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v97_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v97_summary() -> dict:
    v96 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v96_20260321" / "summary.json"
    )
    systemic_field_solid = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_solidification_validation_20260321" / "summary.json"
    )
    brain_v35 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v35_20260321" / "summary.json"
    )
    bridge_v41 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v41_20260321" / "summary.json"
    )

    hv = v96["headline_metrics"]
    hs = systemic_field_solid["headline_metrics"]
    hb = brain_v35["headline_metrics"]
    ht = bridge_v41["headline_metrics"]

    feature_term_v97 = (
        hv["feature_term_v96"]
        + hv["feature_term_v96"] * hs["systemic_low_risk_field_solidification_score"] * 0.004
        + hv["feature_term_v96"] * ht["plasticity_rule_alignment_v41"] * 0.001
        + hv["feature_term_v96"] * hb["direct_feature_measure_v35"] * 0.001
    )
    structure_term_v97 = (
        hv["structure_term_v96"]
        + hv["structure_term_v96"] * hs["systemic_low_risk_field_structure_solidification"] * 0.007
        + hv["structure_term_v96"] * ht["structure_rule_alignment_v41"] * 0.004
        + hv["structure_term_v96"] * hb["direct_structure_measure_v35"] * 0.002
    )
    learning_term_v97 = (
        hv["learning_term_v96"]
        + hv["learning_term_v96"] * ht["topology_training_readiness_v41"]
        + hs["systemic_low_risk_field_solidification_margin"] * 1000.0
        + hs["systemic_low_risk_field_solidification_score"] * 1000.0
        + hb["direct_brain_measure_v35"] * 1000.0
    )
    pressure_term_v97 = max(
        0.0,
        hv["pressure_term_v96"]
        + ht["topology_training_gap_v41"]
        + hs["systemic_low_risk_field_solidification_penalty"]
        + (1.0 - hs["systemic_low_risk_field_route_solidification"]) * 0.2,
    )
    encoding_margin_v97 = feature_term_v97 + structure_term_v97 + learning_term_v97 - pressure_term_v97

    return {
        "headline_metrics": {
            "feature_term_v97": feature_term_v97,
            "structure_term_v97": structure_term_v97,
            "learning_term_v97": learning_term_v97,
            "pressure_term_v97": pressure_term_v97,
            "encoding_margin_v97": encoding_margin_v97,
        },
        "closed_form_equation_v97": {
            "feature_term": "K_f_v97 = K_f_v96 + K_f_v96 * S_sys_field_solid_score * 0.004 + K_f_v96 * B_plastic_v41 * 0.001 + K_f_v96 * D_feature_v35 * 0.001",
            "structure_term": "K_s_v97 = K_s_v96 + K_s_v96 * S_sys_field_solid * 0.007 + K_s_v96 * B_struct_v41 * 0.004 + K_s_v96 * D_structure_v35 * 0.002",
            "learning_term": "K_l_v97 = K_l_v96 + K_l_v96 * R_train_v41 + M_sys_field_solid * 1000 + S_sys_field_solid_score * 1000 + M_brain_direct_v35 * 1000",
            "pressure_term": "P_v97 = P_v96 + G_train_v41 + P_sys_field_solid + 0.2 * (1 - R_sys_field_solid)",
            "margin_term": "M_encoding_v97 = K_f_v97 + K_s_v97 + K_l_v97 - P_v97",
        },
        "project_readout": {
            "summary": "v97 checks whether systemic low-risk field consolidation starts turning into a solidified low-risk field instead of only consolidating the previous field.",
            "next_question": "next push this solidified low-risk field into larger and longer systems to test whether it keeps solidifying toward a wider low-risk steady field.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Encoding Mechanism Closed Form V97 Report",
        "",
        f"- feature_term_v97: {hm['feature_term_v97']:.6f}",
        f"- structure_term_v97: {hm['structure_term_v97']:.6f}",
        f"- learning_term_v97: {hm['learning_term_v97']:.6f}",
        f"- pressure_term_v97: {hm['pressure_term_v97']:.6f}",
        f"- encoding_margin_v97: {hm['encoding_margin_v97']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v97_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
