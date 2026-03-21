from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v98_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v98_summary() -> dict:
    v97 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v97_20260321" / "summary.json"
    )
    field_crystal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_crystallization_validation_20260321" / "summary.json"
    )
    brain_v36 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v36_20260321" / "summary.json"
    )
    bridge_v42 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v42_20260321" / "summary.json"
    )

    hv = v97["headline_metrics"]
    hs = field_crystal["headline_metrics"]
    hb = brain_v36["headline_metrics"]
    ht = bridge_v42["headline_metrics"]

    feature_term_v98 = (
        hv["feature_term_v97"]
        + hv["feature_term_v97"] * hs["systemic_low_risk_field_crystallization_score"] * 0.004
        + hv["feature_term_v97"] * ht["plasticity_rule_alignment_v42"] * 0.001
        + hv["feature_term_v97"] * hb["direct_feature_measure_v36"] * 0.001
    )
    structure_term_v98 = (
        hv["structure_term_v97"]
        + hv["structure_term_v97"] * hs["systemic_low_risk_field_structure_crystallization"] * 0.007
        + hv["structure_term_v97"] * ht["structure_rule_alignment_v42"] * 0.004
        + hv["structure_term_v97"] * hb["direct_structure_measure_v36"] * 0.002
    )
    learning_term_v98 = (
        hv["learning_term_v97"]
        + hv["learning_term_v97"] * ht["topology_training_readiness_v42"]
        + hs["systemic_low_risk_field_crystallization_margin"] * 1000.0
        + hs["systemic_low_risk_field_crystallization_score"] * 1000.0
        + hb["direct_brain_measure_v36"] * 1000.0
    )
    pressure_term_v98 = max(
        0.0,
        hv["pressure_term_v97"]
        + ht["topology_training_gap_v42"]
        + hs["systemic_low_risk_field_crystallization_penalty"]
        + (1.0 - hs["systemic_low_risk_field_route_crystallization"]) * 0.2,
    )
    encoding_margin_v98 = feature_term_v98 + structure_term_v98 + learning_term_v98 - pressure_term_v98

    return {
        "headline_metrics": {
            "feature_term_v98": feature_term_v98,
            "structure_term_v98": structure_term_v98,
            "learning_term_v98": learning_term_v98,
            "pressure_term_v98": pressure_term_v98,
            "encoding_margin_v98": encoding_margin_v98,
        },
        "closed_form_equation_v98": {
            "feature_term": "K_f_v98 = K_f_v97 + K_f_v97 * S_sys_field_crystal_score * 0.004 + K_f_v97 * B_plastic_v42 * 0.001 + K_f_v97 * D_feature_v36 * 0.001",
            "structure_term": "K_s_v98 = K_s_v97 + K_s_v97 * S_sys_field_crystal * 0.007 + K_s_v97 * B_struct_v42 * 0.004 + K_s_v97 * D_structure_v36 * 0.002",
            "learning_term": "K_l_v98 = K_l_v97 + K_l_v97 * R_train_v42 + M_sys_field_crystal * 1000 + S_sys_field_crystal_score * 1000 + M_brain_direct_v36 * 1000",
            "pressure_term": "P_v98 = P_v97 + G_train_v42 + P_sys_field_crystal + 0.2 * (1 - R_sys_field_crystal)",
            "margin_term": "M_encoding_v98 = K_f_v98 + K_s_v98 + K_l_v98 - P_v98",
        },
        "project_readout": {
            "summary": "v98 checks whether systemic low-risk field solidification starts turning into a crystallized low-risk field instead of only solidifying the previous field.",
            "next_question": "next push this crystallized low-risk field into larger and longer systems to test whether it keeps crystallizing toward a lower-risk steady field.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Encoding Mechanism Closed Form V98 Report",
        "",
        f"- feature_term_v98: {hm['feature_term_v98']:.6f}",
        f"- structure_term_v98: {hm['structure_term_v98']:.6f}",
        f"- learning_term_v98: {hm['learning_term_v98']:.6f}",
        f"- pressure_term_v98: {hm['pressure_term_v98']:.6f}",
        f"- encoding_margin_v98: {hm['encoding_margin_v98']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v98_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
