from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v94_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v94_summary() -> dict:
    v93 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v93_20260321" / "summary.json"
    )
    systemic_field = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_extension_validation_20260321" / "summary.json"
    )
    brain_v32 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v32_20260321" / "summary.json"
    )
    bridge_v38 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v38_20260321" / "summary.json"
    )

    hv = v93["headline_metrics"]
    hs = systemic_field["headline_metrics"]
    hb = brain_v32["headline_metrics"]
    ht = bridge_v38["headline_metrics"]

    feature_term_v94 = (
        hv["feature_term_v93"]
        + hv["feature_term_v93"] * hs["systemic_low_risk_field_score"] * 0.004
        + hv["feature_term_v93"] * ht["plasticity_rule_alignment_v38"] * 0.001
        + hv["feature_term_v93"] * hb["direct_feature_measure_v32"] * 0.001
    )
    structure_term_v94 = (
        hv["structure_term_v93"]
        + hv["structure_term_v93"] * hs["systemic_low_risk_field_structure"] * 0.007
        + hv["structure_term_v93"] * ht["structure_rule_alignment_v38"] * 0.004
        + hv["structure_term_v93"] * hb["direct_structure_measure_v32"] * 0.002
    )
    learning_term_v94 = (
        hv["learning_term_v93"]
        + hv["learning_term_v93"] * ht["topology_training_readiness_v38"]
        + hs["systemic_low_risk_field_margin"] * 1000.0
        + hs["systemic_low_risk_field_score"] * 1000.0
        + hb["direct_brain_measure_v32"] * 1000.0
    )
    pressure_term_v94 = max(
        0.0,
        hv["pressure_term_v93"]
        + ht["topology_training_gap_v38"]
        + hs["systemic_low_risk_field_penalty"]
        + (1.0 - hs["systemic_low_risk_field_route"]) * 0.2,
    )
    encoding_margin_v94 = feature_term_v94 + structure_term_v94 + learning_term_v94 - pressure_term_v94

    return {
        "headline_metrics": {
            "feature_term_v94": feature_term_v94,
            "structure_term_v94": structure_term_v94,
            "learning_term_v94": learning_term_v94,
            "pressure_term_v94": pressure_term_v94,
            "encoding_margin_v94": encoding_margin_v94,
        },
        "closed_form_equation_v94": {
            "feature_term": "K_f_v94 = K_f_v93 + K_f_v93 * S_sys_field_score * 0.004 + K_f_v93 * B_plastic_v38 * 0.001 + K_f_v93 * D_feature_v32 * 0.001",
            "structure_term": "K_s_v94 = K_s_v93 + K_s_v93 * S_sys_field * 0.007 + K_s_v93 * B_struct_v38 * 0.004 + K_s_v93 * D_structure_v32 * 0.002",
            "learning_term": "K_l_v94 = K_l_v93 + K_l_v93 * R_train_v38 + M_sys_field * 1000 + S_sys_field_score * 1000 + M_brain_direct_v32 * 1000",
            "pressure_term": "P_v94 = P_v93 + G_train_v38 + P_sys_field + 0.2 * (1 - R_sys_field)",
            "margin_term": "M_encoding_v94 = K_f_v94 + K_s_v94 + K_l_v94 - P_v94",
        },
        "project_readout": {
            "summary": "v94 checks whether systemic low-risk field extension starts turning into a broader and more connected low-risk field instead of only extending the previous band.",
            "next_question": "next push this broader low-risk field into larger and longer systems to test whether it keeps extending toward a wider low-risk steady field.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Encoding Mechanism Closed Form V94 Report",
        "",
        f"- feature_term_v94: {hm['feature_term_v94']:.6f}",
        f"- structure_term_v94: {hm['structure_term_v94']:.6f}",
        f"- learning_term_v94: {hm['learning_term_v94']:.6f}",
        f"- pressure_term_v94: {hm['pressure_term_v94']:.6f}",
        f"- encoding_margin_v94: {hm['encoding_margin_v94']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v94_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
