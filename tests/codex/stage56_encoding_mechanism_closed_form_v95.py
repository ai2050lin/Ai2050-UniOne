from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v95_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v95_summary() -> dict:
    v94 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v94_20260321" / "summary.json"
    )
    systemic_field_stable = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_stabilization_validation_20260321" / "summary.json"
    )
    brain_v33 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v33_20260321" / "summary.json"
    )
    bridge_v39 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v39_20260321" / "summary.json"
    )

    hv = v94["headline_metrics"]
    hs = systemic_field_stable["headline_metrics"]
    hb = brain_v33["headline_metrics"]
    ht = bridge_v39["headline_metrics"]

    feature_term_v95 = (
        hv["feature_term_v94"]
        + hv["feature_term_v94"] * hs["systemic_low_risk_field_stability_score"] * 0.004
        + hv["feature_term_v94"] * ht["plasticity_rule_alignment_v39"] * 0.001
        + hv["feature_term_v94"] * hb["direct_feature_measure_v33"] * 0.001
    )
    structure_term_v95 = (
        hv["structure_term_v94"]
        + hv["structure_term_v94"] * hs["systemic_low_risk_field_structure_stability"] * 0.007
        + hv["structure_term_v94"] * ht["structure_rule_alignment_v39"] * 0.004
        + hv["structure_term_v94"] * hb["direct_structure_measure_v33"] * 0.002
    )
    learning_term_v95 = (
        hv["learning_term_v94"]
        + hv["learning_term_v94"] * ht["topology_training_readiness_v39"]
        + hs["systemic_low_risk_field_stability_margin"] * 1000.0
        + hs["systemic_low_risk_field_stability_score"] * 1000.0
        + hb["direct_brain_measure_v33"] * 1000.0
    )
    pressure_term_v95 = max(
        0.0,
        hv["pressure_term_v94"]
        + ht["topology_training_gap_v39"]
        + hs["systemic_low_risk_field_residual_penalty"]
        + (1.0 - hs["systemic_low_risk_field_route_stability"]) * 0.2,
    )
    encoding_margin_v95 = feature_term_v95 + structure_term_v95 + learning_term_v95 - pressure_term_v95

    return {
        "headline_metrics": {
            "feature_term_v95": feature_term_v95,
            "structure_term_v95": structure_term_v95,
            "learning_term_v95": learning_term_v95,
            "pressure_term_v95": pressure_term_v95,
            "encoding_margin_v95": encoding_margin_v95,
        },
        "closed_form_equation_v95": {
            "feature_term": "K_f_v95 = K_f_v94 + K_f_v94 * S_sys_field_stability_score * 0.004 + K_f_v94 * B_plastic_v39 * 0.001 + K_f_v94 * D_feature_v33 * 0.001",
            "structure_term": "K_s_v95 = K_s_v94 + K_s_v94 * S_sys_field_stable * 0.007 + K_s_v94 * B_struct_v39 * 0.004 + K_s_v94 * D_structure_v33 * 0.002",
            "learning_term": "K_l_v95 = K_l_v94 + K_l_v94 * R_train_v39 + M_sys_field_stable * 1000 + S_sys_field_stability_score * 1000 + M_brain_direct_v33 * 1000",
            "pressure_term": "P_v95 = P_v94 + G_train_v39 + P_sys_field_stable + 0.2 * (1 - R_sys_field_stable)",
            "margin_term": "M_encoding_v95 = K_f_v95 + K_s_v95 + K_l_v95 - P_v95",
        },
        "project_readout": {
            "summary": "v95 checks whether systemic low-risk field extension starts turning into a stabilized low-risk field instead of only extending the previous field.",
            "next_question": "next push this stabilized low-risk field into larger and longer systems to test whether it keeps stabilizing toward a wider low-risk steady field.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Encoding Mechanism Closed Form V95 Report",
        "",
        f"- feature_term_v95: {hm['feature_term_v95']:.6f}",
        f"- structure_term_v95: {hm['structure_term_v95']:.6f}",
        f"- learning_term_v95: {hm['learning_term_v95']:.6f}",
        f"- pressure_term_v95: {hm['pressure_term_v95']:.6f}",
        f"- encoding_margin_v95: {hm['encoding_margin_v95']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v95_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
