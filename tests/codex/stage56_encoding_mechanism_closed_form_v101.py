from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v101_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v101_summary() -> dict:
    v100 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v100_20260321" / "summary.json"
    )
    field_fabric = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_fabric_validation_20260321" / "summary.json"
    )
    brain_v39 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v39_20260321" / "summary.json"
    )
    bridge_v45 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v45_20260321" / "summary.json"
    )

    hv = v100["headline_metrics"]
    hs = field_fabric["headline_metrics"]
    hb = brain_v39["headline_metrics"]
    ht = bridge_v45["headline_metrics"]

    feature_term_v101 = (
        hv["feature_term_v100"]
        + hv["feature_term_v100"] * hs["systemic_low_risk_field_fabric_score"] * 0.004
        + hv["feature_term_v100"] * ht["plasticity_rule_alignment_v45"] * 0.001
        + hv["feature_term_v100"] * hb["direct_feature_measure_v39"] * 0.001
    )
    structure_term_v101 = (
        hv["structure_term_v100"]
        + hv["structure_term_v100"] * hs["systemic_low_risk_field_structure_fabric"] * 0.007
        + hv["structure_term_v100"] * ht["structure_rule_alignment_v45"] * 0.004
        + hv["structure_term_v100"] * hb["direct_structure_measure_v39"] * 0.002
    )
    learning_term_v101 = (
        hv["learning_term_v100"]
        + hv["learning_term_v100"] * ht["topology_training_readiness_v45"]
        + hs["systemic_low_risk_field_fabric_margin"] * 1000.0
        + hs["systemic_low_risk_field_fabric_score"] * 1000.0
        + hb["direct_brain_measure_v39"] * 1000.0
    )
    pressure_term_v101 = max(
        0.0,
        hv["pressure_term_v100"]
        + ht["topology_training_gap_v45"]
        + hs["systemic_low_risk_field_fabric_penalty"]
        + (1.0 - hs["systemic_low_risk_field_route_fabric"]) * 0.2,
    )
    encoding_margin_v101 = feature_term_v101 + structure_term_v101 + learning_term_v101 - pressure_term_v101

    return {
        "headline_metrics": {
            "feature_term_v101": feature_term_v101,
            "structure_term_v101": structure_term_v101,
            "learning_term_v101": learning_term_v101,
            "pressure_term_v101": pressure_term_v101,
            "encoding_margin_v101": encoding_margin_v101,
        },
        "closed_form_equation_v101": {
            "feature_term": "K_f_v101 = K_f_v100 + K_f_v100 * S_sys_field_fabric_score * 0.004 + K_f_v100 * B_plastic_v45 * 0.001 + K_f_v100 * D_feature_v39 * 0.001",
            "structure_term": "K_s_v101 = K_s_v100 + K_s_v100 * S_sys_field_fabric * 0.007 + K_s_v100 * B_struct_v45 * 0.004 + K_s_v100 * D_structure_v39 * 0.002",
            "learning_term": "K_l_v101 = K_l_v100 + K_l_v100 * R_train_v45 + M_sys_field_fabric * 1000 + S_sys_field_fabric_score * 1000 + M_brain_direct_v39 * 1000",
            "pressure_term": "P_v101 = P_v100 + G_train_v45 + P_sys_field_fabric + 0.2 * (1 - R_sys_field_fabric)",
            "margin_term": "M_encoding_v101 = K_f_v101 + K_s_v101 + K_l_v101 - P_v101",
        },
        "project_readout": {
            "summary": "v101 checks whether systemic low-risk field mesh starts turning into a fabric-like low-risk field instead of only mesh-organizing the previous field.",
            "next_question": "next push this fabric-like low-risk field into larger and longer systems to test whether it keeps organizing toward a lower-risk steady topology.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Encoding Mechanism Closed Form V101 Report",
        "",
        f"- feature_term_v101: {hm['feature_term_v101']:.6f}",
        f"- structure_term_v101: {hm['structure_term_v101']:.6f}",
        f"- learning_term_v101: {hm['learning_term_v101']:.6f}",
        f"- pressure_term_v101: {hm['pressure_term_v101']:.6f}",
        f"- encoding_margin_v101: {hm['encoding_margin_v101']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v101_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
