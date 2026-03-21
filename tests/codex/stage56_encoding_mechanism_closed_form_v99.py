from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v99_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v99_summary() -> dict:
    v98 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v98_20260321" / "summary.json"
    )
    field_lattice = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_lattice_validation_20260321" / "summary.json"
    )
    brain_v37 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v37_20260321" / "summary.json"
    )
    bridge_v43 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v43_20260321" / "summary.json"
    )

    hv = v98["headline_metrics"]
    hs = field_lattice["headline_metrics"]
    hb = brain_v37["headline_metrics"]
    ht = bridge_v43["headline_metrics"]

    feature_term_v99 = (
        hv["feature_term_v98"]
        + hv["feature_term_v98"] * hs["systemic_low_risk_field_lattice_score"] * 0.004
        + hv["feature_term_v98"] * ht["plasticity_rule_alignment_v43"] * 0.001
        + hv["feature_term_v98"] * hb["direct_feature_measure_v37"] * 0.001
    )
    structure_term_v99 = (
        hv["structure_term_v98"]
        + hv["structure_term_v98"] * hs["systemic_low_risk_field_structure_lattice"] * 0.007
        + hv["structure_term_v98"] * ht["structure_rule_alignment_v43"] * 0.004
        + hv["structure_term_v98"] * hb["direct_structure_measure_v37"] * 0.002
    )
    learning_term_v99 = (
        hv["learning_term_v98"]
        + hv["learning_term_v98"] * ht["topology_training_readiness_v43"]
        + hs["systemic_low_risk_field_lattice_margin"] * 1000.0
        + hs["systemic_low_risk_field_lattice_score"] * 1000.0
        + hb["direct_brain_measure_v37"] * 1000.0
    )
    pressure_term_v99 = max(
        0.0,
        hv["pressure_term_v98"]
        + ht["topology_training_gap_v43"]
        + hs["systemic_low_risk_field_lattice_penalty"]
        + (1.0 - hs["systemic_low_risk_field_route_lattice"]) * 0.2,
    )
    encoding_margin_v99 = feature_term_v99 + structure_term_v99 + learning_term_v99 - pressure_term_v99

    return {
        "headline_metrics": {
            "feature_term_v99": feature_term_v99,
            "structure_term_v99": structure_term_v99,
            "learning_term_v99": learning_term_v99,
            "pressure_term_v99": pressure_term_v99,
            "encoding_margin_v99": encoding_margin_v99,
        },
        "closed_form_equation_v99": {
            "feature_term": "K_f_v99 = K_f_v98 + K_f_v98 * S_sys_field_lattice_score * 0.004 + K_f_v98 * B_plastic_v43 * 0.001 + K_f_v98 * D_feature_v37 * 0.001",
            "structure_term": "K_s_v99 = K_s_v98 + K_s_v98 * S_sys_field_lattice * 0.007 + K_s_v98 * B_struct_v43 * 0.004 + K_s_v98 * D_structure_v37 * 0.002",
            "learning_term": "K_l_v99 = K_l_v98 + K_l_v98 * R_train_v43 + M_sys_field_lattice * 1000 + S_sys_field_lattice_score * 1000 + M_brain_direct_v37 * 1000",
            "pressure_term": "P_v99 = P_v98 + G_train_v43 + P_sys_field_lattice + 0.2 * (1 - R_sys_field_lattice)",
            "margin_term": "M_encoding_v99 = K_f_v99 + K_s_v99 + K_l_v99 - P_v99",
        },
        "project_readout": {
            "summary": "v99 checks whether systemic low-risk field crystallization starts turning into a lattice-like low-risk field instead of only crystallizing the previous field.",
            "next_question": "next push this lattice-like low-risk field into larger and longer systems to test whether it keeps organizing toward a lower-risk steady field.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Encoding Mechanism Closed Form V99 Report",
        "",
        f"- feature_term_v99: {hm['feature_term_v99']:.6f}",
        f"- structure_term_v99: {hm['structure_term_v99']:.6f}",
        f"- learning_term_v99: {hm['learning_term_v99']:.6f}",
        f"- pressure_term_v99: {hm['pressure_term_v99']:.6f}",
        f"- encoding_margin_v99: {hm['encoding_margin_v99']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v99_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
