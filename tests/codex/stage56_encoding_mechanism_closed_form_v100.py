from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v100_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v100_summary() -> dict:
    v99 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v99_20260321" / "summary.json"
    )
    field_mesh = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_mesh_validation_20260321" / "summary.json"
    )
    brain_v38 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v38_20260321" / "summary.json"
    )
    bridge_v44 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v44_20260321" / "summary.json"
    )

    hv = v99["headline_metrics"]
    hs = field_mesh["headline_metrics"]
    hb = brain_v38["headline_metrics"]
    ht = bridge_v44["headline_metrics"]

    feature_term_v100 = (
        hv["feature_term_v99"]
        + hv["feature_term_v99"] * hs["systemic_low_risk_field_mesh_score"] * 0.004
        + hv["feature_term_v99"] * ht["plasticity_rule_alignment_v44"] * 0.001
        + hv["feature_term_v99"] * hb["direct_feature_measure_v38"] * 0.001
    )
    structure_term_v100 = (
        hv["structure_term_v99"]
        + hv["structure_term_v99"] * hs["systemic_low_risk_field_structure_mesh"] * 0.007
        + hv["structure_term_v99"] * ht["structure_rule_alignment_v44"] * 0.004
        + hv["structure_term_v99"] * hb["direct_structure_measure_v38"] * 0.002
    )
    learning_term_v100 = (
        hv["learning_term_v99"]
        + hv["learning_term_v99"] * ht["topology_training_readiness_v44"]
        + hs["systemic_low_risk_field_mesh_margin"] * 1000.0
        + hs["systemic_low_risk_field_mesh_score"] * 1000.0
        + hb["direct_brain_measure_v38"] * 1000.0
    )
    pressure_term_v100 = max(
        0.0,
        hv["pressure_term_v99"]
        + ht["topology_training_gap_v44"]
        + hs["systemic_low_risk_field_mesh_penalty"]
        + (1.0 - hs["systemic_low_risk_field_route_mesh"]) * 0.2,
    )
    encoding_margin_v100 = feature_term_v100 + structure_term_v100 + learning_term_v100 - pressure_term_v100

    return {
        "headline_metrics": {
            "feature_term_v100": feature_term_v100,
            "structure_term_v100": structure_term_v100,
            "learning_term_v100": learning_term_v100,
            "pressure_term_v100": pressure_term_v100,
            "encoding_margin_v100": encoding_margin_v100,
        },
        "closed_form_equation_v100": {
            "feature_term": "K_f_v100 = K_f_v99 + K_f_v99 * S_sys_field_mesh_score * 0.004 + K_f_v99 * B_plastic_v44 * 0.001 + K_f_v99 * D_feature_v38 * 0.001",
            "structure_term": "K_s_v100 = K_s_v99 + K_s_v99 * S_sys_field_mesh * 0.007 + K_s_v99 * B_struct_v44 * 0.004 + K_s_v99 * D_structure_v38 * 0.002",
            "learning_term": "K_l_v100 = K_l_v99 + K_l_v99 * R_train_v44 + M_sys_field_mesh * 1000 + S_sys_field_mesh_score * 1000 + M_brain_direct_v38 * 1000",
            "pressure_term": "P_v100 = P_v99 + G_train_v44 + P_sys_field_mesh + 0.2 * (1 - R_sys_field_mesh)",
            "margin_term": "M_encoding_v100 = K_f_v100 + K_s_v100 + K_l_v100 - P_v100",
        },
        "project_readout": {
            "summary": "v100 checks whether systemic low-risk field lattice starts turning into a mesh-like low-risk field instead of only lattice-organizing the previous field.",
            "next_question": "next push this mesh-like low-risk field into larger and longer systems to test whether it keeps organizing toward a lower-risk steady topology.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Encoding Mechanism Closed Form V100 Report",
        "",
        f"- feature_term_v100: {hm['feature_term_v100']:.6f}",
        f"- structure_term_v100: {hm['structure_term_v100']:.6f}",
        f"- learning_term_v100: {hm['learning_term_v100']:.6f}",
        f"- pressure_term_v100: {hm['pressure_term_v100']:.6f}",
        f"- encoding_margin_v100: {hm['encoding_margin_v100']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v100_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
