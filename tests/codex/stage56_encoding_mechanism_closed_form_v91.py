from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v91_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v91_summary() -> dict:
    v90 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v90_20260321" / "summary.json"
    )
    systemic_enlarge = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_zone_enlargement_validation_20260321" / "summary.json"
    )
    brain_v29 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v29_20260321" / "summary.json"
    )
    bridge_v35 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v35_20260321" / "summary.json"
    )

    hv = v90["headline_metrics"]
    hs = systemic_enlarge["headline_metrics"]
    hb = brain_v29["headline_metrics"]
    ht = bridge_v35["headline_metrics"]

    feature_term_v91 = (
        hv["feature_term_v90"]
        + hv["feature_term_v90"] * hs["systemic_low_risk_enlargement_score"] * 0.004
        + hv["feature_term_v90"] * ht["plasticity_rule_alignment_v35"] * 0.001
        + hv["feature_term_v90"] * hb["direct_feature_measure_v29"] * 0.001
    )
    structure_term_v91 = (
        hv["structure_term_v90"]
        + hv["structure_term_v90"] * hs["systemic_low_risk_enlargement_structure"] * 0.007
        + hv["structure_term_v90"] * ht["structure_rule_alignment_v35"] * 0.004
        + hv["structure_term_v90"] * hb["direct_structure_measure_v29"] * 0.002
    )
    learning_term_v91 = (
        hv["learning_term_v90"]
        + hv["learning_term_v90"] * ht["topology_training_readiness_v35"]
        + hs["systemic_low_risk_enlargement_margin"] * 1000.0
        + hs["systemic_low_risk_enlargement_score"] * 1000.0
        + hb["direct_brain_measure_v29"] * 1000.0
    )
    pressure_term_v91 = max(
        0.0,
        hv["pressure_term_v90"]
        + ht["topology_training_gap_v35"]
        + hs["systemic_low_risk_enlargement_penalty"]
        + (1.0 - hs["systemic_low_risk_enlargement_route"]) * 0.2,
    )
    encoding_margin_v91 = feature_term_v91 + structure_term_v91 + learning_term_v91 - pressure_term_v91

    return {
        "headline_metrics": {
            "feature_term_v91": feature_term_v91,
            "structure_term_v91": structure_term_v91,
            "learning_term_v91": learning_term_v91,
            "pressure_term_v91": pressure_term_v91,
            "encoding_margin_v91": encoding_margin_v91,
        },
        "closed_form_equation_v91": {
            "feature_term": "K_f_v91 = K_f_v90 + K_f_v90 * S_sys_enlarge_score * 0.004 + K_f_v90 * B_plastic_v35 * 0.001 + K_f_v90 * D_feature_v29 * 0.001",
            "structure_term": "K_s_v91 = K_s_v90 + K_s_v90 * S_sys_enlarge * 0.007 + K_s_v90 * B_struct_v35 * 0.004 + K_s_v90 * D_structure_v29 * 0.002",
            "learning_term": "K_l_v91 = K_l_v90 + K_l_v90 * R_train_v35 + M_sys_enlarge * 1000 + S_sys_enlarge_score * 1000 + M_brain_direct_v29 * 1000",
            "pressure_term": "P_v91 = P_v90 + G_train_v35 + P_sys_enlarge + 0.2 * (1 - R_sys_enlarge)",
            "margin_term": "M_encoding_v91 = K_f_v91 + K_s_v91 + K_l_v91 - P_v91",
        },
        "project_readout": {
            "summary": "v91 checks whether systemic low-risk enlargement starts turning into a broader and more explicit low-risk regime instead of only enlarging the previous expansion shell.",
            "next_question": "next push this enlarged low-risk regime into larger and longer systems to test whether it keeps broadening toward a wider low-risk zone.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Encoding Mechanism Closed Form V91 Report",
        "",
        f"- feature_term_v91: {hm['feature_term_v91']:.6f}",
        f"- structure_term_v91: {hm['structure_term_v91']:.6f}",
        f"- learning_term_v91: {hm['learning_term_v91']:.6f}",
        f"- pressure_term_v91: {hm['pressure_term_v91']:.6f}",
        f"- encoding_margin_v91: {hm['encoding_margin_v91']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v91_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
