from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v92_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v92_summary() -> dict:
    v91 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v91_20260321" / "summary.json"
    )
    systemic_broad = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_zone_broadening_validation_20260321" / "summary.json"
    )
    brain_v30 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v30_20260321" / "summary.json"
    )
    bridge_v36 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v36_20260321" / "summary.json"
    )

    hv = v91["headline_metrics"]
    hs = systemic_broad["headline_metrics"]
    hb = brain_v30["headline_metrics"]
    ht = bridge_v36["headline_metrics"]

    feature_term_v92 = (
        hv["feature_term_v91"]
        + hv["feature_term_v91"] * hs["systemic_low_risk_broadening_score"] * 0.004
        + hv["feature_term_v91"] * ht["plasticity_rule_alignment_v36"] * 0.001
        + hv["feature_term_v91"] * hb["direct_feature_measure_v30"] * 0.001
    )
    structure_term_v92 = (
        hv["structure_term_v91"]
        + hv["structure_term_v91"] * hs["systemic_low_risk_broadening_structure"] * 0.007
        + hv["structure_term_v91"] * ht["structure_rule_alignment_v36"] * 0.004
        + hv["structure_term_v91"] * hb["direct_structure_measure_v30"] * 0.002
    )
    learning_term_v92 = (
        hv["learning_term_v91"]
        + hv["learning_term_v91"] * ht["topology_training_readiness_v36"]
        + hs["systemic_low_risk_broadening_margin"] * 1000.0
        + hs["systemic_low_risk_broadening_score"] * 1000.0
        + hb["direct_brain_measure_v30"] * 1000.0
    )
    pressure_term_v92 = max(
        0.0,
        hv["pressure_term_v91"]
        + ht["topology_training_gap_v36"]
        + hs["systemic_low_risk_broadening_penalty"]
        + (1.0 - hs["systemic_low_risk_broadening_route"]) * 0.2,
    )
    encoding_margin_v92 = feature_term_v92 + structure_term_v92 + learning_term_v92 - pressure_term_v92

    return {
        "headline_metrics": {
            "feature_term_v92": feature_term_v92,
            "structure_term_v92": structure_term_v92,
            "learning_term_v92": learning_term_v92,
            "pressure_term_v92": pressure_term_v92,
            "encoding_margin_v92": encoding_margin_v92,
        },
        "closed_form_equation_v92": {
            "feature_term": "K_f_v92 = K_f_v91 + K_f_v91 * S_sys_broad_score * 0.004 + K_f_v91 * B_plastic_v36 * 0.001 + K_f_v91 * D_feature_v30 * 0.001",
            "structure_term": "K_s_v92 = K_s_v91 + K_s_v91 * S_sys_broad * 0.007 + K_s_v91 * B_struct_v36 * 0.004 + K_s_v91 * D_structure_v30 * 0.002",
            "learning_term": "K_l_v92 = K_l_v91 + K_l_v91 * R_train_v36 + M_sys_broad * 1000 + S_sys_broad_score * 1000 + M_brain_direct_v30 * 1000",
            "pressure_term": "P_v92 = P_v91 + G_train_v36 + P_sys_broad + 0.2 * (1 - R_sys_broad)",
            "margin_term": "M_encoding_v92 = K_f_v92 + K_s_v92 + K_l_v92 - P_v92",
        },
        "project_readout": {
            "summary": "v92 checks whether systemic low-risk broadening starts turning into a broader and more stable low-risk regime instead of only widening the previous band.",
            "next_question": "next push this broader low-risk regime into larger and longer systems to test whether it keeps broadening toward a wider low-risk steady zone.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Encoding Mechanism Closed Form V92 Report",
        "",
        f"- feature_term_v92: {hm['feature_term_v92']:.6f}",
        f"- structure_term_v92: {hm['structure_term_v92']:.6f}",
        f"- learning_term_v92: {hm['learning_term_v92']:.6f}",
        f"- pressure_term_v92: {hm['pressure_term_v92']:.6f}",
        f"- encoding_margin_v92: {hm['encoding_margin_v92']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v92_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
