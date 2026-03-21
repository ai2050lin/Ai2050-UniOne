from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v90_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v90_summary() -> dict:
    v89 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v89_20260321" / "summary.json"
    )
    systemic_expand = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_zone_expansion_validation_20260321" / "summary.json"
    )
    brain_v28 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v28_20260321" / "summary.json"
    )
    bridge_v34 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v34_20260321" / "summary.json"
    )

    hv = v89["headline_metrics"]
    hs = systemic_expand["headline_metrics"]
    hb = brain_v28["headline_metrics"]
    ht = bridge_v34["headline_metrics"]

    feature_term_v90 = (
        hv["feature_term_v89"]
        + hv["feature_term_v89"] * hs["systemic_low_risk_expansion_score"] * 0.004
        + hv["feature_term_v89"] * ht["plasticity_rule_alignment_v34"] * 0.001
        + hv["feature_term_v89"] * hb["direct_feature_measure_v28"] * 0.001
    )
    structure_term_v90 = (
        hv["structure_term_v89"]
        + hv["structure_term_v89"] * hs["systemic_low_risk_expansion_structure"] * 0.007
        + hv["structure_term_v89"] * ht["structure_rule_alignment_v34"] * 0.004
        + hv["structure_term_v89"] * hb["direct_structure_measure_v28"] * 0.002
    )
    learning_term_v90 = (
        hv["learning_term_v89"]
        + hv["learning_term_v89"] * ht["topology_training_readiness_v34"]
        + hs["systemic_low_risk_expansion_margin"] * 1000.0
        + hs["systemic_low_risk_expansion_score"] * 1000.0
        + hb["direct_brain_measure_v28"] * 1000.0
    )
    pressure_term_v90 = max(
        0.0,
        hv["pressure_term_v89"]
        + ht["topology_training_gap_v34"]
        + hs["systemic_low_risk_expansion_penalty"]
        + (1.0 - hs["systemic_low_risk_expansion_route"]) * 0.2,
    )
    encoding_margin_v90 = feature_term_v90 + structure_term_v90 + learning_term_v90 - pressure_term_v90

    return {
        "headline_metrics": {
            "feature_term_v90": feature_term_v90,
            "structure_term_v90": structure_term_v90,
            "learning_term_v90": learning_term_v90,
            "pressure_term_v90": pressure_term_v90,
            "encoding_margin_v90": encoding_margin_v90,
        },
        "closed_form_equation_v90": {
            "feature_term": "K_f_v90 = K_f_v89 + K_f_v89 * S_sys_expand_score * 0.004 + K_f_v89 * B_plastic_v34 * 0.001 + K_f_v89 * D_feature_v28 * 0.001",
            "structure_term": "K_s_v90 = K_s_v89 + K_s_v89 * S_sys_expand * 0.007 + K_s_v89 * B_struct_v34 * 0.004 + K_s_v89 * D_structure_v28 * 0.002",
            "learning_term": "K_l_v90 = K_l_v89 + K_l_v89 * R_train_v34 + M_sys_expand * 1000 + S_sys_expand_score * 1000 + M_brain_direct_v28 * 1000",
            "pressure_term": "P_v90 = P_v89 + G_train_v34 + P_sys_expand + 0.2 * (1 - R_sys_expand)",
            "margin_term": "M_encoding_v90 = K_f_v90 + K_s_v90 + K_l_v90 - P_v90",
        },
        "project_readout": {
            "summary": "v90 checks whether low-risk steady zone expansion becomes more explicitly systemic instead of staying near the previous boundary.",
            "next_question": "next push this systemic low-risk expansion into larger and longer systems to test whether it keeps enlarging toward a broader low-risk regime.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Encoding Mechanism Closed Form V90 Report",
        "",
        f"- feature_term_v90: {hm['feature_term_v90']:.6f}",
        f"- structure_term_v90: {hm['structure_term_v90']:.6f}",
        f"- learning_term_v90: {hm['learning_term_v90']:.6f}",
        f"- pressure_term_v90: {hm['pressure_term_v90']:.6f}",
        f"- encoding_margin_v90: {hm['encoding_margin_v90']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v90_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
