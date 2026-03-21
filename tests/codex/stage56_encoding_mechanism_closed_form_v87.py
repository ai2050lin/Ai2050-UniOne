from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v87_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v87_summary() -> dict:
    v86 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v86_20260321" / "summary.json"
    )
    low_risk = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_low_risk_steady_amplification_validation_20260321" / "summary.json"
    )
    brain_v25 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v25_20260321" / "summary.json"
    )
    bridge_v31 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v31_20260321" / "summary.json"
    )

    hv = v86["headline_metrics"]
    hs = low_risk["headline_metrics"]
    hb = brain_v25["headline_metrics"]
    ht = bridge_v31["headline_metrics"]

    feature_term_v87 = (
        hv["feature_term_v86"]
        + hv["feature_term_v86"] * hs["low_risk_score"] * 0.004
        + hv["feature_term_v86"] * ht["plasticity_rule_alignment_v31"] * 0.001
        + hv["feature_term_v86"] * hb["direct_feature_measure_v25"] * 0.001
    )
    structure_term_v87 = (
        hv["structure_term_v86"]
        + hv["structure_term_v86"] * hs["low_risk_structure"] * 0.007
        + hv["structure_term_v86"] * ht["structure_rule_alignment_v31"] * 0.004
        + hv["structure_term_v86"] * hb["direct_structure_measure_v25"] * 0.002
    )
    learning_term_v87 = (
        hv["learning_term_v86"]
        + hv["learning_term_v86"] * ht["topology_training_readiness_v31"]
        + hs["low_risk_margin"] * 1000.0
        + hs["low_risk_score"] * 1000.0
        + hb["direct_brain_measure_v25"] * 1000.0
    )
    pressure_term_v87 = max(
        0.0,
        hv["pressure_term_v86"]
        + ht["topology_training_gap_v31"]
        + hs["low_risk_penalty"]
        + (1.0 - hs["low_risk_route"]) * 0.2,
    )
    encoding_margin_v87 = feature_term_v87 + structure_term_v87 + learning_term_v87 - pressure_term_v87

    return {
        "headline_metrics": {
            "feature_term_v87": feature_term_v87,
            "structure_term_v87": structure_term_v87,
            "learning_term_v87": learning_term_v87,
            "pressure_term_v87": pressure_term_v87,
            "encoding_margin_v87": encoding_margin_v87,
        },
        "closed_form_equation_v87": {
            "feature_term": "K_f_v87 = K_f_v86 + K_f_v86 * S_low_score * 0.004 + K_f_v86 * B_plastic_v31 * 0.001 + K_f_v86 * D_feature_v25 * 0.001",
            "structure_term": "K_s_v87 = K_s_v86 + K_s_v86 * S_low * 0.007 + K_s_v86 * B_struct_v31 * 0.004 + K_s_v86 * D_structure_v25 * 0.002",
            "learning_term": "K_l_v87 = K_l_v86 + K_l_v86 * R_train_v31 + M_low * 1000 + S_low_score * 1000 + M_brain_direct_v25 * 1000",
            "pressure_term": "P_v87 = P_v86 + G_train_v31 + P_low + 0.2 * (1 - R_low)",
            "margin_term": "M_encoding_v87 = K_f_v87 + K_s_v87 + K_l_v87 - P_v87",
        },
        "project_readout": {
            "summary": "v87 checks whether the systemic steady amplification chain starts moving into a lower-risk systemic steady regime.",
            "next_question": "next push this core into larger and longer systems to see whether the lower-risk trend keeps improving instead of flattening.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Encoding Mechanism Closed Form V87 Report",
        "",
        f"- feature_term_v87: {hm['feature_term_v87']:.6f}",
        f"- structure_term_v87: {hm['structure_term_v87']:.6f}",
        f"- learning_term_v87: {hm['learning_term_v87']:.6f}",
        f"- pressure_term_v87: {hm['pressure_term_v87']:.6f}",
        f"- encoding_margin_v87: {hm['encoding_margin_v87']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v87_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
