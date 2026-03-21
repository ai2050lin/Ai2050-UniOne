from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v88_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v88_summary() -> dict:
    v87 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v87_20260321" / "summary.json"
    )
    low_risk_zone = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_low_risk_steady_zone_validation_20260321" / "summary.json"
    )
    brain_v26 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v26_20260321" / "summary.json"
    )
    bridge_v32 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v32_20260321" / "summary.json"
    )

    hv = v87["headline_metrics"]
    hs = low_risk_zone["headline_metrics"]
    hb = brain_v26["headline_metrics"]
    ht = bridge_v32["headline_metrics"]

    feature_term_v88 = (
        hv["feature_term_v87"]
        + hv["feature_term_v87"] * hs["low_risk_zone_score"] * 0.004
        + hv["feature_term_v87"] * ht["plasticity_rule_alignment_v32"] * 0.001
        + hv["feature_term_v87"] * hb["direct_feature_measure_v26"] * 0.001
    )
    structure_term_v88 = (
        hv["structure_term_v87"]
        + hv["structure_term_v87"] * hs["low_risk_zone_structure"] * 0.007
        + hv["structure_term_v87"] * ht["structure_rule_alignment_v32"] * 0.004
        + hv["structure_term_v87"] * hb["direct_structure_measure_v26"] * 0.002
    )
    learning_term_v88 = (
        hv["learning_term_v87"]
        + hv["learning_term_v87"] * ht["topology_training_readiness_v32"]
        + hs["low_risk_zone_margin"] * 1000.0
        + hs["low_risk_zone_score"] * 1000.0
        + hb["direct_brain_measure_v26"] * 1000.0
    )
    pressure_term_v88 = max(
        0.0,
        hv["pressure_term_v87"]
        + ht["topology_training_gap_v32"]
        + hs["low_risk_zone_penalty"]
        + (1.0 - hs["low_risk_zone_route"]) * 0.2,
    )
    encoding_margin_v88 = feature_term_v88 + structure_term_v88 + learning_term_v88 - pressure_term_v88

    return {
        "headline_metrics": {
            "feature_term_v88": feature_term_v88,
            "structure_term_v88": structure_term_v88,
            "learning_term_v88": learning_term_v88,
            "pressure_term_v88": pressure_term_v88,
            "encoding_margin_v88": encoding_margin_v88,
        },
        "closed_form_equation_v88": {
            "feature_term": "K_f_v88 = K_f_v87 + K_f_v87 * S_zone_score * 0.004 + K_f_v87 * B_plastic_v32 * 0.001 + K_f_v87 * D_feature_v26 * 0.001",
            "structure_term": "K_s_v88 = K_s_v87 + K_s_v87 * S_zone * 0.007 + K_s_v87 * B_struct_v32 * 0.004 + K_s_v87 * D_structure_v26 * 0.002",
            "learning_term": "K_l_v88 = K_l_v87 + K_l_v87 * R_train_v32 + M_zone * 1000 + S_zone_score * 1000 + M_brain_direct_v26 * 1000",
            "pressure_term": "P_v88 = P_v87 + G_train_v32 + P_zone + 0.2 * (1 - R_zone)",
            "margin_term": "M_encoding_v88 = K_f_v88 + K_s_v88 + K_l_v88 - P_v88",
        },
        "project_readout": {
            "summary": "v88 checks whether the low-risk steady zone starts turning into a more explicit systemic low-risk steady regime.",
            "next_question": "next push this core into larger and longer systems to see whether the low-risk zone expands instead of flattening.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Encoding Mechanism Closed Form V88 Report",
        "",
        f"- feature_term_v88: {hm['feature_term_v88']:.6f}",
        f"- structure_term_v88: {hm['structure_term_v88']:.6f}",
        f"- learning_term_v88: {hm['learning_term_v88']:.6f}",
        f"- pressure_term_v88: {hm['pressure_term_v88']:.6f}",
        f"- encoding_margin_v88: {hm['encoding_margin_v88']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v88_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
