from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v93_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v93_summary() -> dict:
    v92 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v92_20260321" / "summary.json"
    )
    systemic_band = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_band_extension_validation_20260321" / "summary.json"
    )
    brain_v31 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v31_20260321" / "summary.json"
    )
    bridge_v37 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v37_20260321" / "summary.json"
    )

    hv = v92["headline_metrics"]
    hs = systemic_band["headline_metrics"]
    hb = brain_v31["headline_metrics"]
    ht = bridge_v37["headline_metrics"]

    feature_term_v93 = (
        hv["feature_term_v92"]
        + hv["feature_term_v92"] * hs["systemic_low_risk_band_score"] * 0.004
        + hv["feature_term_v92"] * ht["plasticity_rule_alignment_v37"] * 0.001
        + hv["feature_term_v92"] * hb["direct_feature_measure_v31"] * 0.001
    )
    structure_term_v93 = (
        hv["structure_term_v92"]
        + hv["structure_term_v92"] * hs["systemic_low_risk_band_structure"] * 0.007
        + hv["structure_term_v92"] * ht["structure_rule_alignment_v37"] * 0.004
        + hv["structure_term_v92"] * hb["direct_structure_measure_v31"] * 0.002
    )
    learning_term_v93 = (
        hv["learning_term_v92"]
        + hv["learning_term_v92"] * ht["topology_training_readiness_v37"]
        + hs["systemic_low_risk_band_margin"] * 1000.0
        + hs["systemic_low_risk_band_score"] * 1000.0
        + hb["direct_brain_measure_v31"] * 1000.0
    )
    pressure_term_v93 = max(
        0.0,
        hv["pressure_term_v92"]
        + ht["topology_training_gap_v37"]
        + hs["systemic_low_risk_band_penalty"]
        + (1.0 - hs["systemic_low_risk_band_route"]) * 0.2,
    )
    encoding_margin_v93 = feature_term_v93 + structure_term_v93 + learning_term_v93 - pressure_term_v93

    return {
        "headline_metrics": {
            "feature_term_v93": feature_term_v93,
            "structure_term_v93": structure_term_v93,
            "learning_term_v93": learning_term_v93,
            "pressure_term_v93": pressure_term_v93,
            "encoding_margin_v93": encoding_margin_v93,
        },
        "closed_form_equation_v93": {
            "feature_term": "K_f_v93 = K_f_v92 + K_f_v92 * S_sys_band_score * 0.004 + K_f_v92 * B_plastic_v37 * 0.001 + K_f_v92 * D_feature_v31 * 0.001",
            "structure_term": "K_s_v93 = K_s_v92 + K_s_v92 * S_sys_band * 0.007 + K_s_v92 * B_struct_v37 * 0.004 + K_s_v92 * D_structure_v31 * 0.002",
            "learning_term": "K_l_v93 = K_l_v92 + K_l_v92 * R_train_v37 + M_sys_band * 1000 + S_sys_band_score * 1000 + M_brain_direct_v31 * 1000",
            "pressure_term": "P_v93 = P_v92 + G_train_v37 + P_sys_band + 0.2 * (1 - R_sys_band)",
            "margin_term": "M_encoding_v93 = K_f_v93 + K_s_v93 + K_l_v93 - P_v93",
        },
        "project_readout": {
            "summary": "v93 checks whether systemic low-risk band extension starts turning into a broader and more stable low-risk field instead of only extending the previous band.",
            "next_question": "next push this broader low-risk field into larger and longer systems to test whether it keeps extending toward a wider low-risk steady field.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Encoding Mechanism Closed Form V93 Report",
        "",
        f"- feature_term_v93: {hm['feature_term_v93']:.6f}",
        f"- structure_term_v93: {hm['structure_term_v93']:.6f}",
        f"- learning_term_v93: {hm['learning_term_v93']:.6f}",
        f"- pressure_term_v93: {hm['pressure_term_v93']:.6f}",
        f"- encoding_margin_v93: {hm['encoding_margin_v93']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v93_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
