from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v26_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v26_summary() -> dict:
    v25 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v25_20260320" / "summary.json"
    )
    chain = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neuron_feature_network_chain_20260320" / "summary.json"
    )
    equal_level = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_equal_level_closure_20260320" / "summary.json"
    )

    hv = v25["headline_metrics"]
    hc = chain["headline_metrics"]
    he = equal_level["headline_metrics"]

    feature_term_v26 = hv["feature_term_v25"] + hc["neuron_seed_signal"] + hc["feature_selection_signal"]
    structure_term_v26 = hv["structure_term_v25"] + hc["network_growth_signal"] + hc["circuit_closure_signal"]
    learning_term_v26 = hv["learning_term_v25"] + hc["steady_feedback_signal"] + hc["chain_margin"] * he["equalization_confidence"]
    pressure_term_v26 = hv["pressure_term_v25"] + (1.0 - he["equalization_confidence"])
    encoding_margin_v26 = feature_term_v26 + structure_term_v26 + learning_term_v26 - pressure_term_v26

    return {
        "headline_metrics": {
            "feature_term_v26": feature_term_v26,
            "structure_term_v26": structure_term_v26,
            "learning_term_v26": learning_term_v26,
            "pressure_term_v26": pressure_term_v26,
            "encoding_margin_v26": encoding_margin_v26,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v26 = K_f_v25 + N_seed + N_feat",
            "structure_term": "K_s_v26 = K_s_v25 + N_struct_growth + N_circuit",
            "learning_term": "K_l_v26 = K_l_v25 + N_feedback + M_chain * C_equal",
            "pressure_term": "P_v26 = P_v25 + (1 - C_equal)",
            "margin_term": "M_encoding_v26 = K_f_v26 + K_s_v26 + K_l_v26 - P_v26",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第二十六版报告",
        "",
        f"- feature_term_v26: {hm['feature_term_v26']:.6f}",
        f"- structure_term_v26: {hm['structure_term_v26']:.6f}",
        f"- learning_term_v26: {hm['learning_term_v26']:.6f}",
        f"- pressure_term_v26: {hm['pressure_term_v26']:.6f}",
        f"- encoding_margin_v26: {hm['encoding_margin_v26']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v26_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
