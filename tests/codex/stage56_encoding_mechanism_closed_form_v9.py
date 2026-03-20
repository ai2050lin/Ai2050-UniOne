from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v9_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v9_summary() -> dict:
    v8 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v8_20260320" / "summary.json")
    feature = _load_json(ROOT / "tests" / "codex_temp" / "stage56_feature_extraction_primary_structure_20260320" / "summary.json")
    direct = _load_json(ROOT / "tests" / "codex_temp" / "stage56_circuit_native_direct_measure_20260320" / "summary.json")
    learning = _load_json(ROOT / "tests" / "codex_temp" / "stage56_pulse_to_network_learning_dynamics_20260320" / "summary.json")
    threshold = _load_json(ROOT / "tests" / "codex_temp" / "stage56_feature_primary_threshold_closure_20260320" / "summary.json")

    hv8 = v8["headline_metrics"]
    hf = feature["headline_metrics"]
    hd = direct["headline_metrics"]
    hl = learning["headline_metrics"]
    ht = threshold["headline_metrics"]

    feature_term_v9 = hv8["seed_feature_term_v8"] + ht["primary_threshold_margin"] + hf["feature_structure_support"]
    structure_term_v9 = hv8["structure_term_v8"] + hd["direct_circuit_margin"]
    learning_term_v9 = hv8["stability_term_v8"] + hl["learning_global"]
    pressure_term_v9 = hv8["pressure_term_v8"] + hd["direct_gate_measure"]
    encoding_margin_v9 = feature_term_v9 + structure_term_v9 + learning_term_v9 - pressure_term_v9

    return {
        "headline_metrics": {
            "feature_term_v9": feature_term_v9,
            "structure_term_v9": structure_term_v9,
            "learning_term_v9": learning_term_v9,
            "pressure_term_v9": pressure_term_v9,
            "encoding_margin_v9": encoding_margin_v9,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v9 = seed_feature_term_v8 + primary_threshold_margin + feature_structure_support",
            "structure_term": "K_s_v9 = structure_term_v8 + direct_circuit_margin",
            "learning_term": "K_l_v9 = stability_term_v8 + learning_global",
            "pressure_term": "P_v9 = pressure_term_v8 + direct_gate_measure",
            "margin_term": "M_encoding_v9 = K_f_v9 + K_s_v9 + K_l_v9 - P_v9",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第九版报告",
        "",
        f"- feature_term_v9: {hm['feature_term_v9']:.6f}",
        f"- structure_term_v9: {hm['structure_term_v9']:.6f}",
        f"- learning_term_v9: {hm['learning_term_v9']:.6f}",
        f"- pressure_term_v9: {hm['pressure_term_v9']:.6f}",
        f"- encoding_margin_v9: {hm['encoding_margin_v9']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v9_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
