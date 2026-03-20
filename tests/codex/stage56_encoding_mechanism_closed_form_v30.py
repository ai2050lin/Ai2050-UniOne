from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v30_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v30_summary() -> dict:
    v29 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v29_20260320" / "summary.json"
    )
    structure_stable = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_structure_genesis_confidence_stabilization_20260320" / "summary.json"
    )
    origin = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neuron_origin_direct_refinement_20260320" / "summary.json"
    )

    hv = v29["headline_metrics"]
    hs = structure_stable["headline_metrics"]
    ho = origin["headline_metrics"]

    feature_term_v30 = hv["feature_term_v29"] + ho["origin_stability_v2"] * ho["neuron_origin_margin_v2"]
    structure_term_v30 = hv["structure_term_v29"] + hs["stabilized_margin"]
    learning_term_v30 = hv["learning_term_v29"] + hs["stabilized_feedback"] + hs["stabilized_margin"] * hs["stabilized_confidence"]
    pressure_term_v30 = hv["pressure_term_v29"] + (1.0 - hs["stabilized_confidence"])
    encoding_margin_v30 = feature_term_v30 + structure_term_v30 + learning_term_v30 - pressure_term_v30

    return {
        "headline_metrics": {
            "feature_term_v30": feature_term_v30,
            "structure_term_v30": structure_term_v30,
            "learning_term_v30": learning_term_v30,
            "pressure_term_v30": pressure_term_v30,
            "encoding_margin_v30": encoding_margin_v30,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v30 = K_f_v29 + S_origin_v2 * M_origin_v2",
            "structure_term": "K_s_v30 = K_s_v29 + M_struct_v4",
            "learning_term": "K_l_v30 = K_l_v29 + S_fb_v3 + M_struct_v4 * C_struct_v4",
            "pressure_term": "P_v30 = P_v29 + (1 - C_struct_v4)",
            "margin_term": "M_encoding_v30 = K_f_v30 + K_s_v30 + K_l_v30 - P_v30",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第三十版报告",
        "",
        f"- feature_term_v30: {hm['feature_term_v30']:.6f}",
        f"- structure_term_v30: {hm['structure_term_v30']:.6f}",
        f"- learning_term_v30: {hm['learning_term_v30']:.6f}",
        f"- pressure_term_v30: {hm['pressure_term_v30']:.6f}",
        f"- encoding_margin_v30: {hm['encoding_margin_v30']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v30_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
