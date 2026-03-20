from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v29_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v29_summary() -> dict:
    v28 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v28_20260320" / "summary.json"
    )
    origin = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neuron_origin_direct_refinement_20260320" / "summary.json"
    )
    structure = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_structure_genesis_confidence_refinement_20260320" / "summary.json"
    )

    hv = v28["headline_metrics"]
    ho = origin["headline_metrics"]
    hs = structure["headline_metrics"]

    feature_term_v29 = hv["feature_term_v28"] + ho["neuron_origin_margin_v2"]
    structure_term_v29 = hv["structure_term_v28"] + hs["structure_genesis_margin_v3"]
    learning_term_v29 = (
        hv["learning_term_v28"]
        + hs["feedback_refined_v2"]
        + hs["structure_genesis_margin_v3"] * ho["origin_stability_v2"]
    )
    pressure_term_v29 = hv["pressure_term_v28"] + (1.0 - hs["structure_direct_confidence_v3"])
    encoding_margin_v29 = feature_term_v29 + structure_term_v29 + learning_term_v29 - pressure_term_v29

    return {
        "headline_metrics": {
            "feature_term_v29": feature_term_v29,
            "structure_term_v29": structure_term_v29,
            "learning_term_v29": learning_term_v29,
            "pressure_term_v29": pressure_term_v29,
            "encoding_margin_v29": encoding_margin_v29,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v29 = K_f_v28 + M_origin_v2",
            "structure_term": "K_s_v29 = K_s_v28 + M_struct_v3",
            "learning_term": "K_l_v29 = K_l_v28 + S_fb_v2 + M_struct_v3 * S_origin_v2",
            "pressure_term": "P_v29 = P_v28 + (1 - C_struct_v3)",
            "margin_term": "M_encoding_v29 = K_f_v29 + K_s_v29 + K_l_v29 - P_v29",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第二十九版报告",
        "",
        f"- feature_term_v29: {hm['feature_term_v29']:.6f}",
        f"- structure_term_v29: {hm['structure_term_v29']:.6f}",
        f"- learning_term_v29: {hm['learning_term_v29']:.6f}",
        f"- pressure_term_v29: {hm['pressure_term_v29']:.6f}",
        f"- encoding_margin_v29: {hm['encoding_margin_v29']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v29_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
