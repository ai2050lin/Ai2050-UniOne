from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v31_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v31_summary() -> dict:
    v30 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v30_20260320" / "summary.json"
    )
    stability = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_structure_stability_reparameterization_20260320" / "summary.json"
    )
    origin = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neuron_origin_direct_refinement_20260320" / "summary.json"
    )

    hv = v30["headline_metrics"]
    hs = stability["headline_metrics"]
    ho = origin["headline_metrics"]

    feature_term_v31 = hv["feature_term_v30"] + ho["origin_stability_v2"] * hs["stability_strength"]
    structure_term_v31 = hv["structure_term_v30"] + hs["stability_intensity"]
    learning_term_v31 = hv["learning_term_v30"] + hs["stability_balance"] + hs["stability_intensity"] * hs["closure_alignment"]
    pressure_term_v31 = hv["pressure_term_v30"] + (1.0 - hs["stability_strength"])
    encoding_margin_v31 = feature_term_v31 + structure_term_v31 + learning_term_v31 - pressure_term_v31

    return {
        "headline_metrics": {
            "feature_term_v31": feature_term_v31,
            "structure_term_v31": structure_term_v31,
            "learning_term_v31": learning_term_v31,
            "pressure_term_v31": pressure_term_v31,
            "encoding_margin_v31": encoding_margin_v31,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v31 = K_f_v30 + S_origin_v2 * S_strength",
            "structure_term": "K_s_v31 = K_s_v30 + I_struct",
            "learning_term": "K_l_v31 = K_l_v30 + B_struct + I_struct * A_closure",
            "pressure_term": "P_v31 = P_v30 + (1 - S_strength)",
            "margin_term": "M_encoding_v31 = K_f_v31 + K_s_v31 + K_l_v31 - P_v31",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第三十一版报告",
        "",
        f"- feature_term_v31: {hm['feature_term_v31']:.6f}",
        f"- structure_term_v31: {hm['structure_term_v31']:.6f}",
        f"- learning_term_v31: {hm['learning_term_v31']:.6f}",
        f"- pressure_term_v31: {hm['pressure_term_v31']:.6f}",
        f"- encoding_margin_v31: {hm['encoding_margin_v31']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v31_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
