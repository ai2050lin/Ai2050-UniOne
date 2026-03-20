from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v17_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v17_summary() -> dict:
    v16 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v16_20260320" / "summary.json")
    feature_layer = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_layer_definition_20260320" / "summary.json"
    )
    coupling = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_coupling_20260320" / "summary.json"
    )

    hv = v16["headline_metrics"]
    hf = feature_layer["headline_metrics"]
    hc = coupling["headline_metrics"]

    feature_term_v17 = hv["feature_term_v16"] + hf["feature_layer_core"]
    structure_term_v17 = hv["structure_term_v16"] + hc["coupling_margin"]
    learning_term_v17 = hv["learning_term_v16"] + hc["structure_feedback"]
    pressure_term_v17 = hv["pressure_term_v16"]
    encoding_margin_v17 = feature_term_v17 + structure_term_v17 + learning_term_v17 - pressure_term_v17

    return {
        "headline_metrics": {
            "feature_term_v17": feature_term_v17,
            "structure_term_v17": structure_term_v17,
            "learning_term_v17": learning_term_v17,
            "pressure_term_v17": pressure_term_v17,
            "encoding_margin_v17": encoding_margin_v17,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v17 = feature_term_v16 + feature_layer_core",
            "structure_term": "K_s_v17 = structure_term_v16 + coupling_margin",
            "learning_term": "K_l_v17 = learning_term_v16 + structure_feedback",
            "pressure_term": "P_v17 = pressure_term_v16",
            "margin_term": "M_encoding_v17 = K_f_v17 + K_s_v17 + K_l_v17 - P_v17",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第十七版报告",
        "",
        f"- feature_term_v17: {hm['feature_term_v17']:.6f}",
        f"- structure_term_v17: {hm['structure_term_v17']:.6f}",
        f"- learning_term_v17: {hm['learning_term_v17']:.6f}",
        f"- pressure_term_v17: {hm['pressure_term_v17']:.6f}",
        f"- encoding_margin_v17: {hm['encoding_margin_v17']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v17_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
