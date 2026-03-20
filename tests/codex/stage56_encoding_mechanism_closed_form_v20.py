from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v20_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v20_summary() -> dict:
    v19 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v19_20260320" / "summary.json")
    feature_close = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_layer_direct_closure_20260320" / "summary.json"
    )
    structure_close = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_direct_closure_20260320" / "summary.json"
    )

    hv = v19["headline_metrics"]
    hf = feature_close["headline_metrics"]
    hs = structure_close["headline_metrics"]

    feature_term_v20 = hv["feature_term_v19"] + hf["feature_direct_closure_v4"]
    structure_term_v20 = hv["structure_term_v19"] + hs["direct_closure_margin_v2"]
    learning_term_v20 = hv["learning_term_v19"] + hs["direct_feedback_closure"]
    pressure_term_v20 = hv["pressure_term_v19"]
    encoding_margin_v20 = feature_term_v20 + structure_term_v20 + learning_term_v20 - pressure_term_v20

    return {
        "headline_metrics": {
            "feature_term_v20": feature_term_v20,
            "structure_term_v20": structure_term_v20,
            "learning_term_v20": learning_term_v20,
            "pressure_term_v20": pressure_term_v20,
            "encoding_margin_v20": encoding_margin_v20,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v20 = feature_term_v19 + feature_direct_closure_v4",
            "structure_term": "K_s_v20 = structure_term_v19 + direct_closure_margin_v2",
            "learning_term": "K_l_v20 = learning_term_v19 + direct_feedback_closure",
            "pressure_term": "P_v20 = pressure_term_v19",
            "margin_term": "M_encoding_v20 = K_f_v20 + K_s_v20 + K_l_v20 - P_v20",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第二十版报告",
        "",
        f"- feature_term_v20: {hm['feature_term_v20']:.6f}",
        f"- structure_term_v20: {hm['structure_term_v20']:.6f}",
        f"- learning_term_v20: {hm['learning_term_v20']:.6f}",
        f"- pressure_term_v20: {hm['pressure_term_v20']:.6f}",
        f"- encoding_margin_v20: {hm['encoding_margin_v20']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v20_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
