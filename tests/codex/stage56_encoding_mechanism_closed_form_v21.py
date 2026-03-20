from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v21_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v21_summary() -> dict:
    v20 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v20_20260320" / "summary.json")
    feature_terminal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_layer_terminal_direct_20260320" / "summary.json"
    )
    structure_terminal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_terminal_closure_20260320" / "summary.json"
    )

    hv = v20["headline_metrics"]
    hf = feature_terminal["headline_metrics"]
    hs = structure_terminal["headline_metrics"]

    feature_term_v21 = hv["feature_term_v20"] + hf["feature_terminal_core_v5"]
    structure_term_v21 = hv["structure_term_v20"] + hs["terminal_closure_margin_v3"]
    learning_term_v21 = hv["learning_term_v20"] + hs["terminal_feedback_closure"]
    pressure_term_v21 = hv["pressure_term_v20"]
    encoding_margin_v21 = feature_term_v21 + structure_term_v21 + learning_term_v21 - pressure_term_v21

    return {
        "headline_metrics": {
            "feature_term_v21": feature_term_v21,
            "structure_term_v21": structure_term_v21,
            "learning_term_v21": learning_term_v21,
            "pressure_term_v21": pressure_term_v21,
            "encoding_margin_v21": encoding_margin_v21,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v21 = feature_term_v20 + feature_terminal_core_v5",
            "structure_term": "K_s_v21 = structure_term_v20 + terminal_closure_margin_v3",
            "learning_term": "K_l_v21 = learning_term_v20 + terminal_feedback_closure",
            "pressure_term": "P_v21 = pressure_term_v20",
            "margin_term": "M_encoding_v21 = K_f_v21 + K_s_v21 + K_l_v21 - P_v21",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第二十一版报告",
        "",
        f"- feature_term_v21: {hm['feature_term_v21']:.6f}",
        f"- structure_term_v21: {hm['structure_term_v21']:.6f}",
        f"- learning_term_v21: {hm['learning_term_v21']:.6f}",
        f"- pressure_term_v21: {hm['pressure_term_v21']:.6f}",
        f"- encoding_margin_v21: {hm['encoding_margin_v21']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v21_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
