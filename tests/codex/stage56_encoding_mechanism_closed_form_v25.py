from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v25_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v25_summary() -> dict:
    v24 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v24_20260320" / "summary.json"
    )
    equal_level = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_equal_level_closure_20260320" / "summary.json"
    )

    hv = v24["headline_metrics"]
    he = equal_level["headline_metrics"]

    feature_term_v25 = he["equalized_feature_v3"]
    structure_term_v25 = he["equalized_structure_v3"]
    learning_term_v25 = hv["learning_term_v24"] * (1.0 + he["equalization_confidence"])
    pressure_term_v25 = hv["pressure_term_v24"] + he["equalized_gap_v3"]
    encoding_margin_v25 = feature_term_v25 + structure_term_v25 + learning_term_v25 - pressure_term_v25

    return {
        "headline_metrics": {
            "feature_term_v25": feature_term_v25,
            "structure_term_v25": structure_term_v25,
            "learning_term_v25": learning_term_v25,
            "pressure_term_v25": pressure_term_v25,
            "encoding_margin_v25": encoding_margin_v25,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v25 = equalized_feature_v3",
            "structure_term": "K_s_v25 = equalized_structure_v3",
            "learning_term": "K_l_v25 = learning_term_v24 * (1 + equalization_confidence)",
            "pressure_term": "P_v25 = pressure_term_v24 + equalized_gap_v3",
            "margin_term": "M_encoding_v25 = K_f_v25 + K_s_v25 + K_l_v25 - P_v25",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第二十五版报告",
        "",
        f"- feature_term_v25: {hm['feature_term_v25']:.6f}",
        f"- structure_term_v25: {hm['structure_term_v25']:.6f}",
        f"- learning_term_v25: {hm['learning_term_v25']:.6f}",
        f"- pressure_term_v25: {hm['pressure_term_v25']:.6f}",
        f"- encoding_margin_v25: {hm['encoding_margin_v25']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v25_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
