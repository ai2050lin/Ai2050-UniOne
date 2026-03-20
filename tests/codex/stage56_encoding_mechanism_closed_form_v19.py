from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v19_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v19_summary() -> dict:
    v18 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v18_20260320" / "summary.json")
    feature_direct = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_layer_native_direct_measure_20260320" / "summary.json"
    )
    native_closure = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_native_closure_20260320" / "summary.json"
    )

    hv = v18["headline_metrics"]
    hf = feature_direct["headline_metrics"]
    hc = native_closure["headline_metrics"]

    feature_term_v19 = hv["feature_term_v18"] + hf["feature_direct_core_v3"]
    structure_term_v19 = hv["structure_term_v18"] + hc["native_closure_margin"]
    learning_term_v19 = hv["learning_term_v18"] + hc["closure_feedback"]
    pressure_term_v19 = hv["pressure_term_v18"]
    encoding_margin_v19 = feature_term_v19 + structure_term_v19 + learning_term_v19 - pressure_term_v19

    return {
        "headline_metrics": {
            "feature_term_v19": feature_term_v19,
            "structure_term_v19": structure_term_v19,
            "learning_term_v19": learning_term_v19,
            "pressure_term_v19": pressure_term_v19,
            "encoding_margin_v19": encoding_margin_v19,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v19 = feature_term_v18 + feature_direct_core_v3",
            "structure_term": "K_s_v19 = structure_term_v18 + native_closure_margin",
            "learning_term": "K_l_v19 = learning_term_v18 + closure_feedback",
            "pressure_term": "P_v19 = pressure_term_v18",
            "margin_term": "M_encoding_v19 = K_f_v19 + K_s_v19 + K_l_v19 - P_v19",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第十九版报告",
        "",
        f"- feature_term_v19: {hm['feature_term_v19']:.6f}",
        f"- structure_term_v19: {hm['structure_term_v19']:.6f}",
        f"- learning_term_v19: {hm['learning_term_v19']:.6f}",
        f"- pressure_term_v19: {hm['pressure_term_v19']:.6f}",
        f"- encoding_margin_v19: {hm['encoding_margin_v19']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v19_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
