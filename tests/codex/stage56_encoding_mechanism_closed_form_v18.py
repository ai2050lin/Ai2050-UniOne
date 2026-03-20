from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v18_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v18_summary() -> dict:
    v17 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v17_20260320" / "summary.json")
    native_feature = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_layer_nativeization_20260320" / "summary.json"
    )
    native_coupling = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_native_coupling_20260320" / "summary.json"
    )

    hv = v17["headline_metrics"]
    hf = native_feature["headline_metrics"]
    hc = native_coupling["headline_metrics"]

    feature_term_v18 = hv["feature_term_v17"] + hf["feature_native_core_v2"]
    structure_term_v18 = hv["structure_term_v17"] + hc["native_coupling_margin"]
    learning_term_v18 = hv["learning_term_v17"] + hc["native_feedback"]
    pressure_term_v18 = hv["pressure_term_v17"]
    encoding_margin_v18 = feature_term_v18 + structure_term_v18 + learning_term_v18 - pressure_term_v18

    return {
        "headline_metrics": {
            "feature_term_v18": feature_term_v18,
            "structure_term_v18": structure_term_v18,
            "learning_term_v18": learning_term_v18,
            "pressure_term_v18": pressure_term_v18,
            "encoding_margin_v18": encoding_margin_v18,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v18 = feature_term_v17 + feature_native_core_v2",
            "structure_term": "K_s_v18 = structure_term_v17 + native_coupling_margin",
            "learning_term": "K_l_v18 = learning_term_v17 + native_feedback",
            "pressure_term": "P_v18 = pressure_term_v17",
            "margin_term": "M_encoding_v18 = K_f_v18 + K_s_v18 + K_l_v18 - P_v18",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第十八版报告",
        "",
        f"- feature_term_v18: {hm['feature_term_v18']:.6f}",
        f"- structure_term_v18: {hm['structure_term_v18']:.6f}",
        f"- learning_term_v18: {hm['learning_term_v18']:.6f}",
        f"- pressure_term_v18: {hm['pressure_term_v18']:.6f}",
        f"- encoding_margin_v18: {hm['encoding_margin_v18']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v18_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
