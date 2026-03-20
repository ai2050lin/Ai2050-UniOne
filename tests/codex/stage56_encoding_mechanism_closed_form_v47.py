from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v47_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v47_summary() -> dict:
    v46 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v46_20260320" / "summary.json"
    )
    lang = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_system_principles_20260320" / "summary.json"
    )

    hv = v46["headline_metrics"]
    hl = lang["headline_metrics"]

    feature_term_v47 = hv["feature_term_v46"] + hv["feature_term_v46"] * hl["language_feature_core"] * 0.01
    structure_term_v47 = hv["structure_term_v46"] + hv["structure_term_v46"] * hl["language_structure_core"] * 0.01
    learning_term_v47 = hv["learning_term_v46"] + hv["learning_term_v46"] * hl["language_learning_core"] + hl["language_system_margin"] * 1000.0
    pressure_term_v47 = max(0.0, hv["pressure_term_v46"] + hl["language_pressure_core"] - hl["language_entry_core"])
    encoding_margin_v47 = feature_term_v47 + structure_term_v47 + learning_term_v47 - pressure_term_v47

    return {
        "headline_metrics": {
            "feature_term_v47": feature_term_v47,
            "structure_term_v47": structure_term_v47,
            "learning_term_v47": learning_term_v47,
            "pressure_term_v47": pressure_term_v47,
            "encoding_margin_v47": encoding_margin_v47,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v47 = K_f_v46 + K_f_v46 * F_lang * 0.01",
            "structure_term": "K_s_v47 = K_s_v46 + K_s_v46 * S_lang * 0.01",
            "learning_term": "K_l_v47 = K_l_v46 + K_l_v46 * L_lang + M_lang * 1000",
            "pressure_term": "P_v47 = P_v46 + P_lang - E_lang",
            "margin_term": "M_encoding_v47 = K_f_v47 + K_s_v47 + K_l_v47 - P_v47",
        },
        "project_readout": {
            "summary": "第四十七版主核把语言系统五层原理并回主式，开始让语言理论不只是入口判断，而是直接进入统一主核的结构表达。",
            "next_question": "下一步要继续把语言系统五层推进到更原生的神经与回路变量，减少当前中层对象的比例。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第四十七版报告",
        "",
        f"- feature_term_v47: {hm['feature_term_v47']:.6f}",
        f"- structure_term_v47: {hm['structure_term_v47']:.6f}",
        f"- learning_term_v47: {hm['learning_term_v47']:.6f}",
        f"- pressure_term_v47: {hm['pressure_term_v47']:.6f}",
        f"- encoding_margin_v47: {hm['encoding_margin_v47']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v47_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
