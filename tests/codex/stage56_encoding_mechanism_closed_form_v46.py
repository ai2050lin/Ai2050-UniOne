from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v46_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v46_summary() -> dict:
    v45 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v45_20260320" / "summary.json"
    )
    centrality = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_centrality_analysis_20260320" / "summary.json"
    )
    sufficiency = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_sufficiency_analysis_20260320" / "summary.json"
    )

    hv = v45["headline_metrics"]
    hc = centrality["headline_metrics"]
    hs = sufficiency["headline_metrics"]

    feature_term_v46 = hv["feature_term_v45"] + hv["feature_term_v45"] * hc["language_specialness"] * 0.01
    structure_term_v46 = hv["structure_term_v45"] + hv["structure_term_v45"] * hc["language_bridge_power"] * 0.01
    learning_term_v46 = (
        hv["learning_term_v45"]
        + hv["learning_term_v45"] * hs["language_only_sufficiency"]
        + hs["intelligence_theory_completion"] * 1000.0
    )
    pressure_term_v46 = max(
        0.0,
        hv["pressure_term_v45"] + hs["missing_nonlanguage_mass"] + hs["language_to_all_gap"] - hc["language_centrality"]
    )
    encoding_margin_v46 = feature_term_v46 + structure_term_v46 + learning_term_v46 - pressure_term_v46

    return {
        "headline_metrics": {
            "feature_term_v46": feature_term_v46,
            "structure_term_v46": structure_term_v46,
            "learning_term_v46": learning_term_v46,
            "pressure_term_v46": pressure_term_v46,
            "encoding_margin_v46": encoding_margin_v46,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v46 = K_f_v45 + K_f_v45 * S_lang * 0.01",
            "structure_term": "K_s_v46 = K_s_v45 + K_s_v45 * B_lang * 0.01",
            "learning_term": "K_l_v46 = K_l_v45 + K_l_v45 * S_lang_only + S_intel * 1000",
            "pressure_term": "P_v46 = P_v45 + M_nonlang + G_lang_to_all - C_lang",
            "margin_term": "M_encoding_v46 = K_f_v46 + K_s_v46 + K_l_v46 - P_v46",
        },
        "project_readout": {
            "summary": "第四十六版主核把语言中心性和语言充分性分析并回主式，用来检验“只要彻底理解语言就能完成全部智能理论”这条判断。",
            "next_question": "下一步要继续验证语言入口之外的非语言质量是否仍然主导训练终式、统一数学闭合和工程落地。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第四十六版报告",
        "",
        f"- feature_term_v46: {hm['feature_term_v46']:.6f}",
        f"- structure_term_v46: {hm['structure_term_v46']:.6f}",
        f"- learning_term_v46: {hm['learning_term_v46']:.6f}",
        f"- pressure_term_v46: {hm['pressure_term_v46']:.6f}",
        f"- encoding_margin_v46: {hm['encoding_margin_v46']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v46_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
