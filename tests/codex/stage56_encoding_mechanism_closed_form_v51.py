from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v51_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v51_summary() -> dict:
    v50 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v50_20260320" / "summary.json"
    )
    lang = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_total_analysis_20260320" / "summary.json"
    )
    brain = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_reverse_analysis_20260320" / "summary.json"
    )

    hv = v50["headline_metrics"]
    hl = lang["headline_metrics"]
    hb = brain["headline_metrics"]

    feature_term_v51 = hv["feature_term_v50"] + hv["feature_term_v50"] * hl["language_feature_resolution"] * 0.01
    structure_term_v51 = hv["structure_term_v50"] + hv["structure_term_v50"] * hb["structure_recovery"] * 0.01
    learning_term_v51 = hv["learning_term_v50"] + hv["learning_term_v50"] * hl["language_principle_completion"] + hb["reverse_chain_strength"] * 1000.0
    pressure_term_v51 = max(0.0, hv["pressure_term_v50"] + hb["reverse_chain_gap"] + hl["language_remaining_gap"] - hl["language_principle_completion"])
    encoding_margin_v51 = feature_term_v51 + structure_term_v51 + learning_term_v51 - pressure_term_v51

    return {
        "headline_metrics": {
            "feature_term_v51": feature_term_v51,
            "structure_term_v51": structure_term_v51,
            "learning_term_v51": learning_term_v51,
            "pressure_term_v51": pressure_term_v51,
            "encoding_margin_v51": encoding_margin_v51,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v51 = K_f_v50 + K_f_v50 * L_feature * 0.01",
            "structure_term": "K_s_v51 = K_s_v50 + K_s_v50 * R_structure * 0.01",
            "learning_term": "K_l_v51 = K_l_v50 + K_l_v50 * L_principle + M_brain_reverse * 1000",
            "pressure_term": "P_v51 = P_v50 + G_reverse + G_language - L_principle",
            "margin_term": "M_encoding_v51 = K_f_v51 + K_s_v51 + K_l_v51 - P_v51",
        },
        "project_readout": {
            "summary": "第五十一版主核把语言总分析和逆向大脑编码机制一起并回主式，用来表达语言主入口和脑编码形成链的统一关系。",
            "next_question": "下一步要把语言总分析和脑编码逆向分析继续推进到更接近训练终式，而不是只停在解释主核。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第五十一版报告",
        "",
        f"- feature_term_v51: {hm['feature_term_v51']:.6f}",
        f"- structure_term_v51: {hm['structure_term_v51']:.6f}",
        f"- learning_term_v51: {hm['learning_term_v51']:.6f}",
        f"- pressure_term_v51: {hm['pressure_term_v51']:.6f}",
        f"- encoding_margin_v51: {hm['encoding_margin_v51']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v51_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
