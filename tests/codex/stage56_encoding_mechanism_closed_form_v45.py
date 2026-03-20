from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v45_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v45_summary() -> dict:
    v44 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v44_20260320" / "summary.json"
    )
    gap = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_gap_bottleneck_analysis_20260320" / "summary.json"
    )
    spike = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spiking_network_path_analysis_20260320" / "summary.json"
    )

    hv = v44["headline_metrics"]
    hg = gap["headline_metrics"]
    hs = spike["headline_metrics"]

    feature_term_v45 = hv["feature_term_v44"] + hv["feature_term_v44"] * hg["language_theory_completion"] * 0.01
    structure_term_v45 = hv["structure_term_v44"] + hv["structure_term_v44"] * hs["structure_generation_unlock"] * 0.01
    learning_term_v45 = (
        hv["learning_term_v44"]
        + hv["learning_term_v44"] * hs["spiking_network_path_readiness"]
        + hs["direct_agi_unlock"] * 1000.0
    )
    pressure_term_v45 = max(
        0.0,
        hv["pressure_term_v44"] + hg["largest_gap"] + hs["overlinearity_penalty"] - hg["language_theory_completion"]
    )
    encoding_margin_v45 = feature_term_v45 + structure_term_v45 + learning_term_v45 - pressure_term_v45

    return {
        "headline_metrics": {
            "feature_term_v45": feature_term_v45,
            "structure_term_v45": structure_term_v45,
            "learning_term_v45": learning_term_v45,
            "pressure_term_v45": pressure_term_v45,
            "encoding_margin_v45": encoding_margin_v45,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v45 = K_f_v44 + K_f_v44 * R_lang_theory * 0.01",
            "structure_term": "K_s_v45 = K_s_v44 + K_s_v44 * U_structure * 0.01",
            "learning_term": "K_l_v45 = K_l_v44 + K_l_v44 * R_spike_path + U_agi * 1000",
            "pressure_term": "P_v45 = P_v44 + G_max + P_linear - R_lang_theory",
            "margin_term": "M_encoding_v45 = K_f_v45 + K_s_v45 + K_l_v45 - P_v45",
        },
        "project_readout": {
            "summary": "第四十五版主核把语言缺口瓶颈判断和脉冲网络路径一起并回主式，用来检验“补齐语言理论是否足以直接通向 AGI”这一判断。",
            "next_question": "下一步要验证真正的主瓶颈是否已经从语言分析转移到训练终式和工程闭合，而不是继续停留在理论入口层。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第四十五版报告",
        "",
        f"- feature_term_v45: {hm['feature_term_v45']:.6f}",
        f"- structure_term_v45: {hm['structure_term_v45']:.6f}",
        f"- learning_term_v45: {hm['learning_term_v45']:.6f}",
        f"- pressure_term_v45: {hm['pressure_term_v45']:.6f}",
        f"- encoding_margin_v45: {hm['encoding_margin_v45']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v45_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
