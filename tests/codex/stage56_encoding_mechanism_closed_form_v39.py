from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v39_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v39_summary() -> dict:
    v38 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v38_20260320" / "summary.json"
    )
    theory = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_total_theory_bridge_expansion_20260320" / "summary.json"
    )
    stable = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_version_stability_strengthening_20260320" / "summary.json"
    )

    hv = v38["headline_metrics"]
    ht = theory["headline_metrics"]
    hs = stable["headline_metrics"]

    feature_term_v39 = hv["feature_term_v38"] + hv["feature_term_v38"] * ht["dnn_to_brain_alignment"] * 0.1
    structure_term_v39 = hv["structure_term_v38"] + hv["structure_term_v38"] * ht["brain_to_math_alignment"] * 0.1
    learning_term_v39 = hv["learning_term_v38"] + hv["learning_term_v38"] * hs["cross_version_stability_stable"] + ht["total_bridge_strength_expanded"] * 1000.0
    pressure_term_v39 = max(0.0, hv["pressure_term_v38"] - hs["stability_gain"] - 0.1 * ht["dnn_to_brain_alignment"])
    encoding_margin_v39 = feature_term_v39 + structure_term_v39 + learning_term_v39 - pressure_term_v39

    return {
        "headline_metrics": {
            "feature_term_v39": feature_term_v39,
            "structure_term_v39": structure_term_v39,
            "learning_term_v39": learning_term_v39,
            "pressure_term_v39": pressure_term_v39,
            "encoding_margin_v39": encoding_margin_v39,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v39 = K_f_v38 + K_f_v38 * A_db * 0.1",
            "structure_term": "K_s_v39 = K_s_v38 + K_s_v38 * A_bm * 0.1",
            "learning_term": "K_l_v39 = K_l_v38 + K_l_v38 * S_cross_star + T_bridge_plus * 1000",
            "pressure_term": "P_v39 = P_v38 - Delta_stability_star - 0.1 * A_db",
            "margin_term": "M_encoding_v39 = K_f_v39 + K_s_v39 + K_l_v39 - P_v39",
        },
        "project_readout": {
            "summary": "第三十九版主核开始把总理论桥接扩展直接并回主式，让 DNN 语言结构、脑编码机制和数学体系不只是并列，而是开始相互驱动。",
            "next_question": "下一步要验证这种三层相互驱动，是否已经足够支撑更一般的智能能力，而不只停在语言编码闭包。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第三十九版报告",
        "",
        f"- feature_term_v39: {hm['feature_term_v39']:.6f}",
        f"- structure_term_v39: {hm['structure_term_v39']:.6f}",
        f"- learning_term_v39: {hm['learning_term_v39']:.6f}",
        f"- pressure_term_v39: {hm['pressure_term_v39']:.6f}",
        f"- encoding_margin_v39: {hm['encoding_margin_v39']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v39_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
