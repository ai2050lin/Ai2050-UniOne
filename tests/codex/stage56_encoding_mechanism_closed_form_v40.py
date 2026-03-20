from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v40_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v40_summary() -> dict:
    v39 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v39_20260320" / "summary.json"
    )
    feas = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_architecture_feasibility_20260320" / "summary.json"
    )
    cross = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_version_stability_strengthening_20260320" / "summary.json"
    )

    hv = v39["headline_metrics"]
    hf = feas["headline_metrics"]
    hc = cross["headline_metrics"]

    feature_term_v40 = hv["feature_term_v39"] + hv["feature_term_v39"] * hf["language_capability_readiness"] * 0.08
    structure_term_v40 = hv["structure_term_v39"] + hv["structure_term_v39"] * hf["online_stability_readiness"] * 0.08
    learning_term_v40 = (
        hv["learning_term_v39"]
        + hv["learning_term_v39"] * hf["architecture_feasibility"]
        + hc["cross_version_stability_stable"] * 1000.0
    )
    pressure_term_v40 = max(0.0, hv["pressure_term_v39"] + hf["production_gap"] - hc["stability_gain"])
    encoding_margin_v40 = feature_term_v40 + structure_term_v40 + learning_term_v40 - pressure_term_v40

    return {
        "headline_metrics": {
            "feature_term_v40": feature_term_v40,
            "structure_term_v40": structure_term_v40,
            "learning_term_v40": learning_term_v40,
            "pressure_term_v40": pressure_term_v40,
            "encoding_margin_v40": encoding_margin_v40,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v40 = K_f_v39 + K_f_v39 * R_lang * 0.08",
            "structure_term": "K_s_v40 = K_s_v39 + K_s_v39 * R_online * 0.08",
            "learning_term": "K_l_v40 = K_l_v39 + K_l_v39 * F_arch + S_cross_star * 1000",
            "pressure_term": "P_v40 = P_v39 + G_prod - Delta_stability_star",
            "margin_term": "M_encoding_v40 = K_f_v40 + K_s_v40 + K_l_v40 - P_v40",
        },
        "project_readout": {
            "summary": "第四十版主核开始把即时学习网络可行性并回主式，让主核不只解释结构，也开始回答能否支撑具体架构目标。",
            "next_question": "下一步要用原型网络训练结果去校验这个可行性对象，而不是只停在理论层。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第四十版报告",
        "",
        f"- feature_term_v40: {hm['feature_term_v40']:.6f}",
        f"- structure_term_v40: {hm['structure_term_v40']:.6f}",
        f"- learning_term_v40: {hm['learning_term_v40']:.6f}",
        f"- pressure_term_v40: {hm['pressure_term_v40']:.6f}",
        f"- encoding_margin_v40: {hm['encoding_margin_v40']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v40_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
