from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v42_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v42_summary() -> dict:
    v41 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v41_20260320" / "summary.json"
    )
    cross_modal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_modal_unification_bridge_20260320" / "summary.json"
    )
    falsi = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_falsifiability_closure_20260320" / "summary.json"
    )
    cross = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_version_stability_strengthening_20260320" / "summary.json"
    )

    hv = v41["headline_metrics"]
    hm = cross_modal["headline_metrics"]
    hf = falsi["headline_metrics"]
    hc = cross["headline_metrics"]

    feature_term_v42 = hv["feature_term_v41"] + hv["feature_term_v41"] * hm["cross_modal_unification_strength"] * 0.04
    structure_term_v42 = hv["structure_term_v41"] + hv["structure_term_v41"] * hm["modality_extension_strength"] * 0.04
    learning_term_v42 = (
        hv["learning_term_v41"]
        + hv["learning_term_v41"] * hf["falsifiability_closure"]
        + hm["action_planning_bridge"] * 1000.0
    )
    pressure_term_v42 = max(
        0.0,
        hv["pressure_term_v41"]
        + hm["modality_residual"]
        + hf["residual_nonfalsifiable"]
        - hc["stability_gain"],
    )
    encoding_margin_v42 = feature_term_v42 + structure_term_v42 + learning_term_v42 - pressure_term_v42

    return {
        "headline_metrics": {
            "feature_term_v42": feature_term_v42,
            "structure_term_v42": structure_term_v42,
            "learning_term_v42": learning_term_v42,
            "pressure_term_v42": pressure_term_v42,
            "encoding_margin_v42": encoding_margin_v42,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v42 = K_f_v41 + K_f_v41 * T_cross * 0.04",
            "structure_term": "K_s_v42 = K_s_v41 + K_s_v41 * T_mod * 0.04",
            "learning_term": "K_l_v42 = K_l_v41 + K_l_v41 * C_false + T_act * 1000",
            "pressure_term": "P_v42 = P_v41 + R_mod + R_false - Delta_stability_star",
            "margin_term": "M_encoding_v42 = K_f_v42 + K_s_v42 + K_l_v42 - P_v42",
        },
        "project_readout": {
            "summary": "第四十二版主核把跨模态统一桥和可判伪闭合一起并回主式，开始让主核同时表达统一能力和可检验能力。",
            "next_question": "下一步要把这两个对象推进到真实原型网络和跨模态任务里，确认它们不是理论主式里的局部高值。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第四十二版报告",
        "",
        f"- feature_term_v42: {hm['feature_term_v42']:.6f}",
        f"- structure_term_v42: {hm['structure_term_v42']:.6f}",
        f"- learning_term_v42: {hm['learning_term_v42']:.6f}",
        f"- pressure_term_v42: {hm['pressure_term_v42']:.6f}",
        f"- encoding_margin_v42: {hm['encoding_margin_v42']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v42_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
