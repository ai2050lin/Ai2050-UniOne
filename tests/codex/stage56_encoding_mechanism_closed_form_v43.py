from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v43_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v43_summary() -> dict:
    v42 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v42_20260320" / "summary.json"
    )
    cross_modal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_modal_unification_strengthening_20260320" / "summary.json"
    )
    falsi = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_falsifiability_closure_strengthening_20260320" / "summary.json"
    )
    cross = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_version_stability_strengthening_20260320" / "summary.json"
    )

    hv = v42["headline_metrics"]
    hm = cross_modal["headline_metrics"]
    hf = falsi["headline_metrics"]
    hc = cross["headline_metrics"]

    feature_term_v43 = hv["feature_term_v42"] + hv["feature_term_v42"] * hm["cross_modal_unification_stable"] * 0.03
    structure_term_v43 = hv["structure_term_v42"] + hv["structure_term_v42"] * hm["modality_extension_stable"] * 0.03
    learning_term_v43 = (
        hv["learning_term_v42"]
        + hv["learning_term_v42"] * hf["falsifiability_closure_stable"]
        + hm["action_planning_stable"] * 1000.0
    )
    pressure_term_v43 = max(
        0.0,
        hv["pressure_term_v42"]
        + hm["modality_residual_stable"]
        + hf["residual_nonfalsifiable_stable"]
        - hc["stability_gain"],
    )
    encoding_margin_v43 = feature_term_v43 + structure_term_v43 + learning_term_v43 - pressure_term_v43

    return {
        "headline_metrics": {
            "feature_term_v43": feature_term_v43,
            "structure_term_v43": structure_term_v43,
            "learning_term_v43": learning_term_v43,
            "pressure_term_v43": pressure_term_v43,
            "encoding_margin_v43": encoding_margin_v43,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v43 = K_f_v42 + K_f_v42 * T_cross_star * 0.03",
            "structure_term": "K_s_v43 = K_s_v42 + K_s_v42 * T_mod_star * 0.03",
            "learning_term": "K_l_v43 = K_l_v42 + K_l_v42 * C_false_star + T_act_star * 1000",
            "pressure_term": "P_v43 = P_v42 + R_mod_star + R_false_star - Delta_stability_star",
            "margin_term": "M_encoding_v43 = K_f_v43 + K_s_v43 + K_l_v43 - P_v43",
        },
        "project_readout": {
            "summary": "第四十三版主核把跨模态统一强化和可判伪闭合强化一起并回主式，让统一主线更接近强桥接和强检验。",
            "next_question": "下一步要把这组强化对象推进到真实原型网络里，确认它们不是主式里自洽的高值。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第四十三版报告",
        "",
        f"- feature_term_v43: {hm['feature_term_v43']:.6f}",
        f"- structure_term_v43: {hm['structure_term_v43']:.6f}",
        f"- learning_term_v43: {hm['learning_term_v43']:.6f}",
        f"- pressure_term_v43: {hm['pressure_term_v43']:.6f}",
        f"- encoding_margin_v43: {hm['encoding_margin_v43']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v43_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
