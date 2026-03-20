from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v41_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v41_summary() -> dict:
    v40 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v40_20260320" / "summary.json"
    )
    possibility = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_unified_intelligence_theory_possibility_20260320" / "summary.json"
    )
    bridge = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_total_theory_bridge_expansion_20260320" / "summary.json"
    )
    cross = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_version_stability_strengthening_20260320" / "summary.json"
    )

    hv = v40["headline_metrics"]
    hp = possibility["headline_metrics"]
    hb = bridge["headline_metrics"]
    hc = cross["headline_metrics"]

    feature_term_v41 = hv["feature_term_v40"] + hv["feature_term_v40"] * hp["higher_unified_intelligence_possibility"] * 0.05
    structure_term_v41 = hv["structure_term_v40"] + hv["structure_term_v40"] * hb["brain_to_math_alignment"] * 0.05
    learning_term_v41 = (
        hv["learning_term_v40"]
        + hv["learning_term_v40"] * hp["unification_core"]
        + hb["total_bridge_strength_expanded"] * 1000.0
    )
    pressure_term_v41 = max(
        0.0,
        hv["pressure_term_v40"]
        + hp["falsifiability_gap"]
        + hp["modality_gap"]
        - hc["stability_gain"],
    )
    encoding_margin_v41 = feature_term_v41 + structure_term_v41 + learning_term_v41 - pressure_term_v41

    return {
        "headline_metrics": {
            "feature_term_v41": feature_term_v41,
            "structure_term_v41": structure_term_v41,
            "learning_term_v41": learning_term_v41,
            "pressure_term_v41": pressure_term_v41,
            "encoding_margin_v41": encoding_margin_v41,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v41 = K_f_v40 + K_f_v40 * P_unified * 0.05",
            "structure_term": "K_s_v41 = K_s_v40 + K_s_v40 * A_bm * 0.05",
            "learning_term": "K_l_v41 = K_l_v40 + K_l_v40 * U_core + T_bridge * 1000",
            "pressure_term": "P_v41 = P_v40 + D_false + D_mod - Delta_stability_star",
            "margin_term": "M_encoding_v41 = K_f_v41 + K_s_v41 + K_l_v41 - P_v41",
        },
        "project_readout": {
            "summary": "第四十一版主核开始把更高统一智能理论的可能性并回主式，让主核同时表达形成链、总桥和统一可能性。",
            "next_question": "下一步要把这个统一可能性对象推进到跨模态任务和原型网络里，确认它不只是主式里的高值。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第四十一版报告",
        "",
        f"- feature_term_v41: {hm['feature_term_v41']:.6f}",
        f"- structure_term_v41: {hm['structure_term_v41']:.6f}",
        f"- learning_term_v41: {hm['learning_term_v41']:.6f}",
        f"- pressure_term_v41: {hm['pressure_term_v41']:.6f}",
        f"- encoding_margin_v41: {hm['encoding_margin_v41']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v41_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
