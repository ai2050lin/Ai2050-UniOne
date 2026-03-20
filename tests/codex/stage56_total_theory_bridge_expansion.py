from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_total_theory_bridge_expansion_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_total_theory_bridge_expansion_summary() -> dict:
    theory = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_dnn_brain_math_theory_synthesis_20260320" / "summary.json"
    )
    v38 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v38_20260320" / "summary.json"
    )
    stable = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_version_stability_strengthening_20260320" / "summary.json"
    )

    ht = theory["headline_metrics"]
    hv = v38["headline_metrics"]
    hs = stable["headline_metrics"]

    dnn_to_brain_alignment = ht["dnn_language_core"] / (1.0 + ht["brain_encoding_core"])
    brain_to_math_alignment = ht["brain_encoding_core"] / (1.0 + hs["rollback_risk_reduced"])
    math_to_intelligence_alignment = ht["math_system_core"] / (1.0 + hv["pressure_term_v38"])
    total_bridge_strength_expanded = (
        dnn_to_brain_alignment + brain_to_math_alignment + hs["cross_version_stability_stable"]
    ) / 3.0

    return {
        "headline_metrics": {
            "dnn_to_brain_alignment": dnn_to_brain_alignment,
            "brain_to_math_alignment": brain_to_math_alignment,
            "math_to_intelligence_alignment": math_to_intelligence_alignment,
            "total_bridge_strength_expanded": total_bridge_strength_expanded,
        },
        "expansion_equation": {
            "dnn_term": "A_db = T_dnn / (1 + T_brain)",
            "brain_term": "A_bm = T_brain / (1 + R_back_star)",
            "math_term": "A_mi = T_math / (1 + P_v38)",
            "bridge_term": "T_bridge_plus = mean(A_db, A_bm, S_cross_star)",
        },
        "project_readout": {
            "summary": "总理论桥接扩展块开始把语言结构分析、脑编码机制和数学体系之间的桥从静态并置推进到方向性对齐。",
            "next_question": "下一步要把这个方向性总桥从语言主线继续推广到更一般的智能能力结构。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 总理论桥接扩展报告",
        "",
        f"- dnn_to_brain_alignment: {hm['dnn_to_brain_alignment']:.6f}",
        f"- brain_to_math_alignment: {hm['brain_to_math_alignment']:.6f}",
        f"- math_to_intelligence_alignment: {hm['math_to_intelligence_alignment']:.6f}",
        f"- total_bridge_strength_expanded: {hm['total_bridge_strength_expanded']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_total_theory_bridge_expansion_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
