from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v37_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v37_summary() -> dict:
    v36 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v36_20260320" / "summary.json"
    )
    cross = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_unification_cross_version_validation_20260320" / "summary.json"
    )
    theory = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_dnn_brain_math_theory_synthesis_20260320" / "summary.json"
    )

    hv = v36["headline_metrics"]
    hc = cross["headline_metrics"]
    ht = theory["headline_metrics"]

    feature_term_v37 = hv["feature_term_v36"] + hv["feature_term_v36"] * hc["feature_growth_consistency"]
    structure_term_v37 = hv["structure_term_v36"] + hv["structure_term_v36"] * hc["structure_growth_consistency"]
    learning_term_v37 = hv["learning_term_v36"] + hv["learning_term_v36"] * hc["cross_version_stability"] + ht["theory_bridge_strength"] * 1000.0
    pressure_term_v37 = max(0.0, hv["pressure_term_v36"] - hc["unification_persistence"] + hc["rollback_risk"])
    encoding_margin_v37 = feature_term_v37 + structure_term_v37 + learning_term_v37 - pressure_term_v37

    return {
        "headline_metrics": {
            "feature_term_v37": feature_term_v37,
            "structure_term_v37": structure_term_v37,
            "learning_term_v37": learning_term_v37,
            "pressure_term_v37": pressure_term_v37,
            "encoding_margin_v37": encoding_margin_v37,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v37 = K_f_v36 + K_f_v36 * G_f",
            "structure_term": "K_s_v37 = K_s_v36 + K_s_v36 * G_s",
            "learning_term": "K_l_v37 = K_l_v36 + K_l_v36 * S_cross + T_bridge * 1000",
            "pressure_term": "P_v37 = P_v36 - P_unify + R_back",
            "margin_term": "M_encoding_v37 = K_f_v37 + K_s_v37 + K_l_v37 - P_v37",
        },
        "project_readout": {
            "summary": "第三十七版主核开始把跨版本稳定性和总理论桥强度一起并回主式，目标是让主核不只在单一版本上成立，而是在 DNN、脑机制和数学体系三条线上同时稳定。",
            "next_question": "下一步要继续检验这种三线并核是否能跨更多对象成立，而不是只在当前语言主线附近成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第三十七版报告",
        "",
        f"- feature_term_v37: {hm['feature_term_v37']:.6f}",
        f"- structure_term_v37: {hm['structure_term_v37']:.6f}",
        f"- learning_term_v37: {hm['learning_term_v37']:.6f}",
        f"- pressure_term_v37: {hm['pressure_term_v37']:.6f}",
        f"- encoding_margin_v37: {hm['encoding_margin_v37']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v37_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
