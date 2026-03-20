from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_dnn_brain_math_theory_synthesis_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_dnn_brain_math_theory_synthesis_summary() -> dict:
    v36 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v36_20260320" / "summary.json"
    )
    cross = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_unification_cross_version_validation_20260320" / "summary.json"
    )
    high = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_icspb_unification_high_closure_20260320" / "summary.json"
    )
    stable = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_kernel_stability_strengthening_20260320" / "summary.json"
    )

    hv = v36["headline_metrics"]
    hc = cross["headline_metrics"]
    hh = high["headline_metrics"]
    hs = stable["headline_metrics"]

    dnn_language_core = (hv["feature_term_v36"] + hv["learning_term_v36"]) / hv["structure_term_v36"]
    brain_encoding_core = hh["unification_high_stability"] + hs["transport_kernel_stability_stable"]
    math_system_core = hv["encoding_margin_v36"] / (1.0 + hv["pressure_term_v36"])
    theory_bridge_strength = (dnn_language_core + brain_encoding_core + hc["cross_version_stability"]) / 3.0

    return {
        "headline_metrics": {
            "dnn_language_core": dnn_language_core,
            "brain_encoding_core": brain_encoding_core,
            "math_system_core": math_system_core,
            "theory_bridge_strength": theory_bridge_strength,
        },
        "synthesis_equation": {
            "dnn_term": "T_dnn = (K_f_v36 + K_l_v36) / K_s_v36",
            "brain_term": "T_brain = S_unify_high + K_keep_star",
            "math_term": "T_math = M_encoding_v36 / (1 + P_v36)",
            "bridge_term": "T_bridge = mean(T_dnn, T_brain, S_cross)",
        },
        "project_readout": {
            "summary": "总理论综合块把 DNN 语言结构分析、脑编码机制逆向分析和数学主核闭式化压进了同一套总纲。",
            "next_question": "下一步要把这个总纲从语言主线继续推广到更一般的智能结构，而不只停在语言编码闭包。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# DNN-脑机制-数学体系总理论报告",
        "",
        f"- dnn_language_core: {hm['dnn_language_core']:.6f}",
        f"- brain_encoding_core: {hm['brain_encoding_core']:.6f}",
        f"- math_system_core: {hm['math_system_core']:.6f}",
        f"- theory_bridge_strength: {hm['theory_bridge_strength']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_dnn_brain_math_theory_synthesis_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
