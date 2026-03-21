from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v57_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v57_summary() -> dict:
    v56 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v56_20260321" / "summary.json"
    )
    mech = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_3d_topology_encoding_mechanism_20260321" / "summary.json"
    )
    scale = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_3d_topology_scaling_analysis_20260321" / "summary.json"
    )
    framework = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_project_framework_synthesis_20260321" / "summary.json"
    )

    hv = v56["headline_metrics"]
    hm = mech["headline_metrics"]
    hs = scale["headline_metrics"]
    hf = framework["headline_metrics"]

    feature_term_v57 = (
        hv["feature_term_v56"]
        + hv["feature_term_v56"] * hm["local_patch_encoding"] * 0.01
        + hv["feature_term_v56"] * hm["topology_selective_gate"] * 0.005
    )
    structure_term_v57 = (
        hv["structure_term_v56"]
        + hv["structure_term_v56"] * hm["route_superposition_binding"] * 0.01
        + hv["structure_term_v56"] * hs["scale_modular_reuse"] * 0.005
    )
    learning_term_v57 = (
        hv["learning_term_v56"]
        + hv["learning_term_v56"] * hs["scale_ready_score"]
        + hf["framework_synthesis_margin"] * 1000.0
    )
    pressure_term_v57 = max(
        0.0,
        hv["pressure_term_v56"] + hs["scale_collision_penalty"] + hf["critical_bottleneck"] - hf["language_anchor"] * 0.1,
    )
    encoding_margin_v57 = feature_term_v57 + structure_term_v57 + learning_term_v57 - pressure_term_v57

    return {
        "headline_metrics": {
            "feature_term_v57": feature_term_v57,
            "structure_term_v57": structure_term_v57,
            "learning_term_v57": learning_term_v57,
            "pressure_term_v57": pressure_term_v57,
            "encoding_margin_v57": encoding_margin_v57,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v57 = K_f_v56 + K_f_v56 * E_patch * 0.01 + K_f_v56 * E_gate * 0.005",
            "structure_term": "K_s_v57 = K_s_v56 + K_s_v56 * E_route * 0.01 + K_s_v56 * S_reuse * 0.005",
            "learning_term": "K_l_v57 = K_l_v56 + K_l_v56 * M_scale + M_framework * 1000",
            "pressure_term": "P_v57 = P_v56 + P_collision + G_bottleneck - 0.1 * F_lang_anchor",
            "margin_term": "M_encoding_v57 = K_f_v57 + K_s_v57 + K_l_v57 - P_v57",
        },
        "project_readout": {
            "summary": "第五十七版主核第一次把三维拓扑编码机制、三维规模化判断和整个项目框架总整理一起并回主式，使语言系统到脑编码机制的破解路径更像一条完整工程链。",
            "next_question": "下一步要把这条工程链从总结公式推进到可训练脉冲原型，不然当前仍然是解释强于构造。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第五十七版报告",
        "",
        f"- feature_term_v57: {hm['feature_term_v57']:.6f}",
        f"- structure_term_v57: {hm['structure_term_v57']:.6f}",
        f"- learning_term_v57: {hm['learning_term_v57']:.6f}",
        f"- pressure_term_v57: {hm['pressure_term_v57']:.6f}",
        f"- encoding_margin_v57: {hm['encoding_margin_v57']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v57_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
