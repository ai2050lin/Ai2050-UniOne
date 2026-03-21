from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v55_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v55_summary() -> dict:
    v54 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v54_20260321" / "summary.json"
    )
    brain_v3 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v3_20260321" / "summary.json"
    )
    horizon = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_long_horizon_stability_20260321" / "summary.json"
    )

    hv = v54["headline_metrics"]
    hb = brain_v3["headline_metrics"]
    hh = horizon["headline_metrics"]

    feature_term_v55 = hv["feature_term_v54"] + hv["feature_term_v54"] * hb["direct_feature_measure_v3"] * 0.01
    structure_term_v55 = hv["structure_term_v54"] + hv["structure_term_v54"] * hb["direct_structure_measure_v3"] * 0.01
    learning_term_v55 = hv["learning_term_v54"] + hv["learning_term_v54"] * hh["long_horizon_plasticity"] + hh["long_horizon_margin"] * 1000.0
    pressure_term_v55 = max(
        0.0,
        hv["pressure_term_v54"]
        + hb["direct_brain_gap_v3"]
        + hh["cumulative_rollback"]
        - hh["structural_survival"]
        - hh["shared_fiber_survival"],
    )
    encoding_margin_v55 = feature_term_v55 + structure_term_v55 + learning_term_v55 - pressure_term_v55

    return {
        "headline_metrics": {
            "feature_term_v55": feature_term_v55,
            "structure_term_v55": structure_term_v55,
            "learning_term_v55": learning_term_v55,
            "pressure_term_v55": pressure_term_v55,
            "encoding_margin_v55": encoding_margin_v55,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v55 = K_f_v54 + K_f_v54 * D_feature_v3 * 0.01",
            "structure_term": "K_s_v55 = K_s_v54 + K_s_v54 * D_structure_v3 * 0.01",
            "learning_term": "K_l_v55 = K_l_v54 + K_l_v54 * G_h + M_h * 1000",
            "pressure_term": "P_v55 = P_v54 + G_direct_v3 + P_h - H_structure - H_fiber",
            "margin_term": "M_encoding_v55 = K_f_v55 + K_s_v55 + K_l_v55 - P_v55",
        },
        "project_readout": {
            "summary": "第五十五版主核把长时间尺度即时学习稳定性和脑编码直测强化第三版一起并回主式，开始表达多轮更新下结构是否仍能保持。",
            "next_question": "下一步要把这一版主核继续推进到更接近训练终式的规则层，而不是只停留在多轮更新评估层。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第五十五版报告",
        "",
        f"- feature_term_v55: {hm['feature_term_v55']:.6f}",
        f"- structure_term_v55: {hm['structure_term_v55']:.6f}",
        f"- learning_term_v55: {hm['learning_term_v55']:.6f}",
        f"- pressure_term_v55: {hm['pressure_term_v55']:.6f}",
        f"- encoding_margin_v55: {hm['encoding_margin_v55']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v55_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
