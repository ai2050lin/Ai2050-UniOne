from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v54_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v54_summary() -> dict:
    v53 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v53_20260320" / "summary.json"
    )
    brain_v2 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v2_20260321" / "summary.json"
    )
    online = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_rollback_probe_20260321" / "summary.json"
    )

    hv = v53["headline_metrics"]
    hb = brain_v2["headline_metrics"]
    ho = online["headline_metrics"]

    feature_term_v54 = hv["feature_term_v53"] + hv["feature_term_v53"] * hb["direct_feature_measure_v2"] * 0.01
    structure_term_v54 = hv["structure_term_v53"] + hv["structure_term_v53"] * hb["direct_structure_measure_v2"] * 0.01
    learning_term_v54 = hv["learning_term_v53"] + hv["learning_term_v53"] * ho["online_gain"] + ho["online_learning_margin"] * 1000.0
    pressure_term_v54 = max(
        0.0,
        hv["pressure_term_v53"]
        + hb["direct_brain_gap_v2"]
        + ho["rollback_penalty"]
        + ho["shared_attribute_drift"]
        - ho["route_split_retention"],
    )
    encoding_margin_v54 = feature_term_v54 + structure_term_v54 + learning_term_v54 - pressure_term_v54

    return {
        "headline_metrics": {
            "feature_term_v54": feature_term_v54,
            "structure_term_v54": structure_term_v54,
            "learning_term_v54": learning_term_v54,
            "pressure_term_v54": pressure_term_v54,
            "encoding_margin_v54": encoding_margin_v54,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v54 = K_f_v53 + K_f_v53 * D_feature_v2 * 0.01",
            "structure_term": "K_s_v54 = K_s_v53 + K_s_v53 * D_structure_v2 * 0.01",
            "learning_term": "K_l_v54 = K_l_v53 + K_l_v53 * G_online + M_online * 1000",
            "pressure_term": "P_v54 = P_v53 + G_direct_v2 + P_back + D_attr - R_route",
            "margin_term": "M_encoding_v54 = K_f_v54 + K_s_v54 + K_l_v54 - P_v54",
        },
        "project_readout": {
            "summary": "第五十四版主核把逆向脑编码直测强化和即时学习回落测试一起并回主式，用来表达从静态结构解释向动态更新稳定性的推进。",
            "next_question": "下一步要把这个主核继续推进到更长时间尺度的在线学习和更强的训练终式，而不是只停留在单轮在线注入上。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第五十四版报告",
        "",
        f"- feature_term_v54: {hm['feature_term_v54']:.6f}",
        f"- structure_term_v54: {hm['structure_term_v54']:.6f}",
        f"- learning_term_v54: {hm['learning_term_v54']:.6f}",
        f"- pressure_term_v54: {hm['pressure_term_v54']:.6f}",
        f"- encoding_margin_v54: {hm['encoding_margin_v54']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v54_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
