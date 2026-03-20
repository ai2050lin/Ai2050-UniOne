from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v49_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v49_summary() -> dict:
    v48 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v48_20260320" / "summary.json"
    )
    color = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_color_pathway_overlap_analysis_20260320" / "summary.json"
    )
    distribution = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_system_region_distribution_analysis_20260320" / "summary.json"
    )

    hv = v48["headline_metrics"]
    hc = color["headline_metrics"]
    hd = distribution["headline_metrics"]

    feature_term_v49 = hv["feature_term_v48"] + hv["feature_term_v48"] * hc["shared_fiber_score"] * 0.01
    structure_term_v49 = hv["structure_term_v48"] + hv["structure_term_v48"] * hd["system_distribution_margin"] * 0.01
    learning_term_v49 = hv["learning_term_v48"] + hv["learning_term_v48"] * hc["full_path_overlap"] + hd["system_distribution_margin"] * 1000.0
    pressure_term_v49 = max(
        0.0,
        hv["pressure_term_v48"] + hc["same_full_route_score"] + hd["contextual_split_mass"] - hc["shared_fiber_score"]
    )
    encoding_margin_v49 = feature_term_v49 + structure_term_v49 + learning_term_v49 - pressure_term_v49

    return {
        "headline_metrics": {
            "feature_term_v49": feature_term_v49,
            "structure_term_v49": structure_term_v49,
            "learning_term_v49": learning_term_v49,
            "pressure_term_v49": pressure_term_v49,
            "encoding_margin_v49": encoding_margin_v49,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v49 = K_f_v48 + K_f_v48 * X_red * 0.01",
            "structure_term": "K_s_v49 = K_s_v48 + K_s_v48 * D_system * 0.01",
            "learning_term": "K_l_v49 = K_l_v48 + K_l_v48 * O_path + D_system * 1000",
            "pressure_term": "P_v49 = P_v48 + R_same + D_ctx - X_red",
            "margin_term": "M_encoding_v49 = K_f_v49 + K_s_v49 + K_l_v49 - P_v49",
        },
        "project_readout": {
            "summary": "第四十九版主核把颜色共享纤维、上下文分叉和系统区域分布一起并回主式，用来表达同一属性在不同概念中的共享通路与分叉通路关系。",
            "next_question": "下一步要把这种共享纤维加上下文分叉的结构推进到更多属性和更多对象家族，确认它是不是普遍机制。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第四十九版报告",
        "",
        f"- feature_term_v49: {hm['feature_term_v49']:.6f}",
        f"- structure_term_v49: {hm['structure_term_v49']:.6f}",
        f"- learning_term_v49: {hm['learning_term_v49']:.6f}",
        f"- pressure_term_v49: {hm['pressure_term_v49']:.6f}",
        f"- encoding_margin_v49: {hm['encoding_margin_v49']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v49_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
