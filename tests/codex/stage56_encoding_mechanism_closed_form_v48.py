from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v48_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v48_summary() -> dict:
    v47 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v47_20260320" / "summary.json"
    )
    region = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_region_topology_analysis_20260320" / "summary.json"
    )
    attr = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_region_attribute_analysis_20260320" / "summary.json"
    )
    sparse = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_sparse_activation_region_analysis_20260320" / "summary.json"
    )

    hv = v47["headline_metrics"]
    hr = region["headline_metrics"]
    ha = attr["headline_metrics"]
    hs = sparse["headline_metrics"]

    feature_term_v48 = hv["feature_term_v47"] + hv["feature_term_v47"] * ha["attribute_distributed_score"] * 0.01
    structure_term_v48 = hv["structure_term_v47"] + hv["structure_term_v47"] * hr["region_topology_margin"] * 0.01
    learning_term_v48 = hv["learning_term_v47"] + hv["learning_term_v47"] * hs["sparse_activation_efficiency"] + hr["region_topology_margin"] * 1000.0
    pressure_term_v48 = max(
        0.0,
        hv["pressure_term_v47"] + ha["attribute_single_region_score"] + hr["local_offset_mobility"] - hs["sparse_seed_activation"]
    )
    encoding_margin_v48 = feature_term_v48 + structure_term_v48 + learning_term_v48 - pressure_term_v48

    return {
        "headline_metrics": {
            "feature_term_v48": feature_term_v48,
            "structure_term_v48": structure_term_v48,
            "learning_term_v48": learning_term_v48,
            "pressure_term_v48": pressure_term_v48,
            "encoding_margin_v48": encoding_margin_v48,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v48 = K_f_v47 + K_f_v47 * D_attr * 0.01",
            "structure_term": "K_s_v48 = K_s_v47 + K_s_v47 * M_region * 0.01",
            "learning_term": "K_l_v48 = K_l_v47 + K_l_v47 * A_sparse + M_region * 1000",
            "pressure_term": "P_v48 = P_v47 + S_attr + R_offset - A_seed",
            "margin_term": "M_encoding_v48 = K_f_v48 + K_s_v48 + K_l_v48 - P_v48",
        },
        "project_readout": {
            "summary": "第四十八版主核把区域拓扑、跨区域属性和稀疏激活一起并回主式，用来表达概念区域、横跨属性和最小激活组合在同一系统中的关系。",
            "next_question": "下一步要继续把这些区域对象推进到更原生的回路变量，确认当前区域拓扑不是中层几何假象，而是真实可实现的神经组织结构。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第四十八版报告",
        "",
        f"- feature_term_v48: {hm['feature_term_v48']:.6f}",
        f"- structure_term_v48: {hm['structure_term_v48']:.6f}",
        f"- learning_term_v48: {hm['learning_term_v48']:.6f}",
        f"- pressure_term_v48: {hm['pressure_term_v48']:.6f}",
        f"- encoding_margin_v48: {hm['encoding_margin_v48']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v48_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
