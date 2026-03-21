from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_language_total_analysis_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_language_total_analysis_summary() -> dict:
    lang_system = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_system_principles_20260320" / "summary.json"
    )
    centrality = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_centrality_analysis_20260320" / "summary.json"
    )
    topology = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_region_topology_analysis_20260320" / "summary.json"
    )
    attr = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_region_attribute_analysis_20260320" / "summary.json"
    )
    color = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_color_fiber_nativeization_20260320" / "summary.json"
    )

    hs = lang_system["headline_metrics"]
    hc = centrality["headline_metrics"]
    ht = topology["headline_metrics"]
    ha = attr["headline_metrics"]
    hred = color["headline_metrics"]

    language_principle_completion = (
        hs["language_entry_core"]
        + hs["language_feature_core"]
        + hs["language_structure_core"]
        + hs["language_learning_core"]
        + (1.0 - hs["language_pressure_core"])
    ) / 5.0
    language_structure_resolution = (hs["language_structure_core"] + ht["region_topology_margin"]) / 2.0
    language_feature_resolution = (hs["language_feature_core"] + ha["cross_region_attribute_strength"] + hred["native_color_fiber"]) / 3.0
    language_transport_resolution = (hc["language_bridge_power"] + ha["attribute_distributed_score"]) / 2.0
    language_total_margin = (
        language_principle_completion
        + language_structure_resolution
        + language_feature_resolution
        + language_transport_resolution
        + hc["language_centrality"]
    ) / 5.0
    language_remaining_gap = 1.0 - min(1.0, language_total_margin)

    return {
        "headline_metrics": {
            "language_principle_completion": language_principle_completion,
            "language_structure_resolution": language_structure_resolution,
            "language_feature_resolution": language_feature_resolution,
            "language_transport_resolution": language_transport_resolution,
            "language_total_margin": language_total_margin,
            "language_remaining_gap": language_remaining_gap,
        },
        "language_total_equation": {
            "principle_term": "L_principle = mean(E_lang, F_lang, S_lang, L_lang, 1 - P_lang)",
            "structure_term": "L_structure = mean(S_lang, M_region)",
            "feature_term": "L_feature = mean(F_lang, D_attr, M_red)",
            "transport_term": "L_transport = mean(B_lang, D_attr)",
            "system_term": "M_language_total = mean(L_principle, L_structure, L_feature, L_transport, C_lang)",
        },
        "project_readout": {
            "summary": "语言分析当前已经不只是入口层强，而是入口、特征、结构、运输四层一起开始收口。",
            "next_question": "下一步要把语言系统继续推进到更原生的回路变量，确认当前语言闭合不是中层几何假象。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 语言总分析报告",
        "",
        f"- language_principle_completion: {hm['language_principle_completion']:.6f}",
        f"- language_structure_resolution: {hm['language_structure_resolution']:.6f}",
        f"- language_feature_resolution: {hm['language_feature_resolution']:.6f}",
        f"- language_transport_resolution: {hm['language_transport_resolution']:.6f}",
        f"- language_total_margin: {hm['language_total_margin']:.6f}",
        f"- language_remaining_gap: {hm['language_remaining_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_language_total_analysis_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
