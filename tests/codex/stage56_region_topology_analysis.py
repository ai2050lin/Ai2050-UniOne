from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_region_topology_analysis_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_region_topology_summary() -> dict:
    charts = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_concept_local_chart_expansion_20260320" / "summary.json"
    )
    attrs = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_attribute_fiber_nativeization_20260320" / "summary.json"
    )
    lang = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_system_principles_20260320" / "summary.json"
    )

    hc = charts["headline_metrics"]
    ha = attrs["headline_metrics"]
    hl = lang["headline_metrics"]

    family_region_density = hc["mean_anchor_strength"] * (1.0 - hc["mean_chart_compactness"])
    family_region_separation = _clip01(hc["mean_separation_gap"] / 2.5)
    local_offset_mobility = hc["mean_chart_support"]
    regional_overlap_control = 1.0 - hc["mean_chart_compactness"]
    region_topology_margin = (
        family_region_density
        + family_region_separation
        + local_offset_mobility
        + regional_overlap_control
        + hl["language_structure_core"]
    ) / 5.0

    return {
        "headline_metrics": {
            "family_region_density": family_region_density,
            "family_region_separation": family_region_separation,
            "local_offset_mobility": local_offset_mobility,
            "regional_overlap_control": regional_overlap_control,
            "region_topology_margin": region_topology_margin,
            "family_count": hc["family_count"],
            "mean_anchor_bundle_strength": ha["mean_anchor_bundle_strength"],
            "mean_local_bundle_strength": ha["mean_local_bundle_strength"],
        },
        "topology_equation": {
            "density_term": "R_family = A_family * (1 - C_chart)",
            "separation_term": "R_sep = norm(G_sep)",
            "offset_term": "R_offset = S_chart",
            "overlap_term": "R_overlap = 1 - C_chart",
            "system_term": "M_region = mean(R_family, R_sep, R_offset, R_overlap, S_lang)",
        },
        "project_readout": {
            "summary": "概念区域当前最像高密度家族片区加局部偏置子区的组合，而不是一个个互不相连的孤立块。",
            "next_question": "下一步要把这种区域拓扑继续推进到更原生的回路变量，确认这些区域不是几何假象，而是稀疏激活下的真实功能分区。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 区域拓扑分析报告",
        "",
        f"- family_region_density: {hm['family_region_density']:.6f}",
        f"- family_region_separation: {hm['family_region_separation']:.6f}",
        f"- local_offset_mobility: {hm['local_offset_mobility']:.6f}",
        f"- regional_overlap_control: {hm['regional_overlap_control']:.6f}",
        f"- region_topology_margin: {hm['region_topology_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_region_topology_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
