from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_system_region_distribution_analysis_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_system_region_distribution_summary() -> dict:
    region = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_region_topology_analysis_20260320" / "summary.json"
    )
    attr = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_region_attribute_analysis_20260320" / "summary.json"
    )
    sparse = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_sparse_activation_region_analysis_20260320" / "summary.json"
    )
    color = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_color_pathway_overlap_analysis_20260320" / "summary.json"
    )

    hr = region["headline_metrics"]
    ha = attr["headline_metrics"]
    hs = sparse["headline_metrics"]
    hc = color["headline_metrics"]

    family_patch_mass = hr["family_region_density"]
    local_subregion_mass = hr["local_offset_mobility"] + hr["mean_local_bundle_strength"]
    transverse_attribute_mass = ha["attribute_distributed_score"]
    route_channel_mass = hs["sparse_route_activation"]
    contextual_split_mass = hc["contextual_split_score"]
    system_distribution_margin = (
        family_patch_mass
        + local_subregion_mass
        + transverse_attribute_mass
        + route_channel_mass
        + contextual_split_mass
    ) / 5.0

    return {
        "headline_metrics": {
            "family_patch_mass": family_patch_mass,
            "local_subregion_mass": local_subregion_mass,
            "transverse_attribute_mass": transverse_attribute_mass,
            "route_channel_mass": route_channel_mass,
            "contextual_split_mass": contextual_split_mass,
            "system_distribution_margin": system_distribution_margin,
        },
        "distribution_equation": {
            "family_term": "D_family = R_family",
            "local_term": "D_local = R_offset + B_local",
            "attribute_term": "D_attr = D_attr",
            "route_term": "D_route = A_route",
            "context_term": "D_ctx = contextual_split",
        },
        "project_readout": {
            "summary": "整个系统当前最像家族片区、局部偏置子区、横跨属性纤维、路线通道和上下文分叉五种区域对象叠加组成的分层网络。",
            "next_question": "下一步要把这五种区域对象推进到更原生的回路变量，确认它们是不是神经回路级可落地结构。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 系统区域分布分析报告",
        "",
        f"- family_patch_mass: {hm['family_patch_mass']:.6f}",
        f"- local_subregion_mass: {hm['local_subregion_mass']:.6f}",
        f"- transverse_attribute_mass: {hm['transverse_attribute_mass']:.6f}",
        f"- route_channel_mass: {hm['route_channel_mass']:.6f}",
        f"- contextual_split_mass: {hm['contextual_split_mass']:.6f}",
        f"- system_distribution_margin: {hm['system_distribution_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_system_region_distribution_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
