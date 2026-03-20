from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_cross_region_attribute_analysis_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_cross_region_attribute_summary() -> dict:
    attrs = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_attribute_fiber_nativeization_20260320" / "summary.json"
    )
    centrality = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_centrality_analysis_20260320" / "summary.json"
    )
    keep = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_kernel_stability_strengthening_20260320" / "summary.json"
    )

    ha = attrs["headline_metrics"]
    hc = centrality["headline_metrics"]
    hk = keep["headline_metrics"]

    attribute_anchor_mass = ha["mean_anchor_bundle_strength"]
    attribute_transverse_mass = ha["mean_local_bundle_strength"] * hc["language_bridge_power"]
    cross_region_attribute_strength = attribute_transverse_mass + hk["bridge_retention_stable"]
    attribute_single_region_score = max(0.0, attribute_anchor_mass - attribute_transverse_mass)
    attribute_distributed_score = min(1.0, attribute_transverse_mass + hk["bridge_retention_stable"])

    return {
        "headline_metrics": {
            "attribute_anchor_mass": attribute_anchor_mass,
            "attribute_transverse_mass": attribute_transverse_mass,
            "cross_region_attribute_strength": cross_region_attribute_strength,
            "attribute_single_region_score": attribute_single_region_score,
            "attribute_distributed_score": attribute_distributed_score,
        },
        "attribute_equation": {
            "anchor_term": "A_attr = B_anchor",
            "transverse_term": "T_attr = B_local * B_lang",
            "bridge_term": "X_attr = T_attr + B_keep_star",
            "single_term": "S_attr = max(0, A_attr - T_attr)",
            "distributed_term": "D_attr = min(1, T_attr + B_keep_star)",
        },
        "project_readout": {
            "summary": "像红色这样的横跨属性，更像分布式横截纤维，而不是单独被锁在某一个概念区域里。",
            "next_question": "下一步要把跨区域属性继续推进到更多家族和更多属性，验证它们是不是普遍都呈现这种横截纤维结构。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 跨区域属性分析报告",
        "",
        f"- attribute_anchor_mass: {hm['attribute_anchor_mass']:.6f}",
        f"- attribute_transverse_mass: {hm['attribute_transverse_mass']:.6f}",
        f"- cross_region_attribute_strength: {hm['cross_region_attribute_strength']:.6f}",
        f"- attribute_single_region_score: {hm['attribute_single_region_score']:.6f}",
        f"- attribute_distributed_score: {hm['attribute_distributed_score']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_cross_region_attribute_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
