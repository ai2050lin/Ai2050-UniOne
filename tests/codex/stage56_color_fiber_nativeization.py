from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_color_fiber_nativeization_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_color_fiber_nativeization_summary() -> dict:
    color = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_color_pathway_overlap_analysis_20260320" / "summary.json"
    )
    attr = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_region_attribute_analysis_20260320" / "summary.json"
    )
    sparse = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_sparse_activation_region_analysis_20260320" / "summary.json"
    )

    hc = color["headline_metrics"]
    ha = attr["headline_metrics"]
    hs = sparse["headline_metrics"]

    native_color_fiber = hc["shared_color_core"] + ha["attribute_transverse_mass"]
    native_color_binding = hc["color_fiber_overlap"] * hs["sparse_feature_activation"]
    native_color_route_split = hc["contextual_split_score"] * hs["sparse_route_activation"]
    native_color_specificity = max(0.0, native_color_binding - 0.5 * native_color_route_split)
    color_native_margin = native_color_fiber + native_color_binding + native_color_specificity - native_color_route_split

    return {
        "headline_metrics": {
            "native_color_fiber": native_color_fiber,
            "native_color_binding": native_color_binding,
            "native_color_route_split": native_color_route_split,
            "native_color_specificity": native_color_specificity,
            "color_native_margin": color_native_margin,
        },
        "native_equation": {
            "fiber_term": "N_red = X_red + T_attr",
            "binding_term": "B_red = overlap(X_red) * A_feature",
            "split_term": "S_red = D_ctx * A_route",
            "specificity_term": "Q_red = max(0, B_red - 0.5 * S_red)",
            "margin_term": "M_red = N_red + B_red + Q_red - S_red",
        },
        "project_readout": {
            "summary": "红色当前已经不只是跨区域属性纤维，也开始能被写成共享纤维、绑定增强和路径分叉三部分组成的近原生对象。",
            "next_question": "下一步要把颜色纤维继续推进到更多颜色和更多对象家族，确认共享纤维加上下文分叉是不是普遍原理。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 颜色纤维原生化报告",
        "",
        f"- native_color_fiber: {hm['native_color_fiber']:.6f}",
        f"- native_color_binding: {hm['native_color_binding']:.6f}",
        f"- native_color_route_split: {hm['native_color_route_split']:.6f}",
        f"- native_color_specificity: {hm['native_color_specificity']:.6f}",
        f"- color_native_margin: {hm['color_native_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_color_fiber_nativeization_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
