from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_concept_chart_cross_asset_validation_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_concept_chart_cross_asset_validation_summary() -> dict:
    charts = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_local_chart_expansion_20260320" / "summary.json")
    concept = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_encoding_formation_20260320" / "summary.json")
    closed_v2 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_formation_closed_form_v2_20260320" / "summary.json")
    cross_asset = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_kernel_cross_asset_validation_20260320" / "summary.json")

    chm = charts["headline_metrics"]
    cm = concept["headline_metrics"]
    cv2 = closed_v2["headline_metrics"]
    xhm = cross_asset["headline_metrics"]

    chart_family_support = chm["mean_anchor_strength"] + chm["mean_chart_support"]
    chart_separation_support = chm["mean_separation_gap"] / (1.0 + chm["mean_separation_gap"])
    concept_transfer_support = cm["apple_banana_transfer_support"]
    concept_form_support = cv2["concept_margin_v2"] / (1.0 + cv2["concept_margin_v2"])
    cross_asset_support_v2 = (
        chart_family_support
        + chart_separation_support
        + concept_transfer_support
        + concept_form_support
        + xhm["cross_asset_support"]
    ) / 5.0
    support_gap_v2 = max(
        chart_family_support,
        chart_separation_support,
        concept_transfer_support,
        concept_form_support,
        xhm["cross_asset_support"],
    ) - min(
        chart_family_support,
        chart_separation_support,
        concept_transfer_support,
        concept_form_support,
        xhm["cross_asset_support"],
    )

    return {
        "headline_metrics": {
            "chart_family_support": chart_family_support,
            "chart_separation_support": chart_separation_support,
            "concept_transfer_support": concept_transfer_support,
            "concept_form_support": concept_form_support,
            "cross_asset_support_v2": cross_asset_support_v2,
            "support_gap_v2": support_gap_v2,
        },
        "validation_equation": {
            "family_term": "S_family = mean_anchor_strength + mean_chart_support",
            "separation_term": "S_sep = mean_separation_gap / (1 + mean_separation_gap)",
            "transfer_term": "S_transfer = apple_banana_transfer_support",
            "form_term": "S_form = concept_margin_v2 / (1 + concept_margin_v2)",
            "total_term": "S_concept_cross = mean(S_family, S_sep, S_transfer, S_form, S_cross_asset)",
        },
        "project_readout": {
            "summary": (
                "这一轮把概念局部图册、概念形成核和已有跨资产编码核支持度并场，"
                "检查概念形成链能否跨资产保持稳定方向。"
            ),
            "next_question": (
                "下一步要继续压缩 support_gap_v2，"
                "让概念形成核不仅方向跨资产成立，而且强度也能收口。"
            ),
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 概念图册跨资产验证报告",
        "",
        f"- chart_family_support: {hm['chart_family_support']:.6f}",
        f"- chart_separation_support: {hm['chart_separation_support']:.6f}",
        f"- concept_transfer_support: {hm['concept_transfer_support']:.6f}",
        f"- concept_form_support: {hm['concept_form_support']:.6f}",
        f"- cross_asset_support_v2: {hm['cross_asset_support_v2']:.6f}",
        f"- support_gap_v2: {hm['support_gap_v2']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_concept_chart_cross_asset_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
