from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_concept_formation_closed_form_v3_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_concept_formation_closed_form_v3_summary() -> dict:
    v2 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_formation_closed_form_v2_20260320" / "summary.json")
    cross_asset = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_chart_cross_asset_validation_20260320" / "summary.json")
    local_fiber = _load_json(ROOT / "tests" / "codex_temp" / "stage56_local_differential_fiber_strengthening_20260320" / "summary.json")

    hv2 = v2["headline_metrics"]
    hca = cross_asset["headline_metrics"]
    hlf = local_fiber["headline_metrics"]

    anchor_chart_term_v3 = hv2["family_anchor_term"] + hv2["local_chart_term"]
    strengthened_fiber_term_v3 = hv2["local_fiber_term"] + hlf["mean_strengthened_local_fiber"]
    cross_asset_term_v3 = hca["cross_asset_support_v2"]
    pressure_term_v3 = hv2["formation_pressure_term"] + 0.5 * hca["support_gap_v2"]
    concept_margin_v3 = anchor_chart_term_v3 + strengthened_fiber_term_v3 + cross_asset_term_v3 - pressure_term_v3

    return {
        "headline_metrics": {
            "anchor_chart_term_v3": anchor_chart_term_v3,
            "strengthened_fiber_term_v3": strengthened_fiber_term_v3,
            "cross_asset_term_v3": cross_asset_term_v3,
            "pressure_term_v3": pressure_term_v3,
            "concept_margin_v3": concept_margin_v3,
        },
        "closed_form_equation": {
            "anchor_chart_term": "AC_v3 = family_anchor_term + local_chart_term",
            "fiber_term": "F_v3 = local_fiber_term + strengthened_local_fiber",
            "cross_asset_term": "X_v3 = cross_asset_support_v2",
            "pressure_term": "P_v3 = formation_pressure_term + 0.5 * support_gap_v2",
            "margin_term": "M_concept_v3 = AC_v3 + F_v3 + X_v3 - P_v3",
        },
        "project_readout": {
            "summary": (
                "这一轮把跨资产支持和强化后的局部差分纤维并回概念形成核，"
                "让概念形成闭式第三版开始同时容纳家族图册、局部纤维和跨资产稳定性。"
            ),
            "next_question": (
                "下一步要检验概念形成第三版核，是否已经足够稳定到可以跨更多家族和更多资产成立。"
            ),
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 概念形成闭式第三版报告",
        "",
        f"- anchor_chart_term_v3: {hm['anchor_chart_term_v3']:.6f}",
        f"- strengthened_fiber_term_v3: {hm['strengthened_fiber_term_v3']:.6f}",
        f"- cross_asset_term_v3: {hm['cross_asset_term_v3']:.6f}",
        f"- pressure_term_v3: {hm['pressure_term_v3']:.6f}",
        f"- concept_margin_v3: {hm['concept_margin_v3']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_concept_formation_closed_form_v3_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
