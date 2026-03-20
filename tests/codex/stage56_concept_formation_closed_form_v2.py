from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_concept_formation_closed_form_v2_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_concept_formation_closed_form_v2_summary() -> dict:
    concept = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_encoding_formation_20260320" / "summary.json")
    charts = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_local_chart_expansion_20260320" / "summary.json")
    fibers = _load_json(ROOT / "tests" / "codex_temp" / "stage56_attribute_fiber_nativeization_20260320" / "summary.json")

    ch = concept["headline_metrics"]
    hm_chart = charts["headline_metrics"]
    hm_fiber = fibers["headline_metrics"]

    family_anchor_term = ch["family_anchor_strength"] + 0.5 * hm_fiber["mean_anchor_bundle_strength"]
    local_chart_term = ch["apple_local_offset_norm"] + hm_chart["mean_chart_support"] + hm_chart["mean_separation_gap"]
    local_fiber_term = abs(hm_fiber["apple_round_local_coeff"]) + abs(hm_fiber["apple_elongated_local_coeff"])
    formation_pressure_term = ch["concept_pressure"] + hm_chart["mean_reconstruction_error"]
    concept_margin_v2 = family_anchor_term + local_chart_term + local_fiber_term - formation_pressure_term

    return {
        "headline_metrics": {
            "family_anchor_term": family_anchor_term,
            "local_chart_term": local_chart_term,
            "local_fiber_term": local_fiber_term,
            "formation_pressure_term": formation_pressure_term,
            "concept_margin_v2": concept_margin_v2,
        },
        "closed_form_equation": {
            "anchor_term": "A_concept = family_anchor + 0.5 * anchor_bundle",
            "chart_term": "C_chart = local_offset + chart_support + separation_gap",
            "fiber_term": "F_local = |round_coeff| + |elongated_coeff|",
            "pressure_term": "P_form = structural_pressure + chart_reconstruction_error",
            "margin_term": "M_concept_v2 = A_concept + C_chart + F_local - P_form",
        },
        "project_readout": {
            "summary": (
                "这一轮把家族锚点、多家族局部图册和属性纤维并回同一个概念形成闭式候选，"
                "让“苹果这样的概念如何形成”开始从单案例解释推进到更一般的概念形成核。"
            ),
            "next_question": (
                "下一步要检查这个第二版概念形成核，是否能跨更多家族和更多概念稳定成立，"
                "而不是只对水果家族有效。"
            ),
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 概念形成闭式第二版报告",
        "",
        f"- family_anchor_term: {hm['family_anchor_term']:.6f}",
        f"- local_chart_term: {hm['local_chart_term']:.6f}",
        f"- local_fiber_term: {hm['local_fiber_term']:.6f}",
        f"- formation_pressure_term: {hm['formation_pressure_term']:.6f}",
        f"- concept_margin_v2: {hm['concept_margin_v2']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_concept_formation_closed_form_v2_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
