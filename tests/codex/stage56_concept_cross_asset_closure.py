from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_concept_cross_asset_closure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_concept_cross_asset_closure_summary() -> dict:
    cross = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_chart_cross_asset_validation_20260320" / "summary.json")
    v3 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_formation_closed_form_v3_20260320" / "summary.json")

    xhm = cross["headline_metrics"]
    vhm = v3["headline_metrics"]

    support_consensus = (xhm["chart_separation_support"] + xhm["concept_transfer_support"] + xhm["concept_form_support"]) / 3.0
    gap_penalty = xhm["support_gap_v2"] / (1.0 + xhm["cross_asset_support_v2"])
    closure_support = (xhm["cross_asset_support_v2"] + support_consensus + vhm["cross_asset_term_v3"]) / 3.0
    closure_margin = closure_support - gap_penalty

    return {
        "headline_metrics": {
            "support_consensus": support_consensus,
            "gap_penalty": gap_penalty,
            "closure_support": closure_support,
            "closure_margin": closure_margin,
        },
        "closure_equation": {
            "consensus_term": "C_consensus = mean(chart_separation_support, concept_transfer_support, concept_form_support)",
            "gap_term": "P_gap = support_gap_v2 / (1 + cross_asset_support_v2)",
            "support_term": "S_closure = mean(cross_asset_support_v2, C_consensus, cross_asset_term_v3)",
            "margin_term": "M_closure = S_closure - P_gap",
        },
        "project_readout": {
            "summary": "这一轮把概念形成的跨资产支持度继续压缩成一个更短的收口对象，重点不再是证明方向成立，而是直接看强度差能不能继续缩小。",
            "next_question": "下一步最关键的是把这个收口边距并回概念形成核，检查它能否进一步压低跨资产不稳定性。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 概念跨资产收口报告",
        "",
        f"- support_consensus: {hm['support_consensus']:.6f}",
        f"- gap_penalty: {hm['gap_penalty']:.6f}",
        f"- closure_support: {hm['closure_support']:.6f}",
        f"- closure_margin: {hm['closure_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_concept_cross_asset_closure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
