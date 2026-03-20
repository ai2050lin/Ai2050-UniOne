from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_concept_cross_asset_final_closure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_concept_cross_asset_final_closure_summary() -> dict:
    closure = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_cross_asset_closure_20260320" / "summary.json")
    cross_asset = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_concept_chart_cross_asset_validation_20260320" / "summary.json"
    )
    v4 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_formation_closed_form_v4_20260320" / "summary.json")

    hcl = closure["headline_metrics"]
    hca = cross_asset["headline_metrics"]
    hv4 = v4["headline_metrics"]

    support_floor = min(
        hca["cross_asset_support_v2"],
        hca["concept_form_support"],
        hcl["support_consensus"],
    )
    support_spread = hca["support_gap_v2"] / (1.0 + hcl["closure_support"])
    final_closure_support = 0.5 * (hcl["closure_support"] + support_floor)
    final_gap_penalty = hcl["gap_penalty"] / (1.0 + support_floor)
    final_closure_margin = final_closure_support - final_gap_penalty
    closure_to_margin_ratio = final_closure_margin / max(hv4["concept_margin_v4"], 1e-9)

    return {
        "headline_metrics": {
            "support_floor": support_floor,
            "support_spread": support_spread,
            "final_closure_support": final_closure_support,
            "final_gap_penalty": final_gap_penalty,
            "final_closure_margin": final_closure_margin,
            "closure_to_margin_ratio": closure_to_margin_ratio,
        },
        "final_closure_equation": {
            "floor_term": "S_floor = min(cross_asset_support_v2, concept_form_support, support_consensus)",
            "spread_term": "P_spread = support_gap_v2 / (1 + closure_support)",
            "support_term": "S_final = 0.5 * (closure_support + S_floor)",
            "penalty_term": "P_final = gap_penalty / (1 + S_floor)",
            "margin_term": "M_final_closure = S_final - P_final",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 概念形成跨资产最终收口报告",
        "",
        f"- support_floor: {hm['support_floor']:.6f}",
        f"- support_spread: {hm['support_spread']:.6f}",
        f"- final_closure_support: {hm['final_closure_support']:.6f}",
        f"- final_gap_penalty: {hm['final_gap_penalty']:.6f}",
        f"- final_closure_margin: {hm['final_closure_margin']:.6f}",
        f"- closure_to_margin_ratio: {hm['closure_to_margin_ratio']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_concept_cross_asset_final_closure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
