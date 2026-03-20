from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_concept_formation_closed_form_v4_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_concept_formation_closed_form_v4_summary() -> dict:
    v3 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_formation_closed_form_v3_20260320" / "summary.json")
    closure = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_cross_asset_closure_20260320" / "summary.json")
    fiber_primary = _load_json(ROOT / "tests" / "codex_temp" / "stage56_local_fiber_primary_term_20260320" / "summary.json")
    circuit_v2 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_circuit_bridge_v2_20260320" / "summary.json")

    hv3 = v3["headline_metrics"]
    hcl = closure["headline_metrics"]
    hfp = fiber_primary["headline_metrics"]
    hcb = circuit_v2["headline_metrics"]

    anchor_chart_term_v4 = hv3["anchor_chart_term_v3"] + hcl["closure_support"]
    local_primary_term_v4 = hv3["strengthened_fiber_term_v3"] + hfp["local_primary_margin"]
    circuit_term_v4 = hcb["concept_circuit_margin_v2"] / (1.0 + hcb["concept_circuit_margin_v2"])
    pressure_term_v4 = hv3["pressure_term_v3"] + hcl["gap_penalty"]
    concept_margin_v4 = anchor_chart_term_v4 + local_primary_term_v4 + circuit_term_v4 - pressure_term_v4

    return {
        "headline_metrics": {
            "anchor_chart_term_v4": anchor_chart_term_v4,
            "local_primary_term_v4": local_primary_term_v4,
            "circuit_term_v4": circuit_term_v4,
            "pressure_term_v4": pressure_term_v4,
            "concept_margin_v4": concept_margin_v4,
        },
        "closed_form_equation": {
            "anchor_chart_term": "AC_v4 = anchor_chart_term_v3 + closure_support",
            "local_term": "L_v4 = strengthened_fiber_term_v3 + local_primary_margin",
            "circuit_term": "C_v4 = concept_circuit_margin_v2 / (1 + concept_circuit_margin_v2)",
            "pressure_term": "P_v4 = pressure_term_v3 + gap_penalty",
            "margin_term": "M_concept_v4 = AC_v4 + L_v4 + C_v4 - P_v4",
        },
        "project_readout": {
            "summary": "这一轮把跨资产收口、局部纤维主项和回路桥接第二版并回概念形成核，让概念形成第四版开始同时容纳图册、局部差分纤维、跨资产稳定性和回路级对象。",
            "next_question": "下一步要检查第四版概念形成核，是否足够稳定到可以作为阶段性最终概念形成核。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 概念形成闭式第四版报告",
        "",
        f"- anchor_chart_term_v4: {hm['anchor_chart_term_v4']:.6f}",
        f"- local_primary_term_v4: {hm['local_primary_term_v4']:.6f}",
        f"- circuit_term_v4: {hm['circuit_term_v4']:.6f}",
        f"- pressure_term_v4: {hm['pressure_term_v4']:.6f}",
        f"- concept_margin_v4: {hm['concept_margin_v4']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_concept_formation_closed_form_v4_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
