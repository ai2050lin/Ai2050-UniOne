from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_concept_formation_closed_form_v5_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_concept_formation_closed_form_v5_summary() -> dict:
    v4 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_formation_closed_form_v4_20260320" / "summary.json")
    closure_final = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_concept_cross_asset_final_closure_20260320" / "summary.json"
    )
    local_structure = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_local_fiber_primary_structure_20260320" / "summary.json"
    )
    circuit_v3 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_circuit_bridge_v3_20260320" / "summary.json")

    hv4 = v4["headline_metrics"]
    hcf = closure_final["headline_metrics"]
    hls = local_structure["headline_metrics"]
    hcb = circuit_v3["headline_metrics"]

    anchor_chart_term_v5 = hv4["anchor_chart_term_v4"] + hcf["final_closure_support"]
    local_primary_term_v5 = hv4["local_primary_term_v4"] + hls["local_primary_structure"]
    circuit_term_v5 = hcb["concept_circuit_balance_v3"] / (1.0 + hcb["concept_circuit_balance_v3"])
    pressure_term_v5 = hv4["pressure_term_v4"] + hcf["final_gap_penalty"]
    concept_margin_v5 = anchor_chart_term_v5 + local_primary_term_v5 + circuit_term_v5 - pressure_term_v5

    return {
        "headline_metrics": {
            "anchor_chart_term_v5": anchor_chart_term_v5,
            "local_primary_term_v5": local_primary_term_v5,
            "circuit_term_v5": circuit_term_v5,
            "pressure_term_v5": pressure_term_v5,
            "concept_margin_v5": concept_margin_v5,
        },
        "closed_form_equation": {
            "anchor_chart_term": "AC_v5 = anchor_chart_term_v4 + final_closure_support",
            "local_term": "L_v5 = local_primary_term_v4 + local_primary_structure",
            "circuit_term": "C_v5 = concept_circuit_balance_v3 / (1 + concept_circuit_balance_v3)",
            "pressure_term": "P_v5 = pressure_term_v4 + final_gap_penalty",
            "margin_term": "M_concept_v5 = AC_v5 + L_v5 + C_v5 - P_v5",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 概念形成闭式第五版报告",
        "",
        f"- anchor_chart_term_v5: {hm['anchor_chart_term_v5']:.6f}",
        f"- local_primary_term_v5: {hm['local_primary_term_v5']:.6f}",
        f"- circuit_term_v5: {hm['circuit_term_v5']:.6f}",
        f"- pressure_term_v5: {hm['pressure_term_v5']:.6f}",
        f"- concept_margin_v5: {hm['concept_margin_v5']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_concept_formation_closed_form_v5_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
