from __future__ import annotations

import json
from pathlib import Path

from stage109_invariant_boundary_quantity_search import build_invariant_boundary_quantity_search_summary


ROOT = Path(__file__).resolve().parents[2]


def test_stage109_invariant_boundary_quantity_search() -> None:
    summary = build_invariant_boundary_quantity_search_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["invariant_quantity_strength"] >= 0.55
    assert hm["boundary_quantity_resilience"] >= 0.20
    assert hm["theory_breakthrough_readiness"] >= 0.45
    assert hm["strongest_quantity_name"] in {
        "hierarchical_concept_span_quantity",
        "context_covariant_uniqueness_quantity",
        "minimal_transport_efficiency_quantity",
        "relational_linearity_quantity",
        "repair_stability_quantity",
    }
    assert hm["weakest_quantity_name"] in {
        "hierarchical_concept_span_quantity",
        "context_covariant_uniqueness_quantity",
        "minimal_transport_efficiency_quantity",
        "relational_linearity_quantity",
        "repair_stability_quantity",
    }
    assert hm["weakest_quantity_score"] >= 0.45
    assert hm["highest_boundary_name"] in {
        "macro_data_gap_boundary",
        "evidence_boundary",
        "anchor_ambiguity_boundary",
        "task_bridge_boundary",
        "linearity_proof_boundary",
    }
    assert hm["highest_boundary_pressure"] >= 0.25
    assert hm["invariant_boundary_quantity_score"] >= 0.52
    assert len(summary["quantity_records"]) == 5
    assert len(summary["boundary_records"]) == 5
    assert status["status_short"] in {
        "invariant_boundary_quantity_search_ready",
        "invariant_boundary_quantity_search_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage109_invariant_boundary_quantity_search_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
