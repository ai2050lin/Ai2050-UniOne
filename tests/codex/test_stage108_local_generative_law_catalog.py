from __future__ import annotations

import json
from pathlib import Path

from stage108_local_generative_law_catalog import build_local_generative_law_catalog_summary


ROOT = Path(__file__).resolve().parents[2]


def test_stage108_local_generative_law_catalog() -> None:
    summary = build_local_generative_law_catalog_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["law_catalog_coverage"] >= 0.60
    assert hm["law_composability_score"] >= 0.60
    assert hm["law_failure_resilience"] >= 0.25
    assert hm["strongest_law_name"] in {
        "projection_transport_law",
        "distributed_route_settlement_law",
        "bounded_repair_contraction_law",
        "anchor_refinement_law",
        "boundary_exposure_law",
    }
    assert hm["weakest_law_name"] in {
        "projection_transport_law",
        "distributed_route_settlement_law",
        "bounded_repair_contraction_law",
        "anchor_refinement_law",
        "boundary_exposure_law",
    }
    assert hm["weakest_law_score"] >= 0.45
    assert hm["highest_failure_boundary_name"] in {
        "projection_boundary",
        "routing_boundary",
        "repair_boundary",
        "evidence_boundary",
        "anchor_boundary",
    }
    assert hm["highest_failure_boundary_pressure"] >= 0.25
    assert hm["local_generative_law_catalog_score"] >= 0.56
    assert len(summary["law_records"]) == 5
    assert status["status_short"] in {
        "local_generative_law_catalog_ready",
        "local_generative_law_catalog_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage108_local_generative_law_catalog_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
