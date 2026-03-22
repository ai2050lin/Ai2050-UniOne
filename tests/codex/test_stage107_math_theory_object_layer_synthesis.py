from __future__ import annotations

import json
from pathlib import Path

from stage107_math_theory_object_layer_synthesis import build_math_theory_object_layer_synthesis_summary


ROOT = Path(__file__).resolve().parents[2]


def test_stage107_math_theory_object_layer_synthesis() -> None:
    summary = build_math_theory_object_layer_synthesis_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["object_layer_viability_score"] >= 0.65
    assert hm["axiom_layer_viability_score"] >= 0.55
    assert hm["boundary_layer_viability_score"] >= 0.35
    assert hm["strongest_object_name"] in {
        "conditional_projection_field",
        "distributed_route_fiber",
        "repair_closure_loop",
        "anchor_recurrence_family",
        "falsification_boundary_shell",
    }
    assert hm["weakest_axiom_name"] in {
        "projection_covariance_axiom",
        "distributed_routing_axiom",
        "bounded_repair_axiom",
        "anchor_separability_axiom",
        "falsifiable_boundary_axiom",
    }
    assert hm["weakest_axiom_score"] >= 0.45
    assert hm["highest_boundary_name"] in {
        "projection_boundary",
        "routing_boundary",
        "repair_boundary",
        "evidence_boundary",
        "anchor_boundary",
    }
    assert hm["highest_boundary_pressure"] >= 0.25
    assert hm["theorem_core_transition_gap"] <= 0.80
    assert hm["math_theory_object_layer_score"] >= 0.58
    assert len(summary["object_records"]) == 5
    assert len(summary["axiom_records"]) == 5
    assert len(summary["boundary_records"]) == 5
    assert status["status_short"] in {
        "math_theory_object_layer_ready",
        "math_theory_object_layer_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage107_math_theory_object_layer_synthesis_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
