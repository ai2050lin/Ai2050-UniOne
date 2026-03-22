from __future__ import annotations

import json
from pathlib import Path

from stage111_native_variable_registry_pruning import build_native_variable_registry_pruning_summary


ROOT = Path(__file__).resolve().parents[2]


def test_stage111_native_variable_registry_pruning() -> None:
    summary = build_native_variable_registry_pruning_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["native_core_variable_count"] >= 2
    assert hm["projection_variable_count"] >= 5
    assert hm["proxy_variable_count"] >= 2
    assert hm["deferred_variable_count"] >= 1
    assert hm["strongest_native_name"] in {
        "conditional_projection_field",
        "distributed_route_fiber",
        "repair_closure_loop",
        "anchor_recurrence_family",
        "hierarchical_concept_span_quantity",
        "context_covariant_uniqueness_quantity",
        "minimal_transport_efficiency_quantity",
    }
    assert hm["weakest_native_name"] in {
        "conditional_projection_field",
        "distributed_route_fiber",
        "repair_closure_loop",
        "anchor_recurrence_family",
        "hierarchical_concept_span_quantity",
        "context_covariant_uniqueness_quantity",
        "minimal_transport_efficiency_quantity",
    }
    assert hm["weakest_native_score"] >= 0.60
    assert hm["native_variable_purity"] >= 0.55
    assert hm["proxy_load_penalty"] <= 0.30
    assert hm["native_variable_registry_pruning_score"] >= 0.52
    assert len(summary["registry_records"]) == 14
    assert status["status_short"] in {
        "native_variable_registry_pruning_ready",
        "native_variable_registry_pruning_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage111_native_variable_registry_pruning_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
