from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_generation_gate_coupling import (  # noqa: E402
    build_summary,
    compute_axis_gate_summary,
    compute_gate_proxies,
    filter_taxonomy_rows,
)


def test_compute_gate_proxies_recovers_bridge_conflict_and_mismatch() -> None:
    proxies = compute_gate_proxies(
        prototype_drop=0.09,
        instance_drop=0.04,
        strong_drop=0.03,
        mixed_drop=0.08,
    )
    assert proxies["prototype_field_proxy"] == 0.09
    assert proxies["instance_field_proxy"] == 0.04
    assert abs(proxies["bridge_field_proxy"] - 0.05) < 1e-9
    assert proxies["conflict_field_proxy"] == 0.0
    assert abs(proxies["mismatch_field_proxy"] - 0.01) < 1e-9


def test_compute_axis_gate_summary_compares_against_control() -> None:
    variant_rows = [
        {
            "axis": "control",
            "strong_drop": 0.02,
            "mixed_drop": 0.03,
            "bridge_gain": 0.01,
            "gate_proxies": compute_gate_proxies(0.04, 0.03, 0.02, 0.03),
        },
        {
            "axis": "style",
            "strong_drop": 0.01,
            "mixed_drop": 0.05,
            "bridge_gain": 0.04,
            "gate_proxies": compute_gate_proxies(0.05, 0.02, 0.01, 0.05),
        },
        {
            "axis": "style",
            "strong_drop": 0.02,
            "mixed_drop": 0.04,
            "bridge_gain": 0.02,
            "gate_proxies": compute_gate_proxies(0.03, 0.01, 0.02, 0.04),
        },
        {
            "axis": "logic",
            "strong_drop": 0.04,
            "mixed_drop": 0.01,
            "bridge_gain": -0.03,
            "gate_proxies": compute_gate_proxies(0.01, 0.02, 0.04, 0.01),
        },
        {
            "axis": "syntax",
            "strong_drop": 0.02,
            "mixed_drop": 0.03,
            "bridge_gain": 0.01,
            "gate_proxies": compute_gate_proxies(0.04, 0.03, 0.02, 0.03),
        },
    ]
    summary = compute_axis_gate_summary(variant_rows)
    assert summary["control_bridge_gain"] == 0.01
    assert summary["axes"]["style"]["deltas"]["bridge_field_proxy"] == 0.02
    assert summary["axes"]["style"]["directions"]["bridge_field_proxy"] == "positive"
    assert summary["axes"]["logic"]["deltas"]["conflict_field_proxy"] == 0.03
    assert summary["axes"]["logic"]["directions"]["conflict_field_proxy"] == "positive"
    assert summary["axes"]["syntax"]["deltas"]["prototype_field_proxy"] == 0.0


def test_filter_taxonomy_rows_applies_all_conditions() -> None:
    rows = [
        {"group_label": "g1", "category": "fruit", "case_role": "weak_bridge_positive", "model_id": "m1"},
        {"group_label": "g1", "category": "fruit", "case_role": "weak_bridge_positive", "model_id": "m1"},
        {"group_label": "g2", "category": "animal", "case_role": "weak_drag_or_conflict", "model_id": "m2"},
    ]
    out = filter_taxonomy_rows(
        rows,
        group_labels=["g1"],
        categories=["fruit"],
        case_roles=["weak_bridge_positive"],
        max_cases_per_model=1,
    )
    assert len(out) == 1
    assert out[0]["model_id"] == "m1"


def test_build_summary_uses_case_rows_for_per_model_stats() -> None:
    case_rows = [
        {
            "model_id": "m1",
            "axis_gate_summary": {
                "axes": {
                    "style": {"deltas": {name: 0.1 for name in (
                        "prototype_field_proxy",
                        "instance_field_proxy",
                        "bridge_field_proxy",
                        "conflict_field_proxy",
                        "mismatch_field_proxy",
                    )}, "directions": {name: "positive" for name in (
                        "prototype_field_proxy",
                        "instance_field_proxy",
                        "bridge_field_proxy",
                        "conflict_field_proxy",
                        "mismatch_field_proxy",
                    )}},
                    "logic": {"deltas": {name: 0.0 for name in (
                        "prototype_field_proxy",
                        "instance_field_proxy",
                        "bridge_field_proxy",
                        "conflict_field_proxy",
                        "mismatch_field_proxy",
                    )}, "directions": {name: "neutral" for name in (
                        "prototype_field_proxy",
                        "instance_field_proxy",
                        "bridge_field_proxy",
                        "conflict_field_proxy",
                        "mismatch_field_proxy",
                    )}},
                    "syntax": {"deltas": {name: -0.1 for name in (
                        "prototype_field_proxy",
                        "instance_field_proxy",
                        "bridge_field_proxy",
                        "conflict_field_proxy",
                        "mismatch_field_proxy",
                    )}, "directions": {name: "negative" for name in (
                        "prototype_field_proxy",
                        "instance_field_proxy",
                        "bridge_field_proxy",
                        "conflict_field_proxy",
                        "mismatch_field_proxy",
                    )}},
                }
            },
        },
        {
            "model_id": "m2",
            "axis_gate_summary": {
                "axes": {
                    "style": {"deltas": {name: 0.2 for name in (
                        "prototype_field_proxy",
                        "instance_field_proxy",
                        "bridge_field_proxy",
                        "conflict_field_proxy",
                        "mismatch_field_proxy",
                    )}, "directions": {name: "positive" for name in (
                        "prototype_field_proxy",
                        "instance_field_proxy",
                        "bridge_field_proxy",
                        "conflict_field_proxy",
                        "mismatch_field_proxy",
                    )}},
                    "logic": {"deltas": {name: 0.0 for name in (
                        "prototype_field_proxy",
                        "instance_field_proxy",
                        "bridge_field_proxy",
                        "conflict_field_proxy",
                        "mismatch_field_proxy",
                    )}, "directions": {name: "neutral" for name in (
                        "prototype_field_proxy",
                        "instance_field_proxy",
                        "bridge_field_proxy",
                        "conflict_field_proxy",
                        "mismatch_field_proxy",
                    )}},
                    "syntax": {"deltas": {name: 0.0 for name in (
                        "prototype_field_proxy",
                        "instance_field_proxy",
                        "bridge_field_proxy",
                        "conflict_field_proxy",
                        "mismatch_field_proxy",
                    )}, "directions": {name: "neutral" for name in (
                        "prototype_field_proxy",
                        "instance_field_proxy",
                        "bridge_field_proxy",
                        "conflict_field_proxy",
                        "mismatch_field_proxy",
                    )}},
                }
            },
        },
    ]
    summary = build_summary(case_rows, runtime_sec=1.5)
    assert summary["case_count"] == 2
    assert summary["per_model"]["m1"]["case_count"] == 1
    assert summary["per_model"]["m1"]["per_axis"]["style"]["mean_deltas"]["bridge_field_proxy"] == 0.1
    assert summary["per_model"]["m2"]["per_axis"]["style"]["mean_deltas"]["bridge_field_proxy"] == 0.2
