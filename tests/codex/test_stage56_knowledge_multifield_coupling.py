from __future__ import annotations

from stage56_knowledge_multifield_coupling import classify_regime, compute_case_fields, dominant_field


def test_compute_case_fields_bridge_and_mismatch() -> None:
    row = {
        "stage6_reference": {
            "proto_joint_adv": 0.2,
            "instance_joint_adv": 0.1,
            "union_joint_adv": -0.05,
            "union_synergy_joint": 0.03,
        },
        "best_strong": {"metrics": {"joint_adv_mean": 0.03}},
        "best_weak": {"metrics": {"joint_adv_mean": 0.01}},
        "best_mixed": {"metrics": {"joint_adv_mean": 0.08}},
    }
    fields = compute_case_fields(row)
    assert fields["bridge_field"] == 0.03
    assert fields["conflict_field"] == 0.0
    assert fields["route_mismatch_field"] == 0.25


def test_classify_regime_bridge_compensated() -> None:
    fields = {
        "prototype_field": 0.2,
        "instance_field": -0.1,
        "bridge_field": 0.05,
        "conflict_field": 0.0,
        "route_mismatch_field": 0.1,
    }
    assert classify_regime(fields) == "bridge_compensated_mismatch"


def test_dominant_field_prefers_largest_magnitude() -> None:
    fields = {
        "prototype_field": 0.1,
        "instance_field": -0.3,
        "bridge_field": 0.05,
        "conflict_field": 0.02,
        "route_mismatch_field": 0.15,
    }
    assert dominant_field(fields) == "instance_field"


def test_compute_case_fields_uses_negative_synergy_as_conflict() -> None:
    row = {
        "stage6_reference": {
            "proto_joint_adv": 0.04,
            "instance_joint_adv": 0.02,
            "union_joint_adv": 0.01,
            "union_synergy_joint": -0.05,
        },
        "best_strong": {"metrics": {"joint_adv_mean": 0.02}},
        "best_weak": {"metrics": {"joint_adv_mean": 0.0}},
        "best_mixed": {"metrics": {"joint_adv_mean": 0.01}},
    }
    fields = compute_case_fields(row)
    assert fields["bridge_field"] == 0.0
    assert fields["conflict_field"] == 0.05
