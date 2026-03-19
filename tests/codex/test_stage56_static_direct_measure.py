from __future__ import annotations

from stage56_static_direct_measure import build_rows, build_summary


def test_build_rows_creates_direct_static_features() -> None:
    design_rows = [
        {
            "model_id": "m",
            "category": "fruit",
            "atlas_static_proxy": 0.9,
            "frontier_dynamic_proxy": 0.5,
            "offset_static_proxy": 0.1,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": 1.0,
        },
        {
            "model_id": "m",
            "category": "fruit",
            "atlas_static_proxy": 0.7,
            "frontier_dynamic_proxy": 0.3,
            "offset_static_proxy": 0.2,
            "union_joint_adv": 0.1,
            "union_synergy_joint": 0.0,
            "strict_positive_synergy": 0.0,
        },
    ]
    rows = build_rows(design_rows)
    assert len(rows) == 2
    assert "family_patch_direct" in rows[0]
    assert "concept_offset_direct" in rows[0]
    assert "identity_margin_direct" in rows[0]


def test_build_summary_returns_three_fits() -> None:
    rows = [
        {
            "family_patch_direct": 0.7,
            "concept_offset_direct": 0.1,
            "identity_margin_direct": 0.6,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": 1.0,
        },
        {
            "family_patch_direct": 0.6,
            "concept_offset_direct": 0.2,
            "identity_margin_direct": 0.4,
            "union_joint_adv": 0.1,
            "union_synergy_joint": 0.0,
            "strict_positive_synergy": 0.0,
        },
    ]
    summary = build_summary(rows)
    assert summary["record_type"] == "stage56_static_direct_measure_summary"
    assert len(summary["fits"]) == 3
