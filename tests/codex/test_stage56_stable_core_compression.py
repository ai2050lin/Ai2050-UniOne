from __future__ import annotations

from stage56_stable_core_compression import build_rows, build_summary


def test_build_rows_creates_core_terms() -> None:
    rows = [
        {
            "identity_margin_term": 0.6,
            "syntax_constraint_conflict_term": 0.2,
            "logic_fragile_bridge_term": 0.1,
            "style_alignment_term": 0.3,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": 1.0,
        }
    ]
    out = build_rows(rows)
    assert len(out) == 1
    assert out[0]["positive_core_term"] == 0.4
    assert out[0]["negative_core_term"] == 0.2
    assert out[0]["stable_core_balance"] == 0.2


def test_build_summary_returns_three_fits() -> None:
    rows = [
        {
            "positive_core_term": 0.4,
            "negative_core_term": 0.2,
            "stable_core_balance": 0.2,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": 1.0,
        },
        {
            "positive_core_term": 0.2,
            "negative_core_term": 0.4,
            "stable_core_balance": -0.2,
            "union_joint_adv": 0.1,
            "union_synergy_joint": 0.0,
            "strict_positive_synergy": 0.0,
        },
    ]
    summary = build_summary(rows)
    assert summary["record_type"] == "stage56_stable_core_compression_summary"
    assert len(summary["fits"]) == 3
