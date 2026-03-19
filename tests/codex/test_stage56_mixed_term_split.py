from __future__ import annotations

from stage56_mixed_term_split import build_rows, build_summary


def test_build_rows_creates_mixed_interactions() -> None:
    rows = [
        {
            "logic_prototype_term": 0.5,
            "identity_margin_term": 0.7,
            "frontier_term": 0.4,
            "syntax_constraint_conflict_term": 0.2,
            "window_dominance_term": 0.8,
            "style_alignment_term": 0.6,
            "style_midfield_term": 0.3,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": 1.0,
        }
    ]
    out = build_rows(rows)
    assert len(out) == 1
    assert out[0]["logic_prototype_margin_term"] == 0.35
    assert out[0]["window_dominance_style_alignment_term"] == 0.48


def test_build_summary_returns_three_fits() -> None:
    rows = [
        {
            "logic_prototype_margin_term": 0.35,
            "logic_prototype_frontier_term": 0.2,
            "logic_prototype_syntax_term": 0.1,
            "window_dominance_style_alignment_term": 0.48,
            "window_dominance_style_midfield_term": 0.24,
            "window_dominance_frontier_term": 0.32,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": 1.0,
        },
        {
            "logic_prototype_margin_term": 0.10,
            "logic_prototype_frontier_term": 0.05,
            "logic_prototype_syntax_term": 0.02,
            "window_dominance_style_alignment_term": 0.20,
            "window_dominance_style_midfield_term": 0.10,
            "window_dominance_frontier_term": 0.15,
            "union_joint_adv": 0.1,
            "union_synergy_joint": 0.0,
            "strict_positive_synergy": 0.0,
        },
    ]
    summary = build_summary(rows)
    assert summary["record_type"] == "stage56_mixed_term_split_summary"
    assert len(summary["fits"]) == 3
