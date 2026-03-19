from __future__ import annotations

from stage56_window_dominance_deep_split import build_rows, build_summary


def test_build_rows_creates_window_core_terms() -> None:
    rows = build_rows(
        [
            {
                "identity_margin_term": 2.0,
                "syntax_constraint_conflict_term": 4.0,
                "logic_fragile_bridge_term": 6.0,
                "style_alignment_term": 8.0,
                "frontier_term": 3.0,
                "window_dominance_term": 0.5,
            }
        ]
    )
    row = rows[0]
    assert row["window_identity_term"] == 1.0
    assert row["window_positive_core_term"] == 1.5
    assert row["window_negative_core_term"] == 3.5


def test_build_summary_marks_stable_signs() -> None:
    rows = []
    for scale in (1.0, 2.0, 3.0, 4.0):
        rows.append(
            {
                "window_identity_term": scale,
                "window_syntax_term": scale,
                "window_fragile_term": -scale,
                "window_style_term": -scale,
                "window_frontier_term": scale,
                "window_positive_core_term": scale,
                "window_negative_core_term": -scale,
                "union_joint_adv": scale,
                "union_synergy_joint": scale,
                "strict_positive_synergy": scale,
            }
        )
    summary = build_summary(rows)
    features = {row["feature"]: row["sign"] for row in summary["stable_features"]}
    assert features["window_positive_core_term"] == "positive"

