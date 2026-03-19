from __future__ import annotations

from stage56_negative_mass_deep_split import build_rows, build_summary


def test_build_rows_splits_negative_mass() -> None:
    rows = build_rows(
        [
            {
                "logic_fragile_bridge_term": 1.0,
                "frontier_negative_base_term": 2.0,
                "window_gate_negative_term": 3.0,
                "style_alignment_term": 4.0,
            }
        ]
    )
    row = rows[0]
    assert row["destructive_negative_term"] == 6.0
    assert row["alignment_load_term"] == 4.0
    assert row["negative_mass_rebalanced_term"] == 2.0


def test_build_summary_marks_destructive_term() -> None:
    rows = []
    for scale in (1.0, 2.0, 3.0, 4.0):
        rows.append(
            {
                "destructive_negative_term": -scale,
                "alignment_load_term": scale,
                "negative_mass_rebalanced_term": -2 * scale,
                "union_joint_adv": scale,
                "union_synergy_joint": scale,
                "strict_positive_synergy": scale,
            }
        )
    summary = build_summary(rows)
    features = {row["feature"]: row["sign"] for row in summary["stable_features"]}
    assert features["alignment_load_term"] == "positive"

