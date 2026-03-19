from __future__ import annotations

from stage56_window_condition_gate_closure import build_summary, join_rows


def test_join_rows_builds_style_and_frontier_gate_terms() -> None:
    rows = join_rows(
        [
            {
                "model_id": "m",
                "category": "c",
                "prototype_term": "p",
                "instance_term": "i",
                "window_dominance_term": 0.5,
                "union_joint_adv": 1.0,
                "union_synergy_joint": 2.0,
                "strict_positive_synergy": 1.0,
            }
        ],
        [
            {
                "model_id": "m",
                "category": "c",
                "prototype_term": "p",
                "instance_term": "i",
                "style_delta_mean_abs": 2.0,
                "style_role_align_compaction": 2.0,
                "style_alignment": 2.0,
                "style_reorder_pressure": 2.0,
                "style_gap": 2.0,
                "style_compaction_mid": 4.0,
                "style_coverage_mid": 4.0,
                "style_delta_l2": 4.0,
                "style_role_align_coverage": 4.0,
                "style_midfield": 4.0,
            }
        ],
        [
            {
                "model_id": "m",
                "category": "c",
                "prototype_term": "p",
                "instance_term": "i",
                "frontier_compaction_late_shift": 6.0,
                "frontier_balance_term": 8.0,
                "frontier_compaction_term": 1.0,
                "frontier_coverage_term": 3.0,
                "frontier_separation_term": 5.0,
            }
        ],
    )
    row = rows[0]
    assert row["window_style_positive_term"] == 1.0
    assert row["window_style_negative_term"] == 2.0
    assert row["window_frontier_positive_term"] == 3.5
    assert row["window_frontier_negative_term"] == 1.5


def test_build_summary_marks_stable_terms() -> None:
    rows = []
    for scale in (1.0, 2.0, 3.0, 4.0):
        rows.append(
            {
                "window_style_positive_term": scale,
                "window_style_negative_term": -scale,
                "window_frontier_positive_term": scale,
                "window_frontier_negative_term": -scale,
                "union_joint_adv": scale,
                "union_synergy_joint": scale,
                "strict_positive_synergy": scale,
            }
        )
    summary = build_summary(rows)
    features = {row["feature"]: row["sign"] for row in summary["stable_features"]}
    assert features["window_frontier_positive_term"] == "positive"

