from __future__ import annotations

from stage56_frontier_migration_master_refit import build_summary, join_rows


def test_join_rows_builds_new_master_terms() -> None:
    rows = join_rows(
        [
            {
                "model_id": "m",
                "category": "c",
                "prototype_term": "p",
                "instance_term": "i",
                "identity_margin_term": 1.0,
                "syntax_constraint_conflict_term": 2.0,
                "logic_fragile_bridge_term": 3.0,
                "style_alignment_term": 4.0,
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
                "frontier_compaction_late_shift": 5.0,
                "frontier_balance_term": 6.0,
                "frontier_compaction_term": 1.0,
                "frontier_coverage_term": 2.0,
                "frontier_separation_term": 3.0,
            }
        ],
        [
            {
                "model_id": "m",
                "category": "c",
                "prototype_term": "p",
                "instance_term": "i",
                "window_positive_core_term": 7.0,
                "window_syntax_term": 8.0,
                "window_negative_core_term": 9.0,
                "window_fragile_term": 10.0,
            }
        ],
    )
    row = rows[0]
    assert row["frontier_positive_migration_term"] == 11.0
    assert row["frontier_negative_base_term"] == 6.0
    assert row["window_gate_positive_term"] == 15.0
    assert row["window_gate_negative_term"] == 19.0


def test_build_summary_marks_stable_signs() -> None:
    rows = []
    for scale in (1.0, 2.0, 3.0, 4.0):
        rows.append(
            {
                "identity_margin_term": scale,
                "syntax_constraint_conflict_term": scale,
                "logic_fragile_bridge_term": -scale,
                "style_alignment_term": -scale,
                "frontier_positive_migration_term": scale,
                "frontier_negative_base_term": -scale,
                "window_gate_positive_term": scale,
                "window_gate_negative_term": -scale,
                "union_joint_adv": scale,
                "union_synergy_joint": scale,
                "strict_positive_synergy": scale,
            }
        )
    summary = build_summary(rows)
    features = {row["feature"]: row["sign"] for row in summary["stable_features"]}
    assert features["frontier_negative_base_term"] == "negative"

