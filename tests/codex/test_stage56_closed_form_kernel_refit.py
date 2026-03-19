from __future__ import annotations

from stage56_closed_form_kernel_refit import build_rows, build_summary


def test_build_rows_creates_v2_balances() -> None:
    rows = build_rows(
        [
            {
                "identity_margin_term": 1.0,
                "syntax_constraint_conflict_term": 2.0,
                "frontier_positive_migration_term": 3.0,
                "window_gate_positive_term": 4.0,
                "destructive_negative_term": 5.0,
                "alignment_load_term": 6.0,
            }
        ]
    )
    row = rows[0]
    assert row["positive_mass_v2_term"] == 10.0
    assert row["closed_form_balance_v2_term"] == 5.0
    assert row["strict_balance_v2_term"] == 11.0


def test_build_summary_marks_balance_positive() -> None:
    rows = []
    for scale in (1.0, 2.0, 3.0, 4.0):
        rows.append(
            {
                "positive_mass_v2_term": scale,
                "destructive_negative_v2_term": -scale,
                "alignment_load_v2_term": scale,
                "closed_form_balance_v2_term": 2 * scale,
                "strict_balance_v2_term": 3 * scale,
                "union_joint_adv": scale,
                "union_synergy_joint": scale,
                "strict_positive_synergy": scale,
            }
        )
    summary = build_summary(rows)
    features = {row["feature"]: row["sign"] for row in summary["stable_features"]}
    assert features["closed_form_balance_v2_term"] == "positive"

