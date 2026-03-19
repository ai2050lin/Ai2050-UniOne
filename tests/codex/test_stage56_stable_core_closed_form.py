from __future__ import annotations

from stage56_stable_core_closed_form import build_rows, build_summary


def test_build_rows_creates_closed_form_terms() -> None:
    rows = build_rows(
        [
            {
                "identity_margin_term": 1.0,
                "syntax_constraint_conflict_term": 2.0,
                "frontier_positive_migration_term": 3.0,
                "window_gate_positive_term": 4.0,
                "logic_fragile_bridge_term": 5.0,
                "style_alignment_term": 6.0,
                "frontier_negative_base_term": 7.0,
                "window_gate_negative_term": 8.0,
            }
        ]
    )
    row = rows[0]
    assert row["positive_mass_term"] == 10.0
    assert row["negative_mass_term"] == 26.0
    assert row["closed_form_balance_term"] == -16.0


def test_build_summary_detects_positive_balance() -> None:
    rows = []
    for scale in (1.0, 2.0, 3.0, 4.0):
        rows.append(
            {
                "positive_mass_term": scale,
                "negative_mass_term": -scale,
                "closed_form_balance_term": 2 * scale,
                "union_joint_adv": scale,
                "union_synergy_joint": scale,
                "strict_positive_synergy": scale,
            }
        )
    summary = build_summary(rows)
    features = {row["feature"]: row["sign"] for row in summary["stable_features"]}
    assert features["closed_form_balance_term"] == "positive"
