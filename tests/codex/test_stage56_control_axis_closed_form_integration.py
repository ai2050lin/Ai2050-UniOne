from __future__ import annotations

from stage56_control_axis_closed_form_integration import build_summary, join_rows


def test_join_rows_builds_structure_gain_terms() -> None:
    rows = join_rows(
        [
            {
                "model_id": "m",
                "category": "c",
                "prototype_term": "p",
                "instance_term": "i",
                "closed_form_balance_v2_term": 1.0,
                "alignment_load_v2_term": 2.0,
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
                "logic_compaction_mid": 5.0,
                "logic_delta_l2": 1.0,
                "syntax_coverage_mid": 6.0,
                "syntax_delta_l2": 2.0,
                "style_delta_mean_abs": 7.0,
                "style_coverage_mid": 3.0,
            }
        ],
    )
    row = rows[0]
    assert row["logic_structure_gain_term"] == 4.0
    assert row["syntax_structure_gain_term"] == 4.0
    assert row["style_structure_gain_term"] == 4.0


def test_build_summary_keeps_balance_positive() -> None:
    rows = []
    for scale in (1.0, 2.0, 3.0, 4.0):
        rows.append(
            {
                "closed_form_balance_v2_term": scale,
                "alignment_load_v2_term": scale,
                "logic_structure_gain_term": scale,
                "syntax_structure_gain_term": scale,
                "style_structure_gain_term": scale,
                "union_joint_adv": scale,
                "union_synergy_joint": scale,
                "strict_positive_synergy": scale,
            }
        )
    summary = build_summary(rows)
    features = {row["feature"]: row["sign"] for row in summary["stable_features"]}
    assert features["closed_form_balance_v2_term"] == "positive"
