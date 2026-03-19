from __future__ import annotations

from stage56_window_term_strengthening import build_rows, build_summary


def test_build_rows_aggregates_window_features() -> None:
    joined_rows = [
        {
            "model_id": "m",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "pear",
            "complete_generated_energy": 2.0,
            "complete_prompt_energy": 1.0,
            "complete_energy_gap": 1.0,
            "generated_dominance_score": 0.7,
            "hidden_window_center": 10.0,
            "mlp_window_center": 12.0,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": True,
        },
        {
            "model_id": "m",
            "category": "fruit",
            "prototype_term": "apple",
            "instance_term": "pear",
            "complete_generated_energy": 4.0,
            "complete_prompt_energy": 2.0,
            "complete_energy_gap": 2.0,
            "generated_dominance_score": 0.8,
            "hidden_window_center": 11.0,
            "mlp_window_center": 13.0,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": True,
        },
    ]
    rows = build_rows(joined_rows)
    assert len(rows) == 1
    assert rows[0]["generated_window_mass"] == 3.0
    assert rows[0]["window_center_gap"] == 2.0


def test_build_summary_returns_three_fits() -> None:
    rows = [
        {
            "generated_window_mass": 2.0,
            "prompt_window_mass": 1.0,
            "generated_window_gap": 1.0,
            "generated_dominance_mean": 0.7,
            "window_center_mean": 11.0,
            "window_center_gap": 2.0,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": 1.0,
        },
        {
            "generated_window_mass": 1.0,
            "prompt_window_mass": 1.5,
            "generated_window_gap": -0.5,
            "generated_dominance_mean": 0.4,
            "window_center_mean": 9.0,
            "window_center_gap": 1.0,
            "union_joint_adv": 0.1,
            "union_synergy_joint": 0.0,
            "strict_positive_synergy": 0.0,
        },
    ]
    summary = build_summary(rows)
    assert summary["record_type"] == "stage56_window_term_strengthening_summary"
    assert len(summary["fits"]) == 3
