from __future__ import annotations

from stage56_strict_module_finalizer import build_summary


def test_strict_module_finalizer_prefers_base_when_scores_are_close() -> None:
    summary = build_summary(
        {
            "feature_names": ["strict_module_base_term", "strict_module_combined_term"],
            "fits": [
                {
                    "target_name": "union_joint_adv",
                    "weights": {"strict_module_base_term": -0.2, "strict_module_combined_term": -0.2},
                },
                {
                    "target_name": "union_synergy_joint",
                    "weights": {"strict_module_base_term": -0.1, "strict_module_combined_term": -0.1},
                },
                {
                    "target_name": "strict_positive_synergy",
                    "weights": {"strict_module_base_term": 0.6, "strict_module_combined_term": 0.605},
                },
            ],
        }
    )
    assert summary["final_choice"]["feature"] == "strict_module_base_term"
