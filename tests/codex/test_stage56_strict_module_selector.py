from __future__ import annotations

from stage56_strict_module_selector import build_summary


def test_strict_module_selector_ranks_by_selectivity() -> None:
    summary = build_summary(
        {
            "feature_names": ["a", "b"],
            "fits": [
                {"target_name": "union_joint_adv", "weights": {"a": -0.2, "b": -0.05}},
                {"target_name": "union_synergy_joint", "weights": {"a": -0.1, "b": -0.05}},
                {"target_name": "strict_positive_synergy", "weights": {"a": 0.6, "b": 0.1}},
            ],
        }
    )
    assert summary["ranking"][0]["feature"] == "a"
    assert summary["top_candidates"][0]["feature"] == "a"
