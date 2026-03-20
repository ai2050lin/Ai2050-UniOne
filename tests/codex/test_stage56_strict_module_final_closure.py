from __future__ import annotations

from stage56_strict_module_final_closure import build_summary


def test_build_summary_uses_top_ranking_margin() -> None:
    summary = build_summary(
        {
            "ranking": [
                {"feature": "strict_module_base_term", "final_score": 0.5},
                {"feature": "strict_module_residual_term", "final_score": 0.4},
            ]
        }
    )
    assert summary["final_decision"] == "S_final = strict_module_base_term"
    assert abs(summary["score_margin"] - 0.1) < 1e-9
