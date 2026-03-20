from __future__ import annotations

from stage56_online_learning_stability_outline import build_summary


def test_build_summary_emits_safe_update_condition() -> None:
    summary = build_summary(
        {"learning_state": {"L_select_instability": 0.1}},
        {
            "support": {
                "strict_closure_confidence": 0.5,
                "native_proxy_summary": {"L_select_native_proxy": {"signs": {"a": "negative", "b": "positive"}}},
            }
        },
    )
    assert "safe_update_condition" in summary["stability_rules"]
