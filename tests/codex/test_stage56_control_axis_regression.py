from __future__ import annotations

from stage56_control_axis_regression import build_control_regression


def test_build_control_regression_returns_three_fits() -> None:
    rows = [
        {
            "style_control_proxy": 0.1,
            "logic_control_proxy": 0.2,
            "syntax_control_proxy": 0.3,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.1,
            "strict_positive_synergy": 1.0,
        },
        {
            "style_control_proxy": 0.2,
            "logic_control_proxy": 0.1,
            "syntax_control_proxy": 0.4,
            "union_joint_adv": 0.1,
            "union_synergy_joint": 0.05,
            "strict_positive_synergy": 0.0,
        },
    ]
    out = build_control_regression(rows)
    assert out["record_type"] == "stage56_control_axis_regression_summary"
    assert len(out["fits"]) == 3
    assert "logic_control_proxy" in out["sign_consistency"]
