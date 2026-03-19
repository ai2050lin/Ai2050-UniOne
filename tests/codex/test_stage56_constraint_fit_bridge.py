from __future__ import annotations

from stage56_constraint_fit_bridge import build_bridge


def test_build_bridge_maps_constraints_to_feature_families() -> None:
    constraint_summary = {
        "constraints": [
            {"axiom": "局部图册公理"},
            {"axiom": "高质量前沿公理"},
            {"axiom": "功能子场公理"},
            {"axiom": "时间窗收束公理"},
            {"axiom": "闭包边界公理"},
            {"axiom": "分层统一公理"},
        ]
    }
    regression_outline_summary = {
        "feature_families": [
            {"family": "静态本体项", "features": ["atlas_static_hat", "offset_static_hat"]},
            {"family": "动态前沿项", "features": ["frontier_positive_corr"]},
            {"family": "内部子场项", "features": ["logic_prototype_score"]},
            {"family": "窗口闭包项", "features": ["hidden_window_center"]},
            {"family": "控制轴项", "features": ["style_control"]},
        ]
    }

    out = build_bridge(constraint_summary, regression_outline_summary)
    assert out["record_type"] == "stage56_constraint_fit_bridge_summary"
    assert len(out["bridge_rows"]) == 6
    assert out["coverage_ratio"] > 0.0
