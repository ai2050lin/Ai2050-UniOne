from __future__ import annotations

from stage56_math_framework_bridge import build_framework_bridge


def test_build_framework_bridge_prefers_layered_stack_over_single_math_family() -> None:
    law_summary = {
        "laws": {
            "broad_support_base": 0.19,
            "long_separation_frontier": 0.23,
            "mid_syntax_filter": 0.30,
            "late_window_closure": -0.02,
        }
    }
    frontier_summary = {
        "closure": {
            "pair_positive_ratio": 0.19,
        }
    }
    out = build_framework_bridge(law_summary, frontier_summary)
    assert out["record_type"] == "stage56_math_framework_bridge_summary"
    assert "图册/纤维束负责静态概念层" in out["recommended_stack"]
    assert "分层动力系统负责生成与闭包层" in out["recommended_stack"]
    assert "不是简单升级成某一个现成学科" in out["answer_to_user_question"]
