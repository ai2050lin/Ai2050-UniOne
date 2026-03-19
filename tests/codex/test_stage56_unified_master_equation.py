from __future__ import annotations

from stage56_unified_master_equation import build_unified_master_equation


def test_build_unified_master_equation_supports_general_math_system() -> None:
    law_summary = {
        "laws": {
            "long_separation_frontier": 0.23,
            "mid_syntax_filter": 0.30,
        }
    }
    frontier_summary = {
        "density_frontier": {},
        "closure": {
            "pair_positive_ratio": 0.19,
        },
    }
    framework_summary = {
        "recommended_stack": [
            "线性代数负责局部切片",
            "图册/纤维束负责静态概念层",
        ]
    }
    outline_summary = {
        "proto_axioms": [
            "局部身份由家族片区与概念偏移给出",
            "高质量表示由密度前沿决定",
        ]
    }

    out = build_unified_master_equation(
        law_summary=law_summary,
        frontier_summary=frontier_summary,
        framework_summary=framework_summary,
        outline_summary=outline_summary,
    )

    assert out["record_type"] == "stage56_unified_master_equation_summary"
    assert out["supports_general_math_system"] is True
    assert "Atlas_static" in out["equation_text"]
    assert "Window_closure" in out["equation_text"]
