from __future__ import annotations

from stage56_icspb_closed_equation_v2 import build_summary


def test_build_summary_promotes_g_and_s_to_final_objects() -> None:
    summary = build_summary(
        {"final_score": 1.0},
        {"closure_confidence": 0.5},
        {"native_proxy_summary": {"x": 1}},
        {"short_form": {"general_short_form": "ok"}},
    )
    assert summary["state_dictionary"]["G_final"] == "kernel_v4"
    assert summary["state_dictionary"]["S_final"] == "strict_module_base_term"
