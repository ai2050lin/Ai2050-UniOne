from __future__ import annotations

from stage56_load_operator_closure import build_rows, build_summary


def test_build_rows_adds_operator_terms() -> None:
    rows = [
        {"gs_load_channel_term": 2.0, "sd_load_channel_term": -6.0},
    ]
    out_rows = build_rows(rows)
    assert out_rows[0]["load_mean_term"] == -2.0
    assert out_rows[0]["load_contrast_term"] == -4.0
    assert out_rows[0]["load_abs_sum_term"] == 8.0


def test_build_summary_detects_base_and_selective_patterns() -> None:
    rows = []
    for x in [1.0, 2.0, 3.0, 4.0]:
        rows.append(
            {
                "load_mean_term": x,
                "load_contrast_term": x,
                "load_abs_sum_term": x,
                "union_joint_adv": -x,
                "union_synergy_joint": -2 * x,
                "strict_positive_synergy": x,
            }
        )
    summary = build_summary(rows)
    assert "load_contrast_term" in list(summary["strict_selective_features"])
