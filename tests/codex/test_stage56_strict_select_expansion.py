from __future__ import annotations

from stage56_strict_select_expansion import build_rows, build_summary


def test_build_rows_adds_strictness_deltas() -> None:
    rows = [
        {
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.4,
            "strict_positive_synergy": 1.0,
            "load_contrast_term": 1.0,
            "load_mean_term": -1.0,
        }
    ]
    out_rows = build_rows(rows)
    assert out_rows[0]["strictness_delta_vs_union"] == 0.8
    assert out_rows[0]["strictness_delta_vs_synergy"] == 0.6
    assert out_rows[0]["strictness_delta_vs_mean"] == 0.7


def test_build_summary_detects_stable_positive_strict_feature() -> None:
    rows = []
    for x in [1.0, 2.0, 3.0, 4.0]:
        rows.append(
            {
                "load_contrast_term": x,
                "load_mean_term": -x,
                "strict_positive_synergy": x,
                "strictness_delta_vs_union": x,
                "strictness_delta_vs_synergy": x,
                "strictness_delta_vs_mean": x,
            }
        )
    summary = build_summary(rows)
    assert "load_contrast_term" in list(summary["stable_strict_features"])
