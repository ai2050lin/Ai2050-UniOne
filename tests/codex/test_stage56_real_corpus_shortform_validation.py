from __future__ import annotations

from stage56_real_corpus_shortform_validation import build_rows, build_summary


def test_build_rows_adds_corpus_proxies() -> None:
    rows = [
        {
            "strict_positive_synergy": True,
            "union_joint_adv": 0.2,
            "union_synergy_joint": 0.4,
            "axes": {
                "style": {"pair_mean_delta_l2_topness": 0.1, "pair_mean_delta_l2": 10.0},
                "logic": {"pair_mean_delta_l2_topness": 0.6, "pair_mean_delta_l2": 8.0},
                "syntax": {"pair_mean_delta_l2_topness": 0.7, "pair_mean_delta_l2": 6.0},
            },
        }
    ]
    out_rows = build_rows(rows)
    assert out_rows[0]["G_corpus_proxy"] == 0.55
    assert out_rows[0]["L_base_corpus_proxy"] == 8.0
    assert out_rows[0]["L_select_corpus_proxy"] == 3.0


def test_build_summary_collects_signs() -> None:
    rows = []
    for x in [1.0, 2.0, 3.0, 4.0]:
        rows.append(
            {
                "G_corpus_proxy": x,
                "L_base_corpus_proxy": -x,
                "L_select_corpus_proxy": x,
                "union_joint_adv": x,
                "union_synergy_joint": x,
                "strict_bool": x,
                "strictness_delta_vs_union": x,
                "strictness_delta_vs_synergy": x,
                "strictness_delta_vs_mean": x,
            }
        )
    summary = build_summary(rows)
    assert dict(summary["sign_matrix"])["G_corpus_proxy"]["union_joint_adv"] == "positive"
