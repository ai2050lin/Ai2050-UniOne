from __future__ import annotations

from stage56_real_corpus_native_proxy_refinement import build_rows


def test_build_rows_creates_native_proxies() -> None:
    rows = build_rows(
        [
            {
                "strict_positive_synergy": True,
                "union_joint_adv": 0.1,
                "union_synergy_joint": 0.2,
                "axes": {
                    "style": {
                        "prototype_delta_l2_topness": 0.1,
                        "instance_delta_l2_topness": 0.1,
                        "prototype_delta_l2": 1.0,
                        "instance_delta_l2": 1.0,
                        "pair_mean_delta_l2_topness": 0.1,
                    },
                    "logic": {
                        "prototype_delta_l2_topness": 0.3,
                        "instance_delta_l2_topness": 0.3,
                        "prototype_delta_l2": 2.0,
                        "instance_delta_l2": 2.0,
                        "pair_mean_delta_l2_topness": 0.2,
                    },
                    "syntax": {
                        "prototype_delta_mean_abs_topness": 0.4,
                        "instance_delta_mean_abs_topness": 0.4,
                        "prototype_delta_l2": 3.0,
                        "instance_delta_l2": 3.0,
                        "pair_mean_delta_l2_topness": 0.5,
                    },
                },
            }
        ]
    )
    assert "G_native_proxy" in rows[0]
    assert "L_base_native_proxy" in rows[0]
    assert "L_select_native_proxy" in rows[0]
