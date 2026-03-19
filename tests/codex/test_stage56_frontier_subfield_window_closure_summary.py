from __future__ import annotations

from stage56_frontier_subfield_window_closure_summary import build_summary


def test_build_summary_collects_four_concepts() -> None:
    pair_density_summary = {
        "per_axis": {
            "syntax": {
                "feature_stats": {
                    "pair_compaction_middle_mean": {
                        "mean_value": 0.3,
                        "positive_pair_gap": 0.05,
                        "targets": {"strict_positive_synergy": {"pearson_corr": 0.31}},
                    }
                }
            },
            "logic": {
                "feature_stats": {
                    "pair_delta_l2": {
                        "mean_value": 10.0,
                        "positive_pair_gap": -1.0,
                        "targets": {"strict_positive_synergy": {"pearson_corr": -0.39}},
                    }
                }
            },
        }
    }
    complete_summary = {
        "per_component": {
            "logic_prototype": {
                "case_count": 5,
                "feature_stats": {
                    "hidden_layer_center": {"targets": {"union_synergy_joint": {"pearson_corr": 0.2}}},
                    "complete_prompt_energy": {"targets": {"union_synergy_joint": {"pearson_corr": -0.1}}},
                },
            }
        }
    }
    window_summary = {
        "per_component": {
            "logic_prototype": {
                "overall": {
                    "dominant_hidden_tail_position_mode": "tail_pos_-5",
                    "dominant_mlp_tail_position_mode": "tail_pos_-5",
                    "peak_hidden_tail_position_from_profile": "tail_pos_-5",
                    "peak_mlp_tail_position_from_profile": "tail_pos_-5",
                    "mean_union_synergy_joint": 0.05,
                }
            }
        }
    }
    pair_link_summary = {
        "pair_positive_ratio": 0.2,
        "mean_union_joint_adv": 0.1,
        "mean_union_synergy_joint": -0.02,
        "axis_target_stats": {
            "logic": {
                "prototype_field_proxy": {
                    "targets": {"union_synergy_joint": {"pearson_corr": 0.25}}
                },
                "bridge_field_proxy": {
                    "targets": {"union_synergy_joint": {"pearson_corr": -0.21}}
                },
            }
        },
    }
    law_summary = {
        "laws": {
            "broad_support_base": 0.19,
            "long_separation_frontier": 0.23,
        }
    }

    summary = build_summary(
        pair_density_summary=pair_density_summary,
        complete_summary=complete_summary,
        window_summary=window_summary,
        pair_link_summary=pair_link_summary,
        law_summary=law_summary,
    )

    assert summary["record_type"] == "stage56_frontier_subfield_window_closure_summary"
    assert summary["density_frontier"]["strongest_positive_frontier"]["axis"] == "syntax"
    assert summary["density_frontier"]["strongest_negative_frontier"]["axis"] == "logic"
    assert summary["internal_subfield"]["components"][0]["component_label"] == "logic_prototype"
    assert summary["token_window"]["components"][0]["dominant_hidden_tail_position"] == "tail_pos_-5"
    assert summary["closure"]["strongest_positive_field_to_synergy"]["field_name"] == "prototype_field_proxy"
