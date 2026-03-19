from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_language_system_structure_atlas import (  # noqa: E402
    aggregate_category_rows,
    build_modifier_structure,
    build_payload,
)


def test_aggregate_category_rows_merges_multiple_models() -> None:
    rows = [
        {"category": "fruit", "pair_count": 2, "strict_positive_pair_count": 1, "mean_union_joint_adv": 0.1, "mean_union_synergy_joint": 0.02},
        {"category": "fruit", "pair_count": 4, "strict_positive_pair_count": 1, "mean_union_joint_adv": 0.3, "mean_union_synergy_joint": -0.01},
        {"category": "tech", "pair_count": 3, "strict_positive_pair_count": 0, "mean_union_joint_adv": -0.1, "mean_union_synergy_joint": -0.05},
    ]
    merged = aggregate_category_rows(rows)
    fruit = [row for row in merged if row["category"] == "fruit"][0]
    assert fruit["pair_count"] == 6
    assert fruit["strict_positive_pair_count"] == 2
    assert abs(fruit["mean_union_joint_adv"] - 0.2) < 1e-9


def test_build_modifier_structure_reads_adjective_roles_as_fibers() -> None:
    micro_json = {
        "concepts": {
            "apple": {
                "role_subsets": {
                    "entity": {"size": 4, "layer_distribution": {"0": 3, "7": 1}, "drop_target": 0.0},
                    "size": {"size": 9, "layer_distribution": {"0": 5, "1": 1, "2": 1, "7": 1, "23": 1}, "drop_target": 0.1},
                    "weight": {"size": 6, "layer_distribution": {"1": 3, "2": 1, "23": 2}, "drop_target": 0.2},
                    "fruit": {"size": 2, "layer_distribution": {"1": 1, "26": 1}, "drop_target": 0.3},
                }
            }
        }
    }
    payload = build_modifier_structure(micro_json)
    assert payload["adjective_roles"]["size"]["layer_count"] == 5
    assert payload["noun_anchor_roles"]["entity"]["layer_count"] == 2
    assert payload["adjective_layer_spread_mean"] > payload["noun_anchor_layer_spread_mean"]


def test_build_payload_marks_adverb_as_inference_only() -> None:
    payload = build_payload(
        micro_json={
            "concepts": {
                "apple": {
                    "role_subsets": {
                        "entity": {"size": 4, "layer_distribution": {"0": 3}, "drop_target": 0.0},
                        "size": {"size": 5, "layer_distribution": {"0": 1, "1": 1}, "drop_target": 0.1},
                        "weight": {"size": 5, "layer_distribution": {"1": 1, "2": 1}, "drop_target": 0.2},
                        "fruit": {"size": 2, "layer_distribution": {"1": 1}, "drop_target": 0.3},
                    }
                }
            }
        },
        apple_dossier={
            "metrics": {
                "apple_micro_to_meso_jaccard_mean": 0.02,
                "apple_meso_to_macro_jaccard_mean": 0.37,
                "apple_shared_base_ratio_mean": 0.03,
            }
        },
        concept_family={"metrics": {"apple_vs_cat_shared_base_gap_mean": 0.03}},
        triplet_json={
            "metrics": {
                "king_queen_jaccard": 0.09,
                "apple_king_jaccard": 0.0,
                "axis_specificity_index": 0.62,
                "triplet_separability_index": 0.09,
                "king_axis_projection_abs": 0.62,
                "queen_axis_projection_abs": 0.62,
                "apple_axis_projection_abs": 0.0,
            }
        },
        multidim_json={
            "metrics": {
                "style_logic_syntax_signal": 0.58,
                "cross_dim_decoupling_index": 0.68,
            },
            "dimensions": {
                "style": {"mean_pair_delta_l2": 1.0, "pair_delta_cosine_mean": 0.1},
                "logic": {"mean_pair_delta_l2": 2.0, "pair_delta_cosine_mean": 0.2},
                "syntax": {"mean_pair_delta_l2": 3.0, "pair_delta_cosine_mean": 0.3},
            },
        },
        discovery_summary={"strict_positive_pair_ratio": 0.2, "margin_zero_pair_ratio": 0.01},
        discovery_per_category_rows=[
            {"category": "abstract", "pair_count": 3, "strict_positive_pair_count": 1, "mean_union_joint_adv": 0.1, "mean_union_synergy_joint": 0.0},
            {"category": "action", "pair_count": 4, "strict_positive_pair_count": 1, "mean_union_joint_adv": 0.2, "mean_union_synergy_joint": 0.01},
            {"category": "tech", "pair_count": 4, "strict_positive_pair_count": 0, "mean_union_joint_adv": -0.1, "mean_union_synergy_joint": -0.05},
            {"category": "human", "pair_count": 4, "strict_positive_pair_count": 0, "mean_union_joint_adv": -0.1, "mean_union_synergy_joint": -0.02},
            {"category": "fruit", "pair_count": 4, "strict_positive_pair_count": 1, "mean_union_joint_adv": 0.05, "mean_union_synergy_joint": 0.01},
            {"category": "object", "pair_count": 4, "strict_positive_pair_count": 1, "mean_union_joint_adv": 0.04, "mean_union_synergy_joint": 0.0},
            {"category": "animal", "pair_count": 4, "strict_positive_pair_count": 0, "mean_union_joint_adv": -0.03, "mean_union_synergy_joint": -0.01},
        ],
        gate_category_link={
            "top_findings": {},
            "axis_target_stats": {
                "logic": {"prototype_field_proxy": {"targets": {"union_synergy_joint": {"pearson_corr": 0.2}}}},
                "syntax": {"conflict_field_proxy": {"targets": {"union_synergy_joint": {"pearson_corr": 0.3}}}},
            },
        },
        gate_pair_link={
            "top_findings": {},
            "axis_target_stats": {
                "logic": {
                    "prototype_field_proxy": {"targets": {"union_synergy_joint": {"pearson_corr": 0.4}}},
                    "bridge_field_proxy": {"targets": {"union_synergy_joint": {"pearson_corr": -0.3}}},
                },
                "syntax": {
                    "conflict_field_proxy": {"targets": {"union_synergy_joint": {"pearson_corr": 0.2}}},
                },
            },
        },
    )
    assert payload["parts_of_speech"]["adverb"]["evidence_level"] == "inference_only"
    assert payload["structures"]["relation_axis"]["metrics"]["king_queen_jaccard"] == 0.09
