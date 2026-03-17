from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_multiaxis_language_analyzer import (  # noqa: E402
    build_concept_axis,
    build_generation_axis,
    build_joint_law,
    build_payload,
)


def test_concept_and_generation_axis_classification() -> None:
    metrics = {
        "apple_micro_to_meso_jaccard_mean": 0.02,
        "apple_meso_to_macro_jaccard_mean": 0.37,
        "apple_shared_base_ratio_mean": 0.03,
        "style_logic_syntax_signal": 0.58,
        "cross_dim_decoupling_index": 0.68,
        "axis_specificity_index": 0.62,
        "triplet_separability_index": 0.09,
    }
    concept_axis = build_concept_axis(metrics)
    generation_axis = build_generation_axis(metrics)
    assert concept_axis["hierarchy_type"] == "macro_bridge_dominant"
    assert generation_axis["generation_type"] == "parallel_decoupled_axes"


def test_joint_law_and_payload_capture_expected_structure() -> None:
    apple_dossier = {
        "metrics": {
            "apple_micro_to_meso_jaccard_mean": 0.02,
            "apple_meso_to_macro_jaccard_mean": 0.37,
            "apple_shared_base_ratio_mean": 0.03,
            "style_logic_syntax_signal": 0.58,
            "cross_dim_decoupling_index": 0.68,
            "axis_specificity_index": 0.62,
            "triplet_separability_index": 0.09,
        }
    }
    discovery_summary = {
        "strict_positive_pair_ratio": 0.24,
        "margin_zero_pair_ratio": 0.04,
    }
    per_model_rows = [
        {
            "model_tag": "deepseek_7b",
            "top_prototype_layers": [{"layer": 26}, {"layer": 27}],
            "top_instance_layers": [{"layer": 25}, {"layer": 27}],
            "strict_positive_pair_ratio": 0.04,
        },
        {
            "model_tag": "qwen3_4b",
            "top_prototype_layers": [{"layer": 34}, {"layer": 35}],
            "top_instance_layers": [{"layer": 30}, {"layer": 31}],
            "strict_positive_pair_ratio": 0.41,
        },
    ]
    per_category_rows = [
        {"category": "tech", "pair_count": 4, "strict_positive_pair_count": 2, "mean_union_synergy_joint": 0.07, "mean_union_joint_adv": 0.18},
        {"category": "nature", "pair_count": 5, "strict_positive_pair_count": 2, "mean_union_synergy_joint": 0.00, "mean_union_joint_adv": 0.02},
        {"category": "fruit", "pair_count": 5, "strict_positive_pair_count": 1, "mean_union_synergy_joint": -0.01, "mean_union_joint_adv": 0.02},
        {"category": "animal", "pair_count": 5, "strict_positive_pair_count": 0, "mean_union_synergy_joint": -0.22, "mean_union_joint_adv": -0.24},
    ]

    payload = build_payload(apple_dossier, discovery_summary, per_model_rows, per_category_rows)
    joint_law = build_joint_law(
        payload["concept_axis"],
        payload["generation_axis"],
        payload["cross_model_support"],
        per_category_rows,
    )

    assert payload["concept_axis"]["hierarchy_type"] == "macro_bridge_dominant"
    assert payload["generation_axis"]["generation_type"] == "parallel_decoupled_axes"
    assert payload["cross_model_support"]["strongest_categories"][0] == "tech"
    assert "animal" in payload["cross_model_support"]["weakest_categories"]
    assert joint_law["strong_claim"]["supported"] is True
    assert joint_law["category_support"]["fruit"]["strict_positive_pair_count"] == 1
