from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_language_math_structure_dossier import (  # noqa: E402
    build_hypotheses,
    derive_math_law,
    failure_categories,
    layer_band,
    top_categories,
)


def test_layer_band_detects_late_layers():
    assert layer_band([{"layer": 30}, {"layer": 34}, {"layer": 35}]) == "late"


def test_top_categories_prefers_positive_and_strong_rows():
    rows = [
        {"category": "tech", "strict_positive_pair_count": 2, "mean_union_synergy_joint": 0.1, "mean_union_joint_adv": 0.2},
        {"category": "animal", "strict_positive_pair_count": 0, "mean_union_synergy_joint": -0.2, "mean_union_joint_adv": -0.1},
        {"category": "nature", "strict_positive_pair_count": 1, "mean_union_synergy_joint": 0.02, "mean_union_joint_adv": 0.03},
    ]
    assert top_categories(rows, limit=2, positive_only=True) == ["tech", "nature"]
    assert failure_categories(rows, limit=1) == ["animal"]


def test_derive_math_law_and_hypotheses_capture_expected_structure():
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
        {"category": "tech", "strict_positive_pair_count": 2, "mean_union_synergy_joint": 0.07, "mean_union_joint_adv": 0.18},
        {"category": "nature", "strict_positive_pair_count": 2, "mean_union_synergy_joint": 0.0, "mean_union_joint_adv": 0.02},
        {"category": "human", "strict_positive_pair_count": 0, "mean_union_synergy_joint": -0.01, "mean_union_joint_adv": 0.02},
        {"category": "animal", "strict_positive_pair_count": 0, "mean_union_synergy_joint": -0.22, "mean_union_joint_adv": -0.24},
    ]
    law = derive_math_law(apple_dossier, discovery_summary, per_model_rows, per_category_rows)
    hypotheses = build_hypotheses(law)
    assert law["positive_categories"][:2] == ["tech", "nature"]
    assert "animal" in law["weak_categories"]
    assert all(h["pass"] for h in hypotheses[:4])
