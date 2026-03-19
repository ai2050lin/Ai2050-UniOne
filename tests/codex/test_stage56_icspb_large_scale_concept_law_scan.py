from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_icspb_large_scale_concept_law_scan import (  # noqa: E402
    aggregate_global_summary,
    build_hypotheses,
    build_macro_sets,
    compute_seed_anchor_rows,
)


def test_build_macro_sets_unions_category_prototypes():
    category_prototypes = {
        "animal": {1, 2},
        "human": {2, 3},
        "tech": {9},
    }
    macro_groups = {"living_system": ["animal", "human"]}
    macro_sets = build_macro_sets(category_prototypes, macro_groups)
    assert macro_sets["living_system"] == {1, 2, 3}


def test_compute_seed_anchor_rows_produces_positive_same_cross_margin():
    seed_payload = {
        "seed_tag": "seed101",
        "noun_rows": [
            {"noun": "apple", "category": "fruit", "signature_top_indices": [1, 2, 3], "signature_layer_distribution": {"0": 1}},
            {"noun": "banana", "category": "fruit", "signature_top_indices": [1, 2, 4], "signature_layer_distribution": {"0": 1}},
            {"noun": "cat", "category": "animal", "signature_top_indices": [8, 9, 10], "signature_layer_distribution": {"9": 1}},
        ],
        "noun_map": {
            "apple": {"noun": "apple", "category": "fruit", "signature_top_indices": [1, 2, 3], "signature_layer_distribution": {"0": 1}},
            "banana": {"noun": "banana", "category": "fruit", "signature_top_indices": [1, 2, 4], "signature_layer_distribution": {"0": 1}},
            "cat": {"noun": "cat", "category": "animal", "signature_top_indices": [8, 9, 10], "signature_layer_distribution": {"9": 1}},
        },
        "category_prototypes": {"fruit": {1, 2, 5}, "animal": {8, 9, 11}},
        "macro_sets": {"living_system": {1, 2, 5, 8, 9, 11}},
        "n_layers": 12,
    }
    rows = compute_seed_anchor_rows(seed_payload, {"living_system": ["fruit", "animal"]})
    apple_row = next(row for row in rows if row["noun"] == "apple")
    assert apple_row["same_cross_margin"] > 0.0
    assert apple_row["noun_to_category_jaccard"] > 0.0


def test_global_summary_and_hypotheses_reflect_large_scale_pattern():
    anchor_rows = [
        {
            "seed_tag": "s1",
            "noun": "apple",
            "category": "fruit",
            "noun_to_category_jaccard": 0.2,
            "noun_to_best_macro_jaccard": 0.3,
            "same_category_mean_jaccard": 0.4,
            "cross_category_mean_jaccard": 0.1,
            "same_cross_margin": 0.3,
            "layer_peak_band": "late",
        },
        {
            "seed_tag": "s2",
            "noun": "cat",
            "category": "animal",
            "noun_to_category_jaccard": 0.18,
            "noun_to_best_macro_jaccard": 0.25,
            "same_category_mean_jaccard": 0.35,
            "cross_category_mean_jaccard": 0.05,
            "same_cross_margin": 0.3,
            "layer_peak_band": "early",
        },
    ]
    category_rows = [
        {"category": "fruit", "category_to_best_macro_jaccard": 0.5, "mean_noun_to_category_jaccard": 0.2},
        {"category": "animal", "category_to_best_macro_jaccard": 0.45, "mean_noun_to_category_jaccard": 0.18},
    ]
    cross_seed_rows = [
        {"cross_seed_signature_jaccard_mean": 0.2, "layer_peak_band_agreement_ratio": 1.0},
        {"cross_seed_signature_jaccard_mean": 0.18, "layer_peak_band_agreement_ratio": 0.8},
    ]
    summary = aggregate_global_summary(anchor_rows, category_rows, cross_seed_rows)
    hypotheses = build_hypotheses(summary)
    assert summary["positive_same_cross_margin_ratio"] == 1.0
    assert summary["macro_stronger_than_micro_category_ratio"] == 1.0
    assert all(row["pass"] for row in hypotheses)
