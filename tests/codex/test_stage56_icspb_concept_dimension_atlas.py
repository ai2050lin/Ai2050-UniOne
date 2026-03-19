from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_icspb_concept_dimension_atlas import (  # noqa: E402
    build_global_summary,
    build_verdict,
    compute_concept_row,
    parse_concept_spec,
    summarize_generation_block,
)


def test_parse_concept_spec_extracts_anchor_and_peers():
    spec = parse_concept_spec("apple|fruit|food_composed|apple,banana,pear")
    assert spec["anchor"] == "apple"
    assert spec["meso"] == "fruit"
    assert spec["macro_mode"] == "food_composed"
    assert spec["peers"] == ["apple", "banana", "pear"]


def test_compute_concept_row_reports_covered_metrics():
    noun_map = {
        "apple": {"noun": "apple", "category": "fruit", "signature_top_indices": [1, 2, 3], "signature_layer_distribution": {"0": 1.0, "8": 2.0}},
        "banana": {"noun": "banana", "category": "fruit", "signature_top_indices": [2, 3, 4], "signature_layer_distribution": {"8": 1.0}},
        "pear": {"noun": "pear", "category": "fruit", "signature_top_indices": [2, 4, 5], "signature_layer_distribution": {"9": 1.0}},
    }
    cat_map = {
        "fruit": {2, 3, 6},
        "food": {2, 6, 7},
    }
    noun_cat_map = {"apple": "fruit", "banana": "fruit", "pear": "fruit"}
    spec = {"anchor": "apple", "meso": "fruit", "macro_mode": "food", "peers": ["apple", "banana", "pear"]}
    row = compute_concept_row(spec, noun_map, cat_map, noun_cat_map, n_layers=12)
    assert row["coverage_status"] == "covered"
    assert row["anchor_to_meso_jaccard"] > 0.0
    assert row["meso_to_macro_jaccard"] > 0.0
    assert row["shared_base_ratio_vs_anchor"] > 0.0
    assert row["icspb_view"]["transport_closure_gain"] >= -1.0


def test_compute_concept_row_reports_missing_anchor():
    row = compute_concept_row(
        {"anchor": "justice", "meso": "abstract", "macro_mode": "abstract", "peers": ["justice", "truth"]},
        noun_map={},
        cat_map={},
        noun_cat_map={},
        n_layers=8,
    )
    assert row["coverage_status"] == "missing_anchor"


def test_generation_summary_and_verdict_keep_rules_and_gaps():
    apple_dossier = {
        "metrics": {
            "style_logic_syntax_signal": 0.58,
            "cross_dim_decoupling_index": 0.68,
        }
    }
    gate_summary = {
        "field_consensus": {
            "style": {
                "prototype_field_proxy": {"consensus": "mixed"},
                "instance_field_proxy": {"consensus": "mixed"},
                "bridge_field_proxy": {"consensus": "positive"},
                "conflict_field_proxy": {"consensus": "mixed"},
                "mismatch_field_proxy": {"consensus": "positive"},
            },
            "logic": {
                "prototype_field_proxy": {"consensus": "positive"},
                "instance_field_proxy": {"consensus": "mixed"},
                "bridge_field_proxy": {"consensus": "positive"},
                "conflict_field_proxy": {"consensus": "positive"},
                "mismatch_field_proxy": {"consensus": "positive"},
            },
            "syntax": {
                "prototype_field_proxy": {"consensus": "negative"},
                "instance_field_proxy": {"consensus": "negative"},
                "bridge_field_proxy": {"consensus": "mixed"},
                "conflict_field_proxy": {"consensus": "positive"},
                "mismatch_field_proxy": {"consensus": "positive"},
            },
        }
    }
    generation_block = summarize_generation_block(apple_dossier, gate_summary)
    concept_rows = [
        {
            "anchor": "apple",
            "coverage_status": "covered",
            "micro_to_meso_jaccard": 0.02,
            "meso_to_macro_jaccard": 0.30,
            "shared_base_ratio_vs_anchor": 0.03,
            "icspb_view": {"section_anchor_strength": 0.05, "fiber_dispersion": 0.8},
        },
        {
            "anchor": "justice",
            "coverage_status": "missing_anchor",
        },
    ]
    relation_block = {"axis_specificity_index": 0.62}
    global_summary = build_global_summary(concept_rows, generation_block, relation_block)
    verdict = build_verdict(concept_rows, global_summary, generation_block)
    assert generation_block["axis_rules"]["logic"]["P"] == "positive"
    assert global_summary["covered_anchor_count"] == 1
    assert verdict["meso_dominant_anchors"] == ["apple"]
    assert "justice" in verdict["gap_statement"]
