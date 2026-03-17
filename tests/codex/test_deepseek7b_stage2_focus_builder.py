from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from deepseek7b_stage2_focus_builder import (  # noqa: E402
    build_focus_plan,
    dedupe_keep_order,
)


def test_dedupe_keep_order_keeps_first_role():
    rows = [
        ("apple", "fruit", "anchor"),
        ("apple", "fruit", "challenger"),
        ("banana", "fruit", "support"),
    ]
    assert dedupe_keep_order(rows) == [
        ("apple", "fruit", "anchor"),
        ("banana", "fruit", "support"),
    ]


def test_build_focus_plan_selects_anchor_challenger_support():
    source = {
        "fruit": ["apple", "banana", "orange", "grape"],
    }
    records = [
        {
            "pool": "deep",
            "item": {"term": "banana", "category": "fruit"},
            "aggregate": {"prompt_stability_jaccard_mean": 0.8, "top3_layer_ratio": 0.9},
        },
        {
            "pool": "deep",
            "item": {"term": "orange", "category": "fruit"},
            "aggregate": {"prompt_stability_jaccard_mean": 0.7, "top3_layer_ratio": 0.8},
        },
    ]
    closure = [
        {
            "pool": "closure",
            "item": {"term": "apple", "category": "fruit"},
            "exact_closure_proxy": 0.9,
            "wrong_family_margin": 0.2,
        },
        {
            "pool": "closure",
            "item": {"term": "grape", "category": "fruit"},
            "exact_closure_proxy": 0.4,
            "wrong_family_margin": -0.1,
        },
    ]
    plan = build_focus_plan(
        source_by_category=source,
        records=records,
        closure_candidates=closure,
        anchors_per_category=1,
        challengers_per_category=1,
        supports_per_category=1,
    )
    rows = plan["fruit"]
    assert rows[0]["term"] == "apple"
    assert rows[0]["role"] == "anchor"
    assert rows[1]["term"] == "grape"
    assert rows[1]["role"] == "challenger"
    assert rows[2]["term"] == "banana"
    assert rows[2]["role"] == "support"


def test_build_focus_plan_backfills_challenger_from_weak_deep():
    source = {
        "fruit": ["apple", "banana", "orange", "grape"],
    }
    records = [
        {
            "pool": "deep",
            "item": {"term": "banana", "category": "fruit"},
            "aggregate": {"prompt_stability_jaccard_mean": 0.9, "top3_layer_ratio": 0.9},
        },
        {
            "pool": "deep",
            "item": {"term": "orange", "category": "fruit"},
            "aggregate": {"prompt_stability_jaccard_mean": 0.2, "top3_layer_ratio": 0.5},
        },
    ]
    closure = [
        {
            "pool": "closure",
            "item": {"term": "apple", "category": "fruit"},
            "exact_closure_proxy": 0.9,
            "wrong_family_margin": 0.2,
        },
    ]
    plan = build_focus_plan(
        source_by_category=source,
        records=records,
        closure_candidates=closure,
        anchors_per_category=1,
        challengers_per_category=1,
        supports_per_category=1,
    )
    rows = plan["fruit"]
    assert rows[0]["term"] == "apple"
    assert rows[0]["role"] == "anchor"
    assert rows[1]["term"] == "orange"
    assert rows[1]["role"] == "challenger"
    assert rows[2]["term"] == "banana"
    assert rows[2]["role"] == "support"
