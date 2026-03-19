from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_icspb_expanded_inventory_builder import (  # noqa: E402
    build_inventory,
    build_summary,
    sample_terms,
)


def test_sample_terms_is_deterministic_and_balanced():
    terms = ["a", "b", "c", "d", "e"]
    left = sample_terms(terms, 3, seed=7)
    right = sample_terms(terms, 3, seed=7)
    assert left == right
    assert len(left) == 3


def test_build_inventory_adds_action_and_preserves_source_categories():
    source_by_category = {
        "abstract": ["truth", "justice", "logic", "memory"],
        "fruit": ["apple", "banana", "pear", "grape"],
    }
    inventory = build_inventory(
        source_by_category,
        current_mass_categories={"fruit": 10, "weather": 10},
        current_mass_terms={"weather": ["rain", "snow", "wind", "storm"]},
        terms_per_category=3,
        seed=5,
    )
    assert "abstract" in inventory
    assert "fruit" in inventory
    assert "weather" in inventory
    assert "action" in inventory
    assert len(inventory["action"]) == 3


def test_build_inventory_can_expand_weather_beyond_current_mass_terms():
    source_by_category = {
        "fruit": ["apple", "banana", "pear", "grape"],
    }
    inventory = build_inventory(
        source_by_category,
        current_mass_categories={"weather": 10},
        current_mass_terms={"weather": ["rain", "snow", "wind", "storm"]},
        terms_per_category=12,
        seed=11,
    )
    assert "weather" in inventory
    assert len(inventory["weather"]) == 12
    assert "rain" in inventory["weather"]


def test_build_summary_detects_abstract_gap_and_action_extension():
    source_by_category = {
        "abstract": ["truth", "justice"],
        "fruit": ["apple", "banana"],
    }
    inventory = {
        "abstract": ["truth", "justice"],
        "fruit": ["apple", "banana"],
        "action": ["run", "jump"],
    }
    summary = build_summary(
        source_by_category=source_by_category,
        current_mass_categories={"fruit": 20},
        inventory=inventory,
        terms_per_category=2,
    )
    assert "abstract" in summary["missing_in_current_mass"]
    assert "action" in summary["added_by_inventory"]
