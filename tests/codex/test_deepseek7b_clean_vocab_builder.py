from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from deepseek7b_clean_vocab_builder import (  # noqa: E402
    balanced_round_robin,
    build_clean_inventory,
    diff_rows,
)


def test_build_clean_inventory_filters_non_ascii_and_dedupes():
    rows = [
        ("apple", "fruit"),
        ("苹果", "fruit"),
        ("banana", "fruit"),
        ("apple", "fruit"),
        ("cat", "animal"),
        ("狗", "animal"),
    ]
    clean = build_clean_inventory(rows)
    assert clean == [
        ("cat", "animal"),
        ("apple", "fruit"),
        ("banana", "fruit"),
    ]


def test_balanced_round_robin_interleaves_categories():
    rows = [
        ("apple", "fruit"),
        ("banana", "fruit"),
        ("cat", "animal"),
        ("dog", "animal"),
    ]
    assert balanced_round_robin(rows) == [
        ("cat", "animal"),
        ("apple", "fruit"),
        ("dog", "animal"),
        ("banana", "fruit"),
    ]


def test_diff_rows_reports_missing_terms():
    clean = [("apple", "fruit"), ("memory", "abstract"), ("cluster", "tech")]
    compare = [("apple", "fruit")]
    diff = diff_rows(clean, compare)
    assert diff["missing_in_compare"] == [
        {"term": "cluster", "category": "tech"},
        {"term": "memory", "category": "abstract"},
    ]
    assert diff["extra_in_compare"] == []
