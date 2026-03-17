from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_large_scale_discovery_inventory import (  # noqa: E402
    build_inventory,
    parse_categories,
    read_source_items,
)


def test_read_source_items_groups_by_category(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("# noun,category\nanimal,animal\nrabbit,animal\nhuman,human\nteacher,human\n", encoding="utf-8")
    grouped = read_source_items(str(source))
    assert grouped == {
        "animal": ["animal", "rabbit"],
        "human": ["human", "teacher"],
    }


def test_parse_categories_defaults_to_all():
    categories = parse_categories("", ["vehicle", "animal"])
    assert categories == ["animal", "vehicle"]


def test_build_inventory_keeps_category_word_and_requested_count():
    grouped = {
        "animal": ["animal", "rabbit", "cat", "dog"],
        "human": ["human", "teacher", "doctor", "student"],
    }
    plan = build_inventory(grouped, categories=["animal", "human"], terms_per_category=3, seed=42)
    assert sorted(plan) == ["animal", "human"]
    assert len(plan["animal"]) == 3
    assert len(plan["human"]) == 3
    assert plan["animal"][0] == "animal"
    assert plan["human"][0] == "human"


def test_build_inventory_allows_missing_category_word_when_not_required():
    grouped = {
        "abstract": ["justice", "truth", "freedom", "beauty"],
    }
    plan = build_inventory(grouped, categories=["abstract"], terms_per_category=3, seed=42)
    assert plan["abstract"][0] != "abstract"
    assert len(plan["abstract"]) == 3
