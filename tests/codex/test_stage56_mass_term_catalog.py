from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_mass_term_catalog import (  # noqa: E402
    indefinite_article,
    load_terms,
    pool_term_prompts,
    term_prompts,
)


def test_load_terms_reads_csv_like_lines(tmp_path: Path):
    path = tmp_path / "items.csv"
    path.write_text("# noun,category\njustice,abstract\nrun,action\napple,fruit\n", encoding="utf-8")
    rows = load_terms(str(path), max_terms=None)
    assert [row.term for row in rows] == ["justice", "run", "apple"]
    assert [row.category for row in rows] == ["abstract", "action", "fruit"]


def test_term_prompts_switch_by_category():
    action_prompts = term_prompts("run", "action")
    abstract_prompts = term_prompts("justice", "abstract")
    fruit_prompts = term_prompts("apple", "fruit")
    assert any("to run" in prompt.lower() for prompt in action_prompts)
    assert any("idea of justice" in prompt.lower() for prompt in abstract_prompts)
    assert any("this is an apple" in prompt.lower() for prompt in fruit_prompts)


def test_articles_and_pool_prompts_refine_templates():
    assert indefinite_article("apple") == "an"
    assert indefinite_article("banana") == "a"
    survey = pool_term_prompts("rain", "weather", "survey")
    closure = pool_term_prompts("teacher", "human", "closure")
    assert survey[0].lower().startswith("the weather pattern rain")
    assert any("a teacher" in prompt.lower() for prompt in closure)
