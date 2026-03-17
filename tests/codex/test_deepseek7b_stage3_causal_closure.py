from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from deepseek7b_stage3_causal_closure import (  # noqa: E402
    FocusTerm,
    build_category_priority,
    select_terms_for_family,
)


def test_build_category_priority_prefers_margin_then_count():
    rows = [
        {"pool": "closure", "item": {"category": "fruit"}, "wrong_family_margin": 0.1, "exact_closure_proxy": 0.6},
        {"pool": "closure", "item": {"category": "fruit"}, "wrong_family_margin": 0.2, "exact_closure_proxy": 0.5},
        {"pool": "closure", "item": {"category": "tech"}, "wrong_family_margin": 0.05, "exact_closure_proxy": 0.9},
        {"pool": "closure", "item": {"category": "tech"}, "wrong_family_margin": 0.05, "exact_closure_proxy": 0.9},
    ]
    assert build_category_priority(rows, top_n=2) == ["fruit", "tech"]


def test_select_terms_for_family_prefers_anchor_then_challenger():
    rows = [
        FocusTerm(term="x3", category="fruit", role="support"),
        FocusTerm(term="x1", category="fruit", role="challenger"),
        FocusTerm(term="x0", category="fruit", role="anchor"),
        FocusTerm(term="x2", category="fruit", role="fill"),
    ]
    selected = select_terms_for_family(rows, max_terms=3)
    assert [x.term for x in selected] == ["x0", "x1", "x3"]
