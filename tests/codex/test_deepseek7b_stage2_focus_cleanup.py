from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parent / "deepseek7b_stage2_focus_cleanup.py"
    spec = importlib.util.spec_from_file_location("deepseek7b_stage2_focus_cleanup", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


mod = load_module()


def test_audit_focus_term_trusts_seed_terms() -> None:
    audit = mod.audit_focus_term(
        term="cat",
        category="animal",
        role="anchor",
        seed_terms={("cat", "animal")},
        meta_map={},
    )
    assert audit["trusted_seed"] is True
    assert audit["risky"] is False


def test_audit_focus_term_flags_low_score_and_margin() -> None:
    audit = mod.audit_focus_term(
        term="elf",
        category="animal",
        role="anchor",
        seed_terms=set(),
        meta_map={
            "elf": {
                "top_category": "animal",
                "top_score": 0.19,
                "margin": 0.01,
                "second_category": "human",
            }
        },
    )
    assert audit["risky"] is True
    assert "low_score" in audit["flags"]
    assert "low_margin" in audit["flags"]


def test_cleanup_plan_replaces_risky_term_with_safe_seed() -> None:
    cleaned, audit_rows, board = mod.cleanup_plan(
        focus_plan={"animal": [{"term": "elf", "category": "animal", "role": "anchor"}]},
        source_by_category={"animal": ["elf", "cat", "dog"]},
        seed_terms={("cat", "animal"), ("dog", "animal")},
        meta_map={
            "elf": {"top_category": "animal", "top_score": 0.19, "margin": 0.01, "second_category": "human"},
        },
        deep_score={},
        closure_score={},
    )
    assert cleaned["animal"][0]["term"] == "dog" or cleaned["animal"][0]["term"] == "cat"
    assert audit_rows[0]["replaced"] is True
    assert len(board["animal"]) == 1
