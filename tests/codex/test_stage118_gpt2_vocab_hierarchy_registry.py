#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "tests" / "codex" / "stage118_gpt2_vocab_hierarchy_registry.py"
SPEC = importlib.util.spec_from_file_location("stage118_gpt2_vocab_hierarchy_registry", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

OUTPUT_DIR = MODULE.OUTPUT_DIR
build_registry = MODULE.build_registry
discover_tokenizer = MODULE.discover_tokenizer
write_outputs = MODULE.write_outputs


def test_stage118_gpt2_vocab_hierarchy_registry() -> None:
    tokenizer = discover_tokenizer()
    summary, registry = build_registry(tokenizer)
    write_outputs(summary, registry)

    assert summary["vocab_size"] == 50257
    assert summary["clean_unique_word_count"] > 5000
    assert summary["registered_unique_word_count"] > 400
    assert summary["micro_count"] > 30
    assert summary["meso_count"] > 200
    assert summary["macro_count"] > 50
    assert summary["basis_offset_registry_score"] > 0.60

    apple = summary["anchor_words"]["apple"]
    fruit = summary["anchor_words"]["fruit"]
    justice = summary["anchor_words"]["justice"]
    assert apple is not None and apple["label"] == "meso"
    assert fruit is not None and fruit["label"] == "meso"
    assert justice is not None and justice["label"] == "macro"

    summary_path = OUTPUT_DIR / "summary.json"
    registry_path = OUTPUT_DIR / "registry.json"
    assert summary_path.exists()
    assert registry_path.exists()

    loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    loaded_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert loaded_summary["status_short"] in {"gpt2_vocab_registry_ready", "gpt2_vocab_registry_transition"}
    assert len(loaded_registry["rows"]) == summary["clean_unique_word_count"]


if __name__ == "__main__":
    test_stage118_gpt2_vocab_hierarchy_registry()
    print("test_stage118_gpt2_vocab_hierarchy_registry: PASS")
