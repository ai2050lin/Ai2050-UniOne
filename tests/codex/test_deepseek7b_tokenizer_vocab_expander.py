from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parent / "deepseek7b_tokenizer_vocab_expander.py"
    spec = importlib.util.spec_from_file_location("deepseek7b_tokenizer_vocab_expander", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


mod = load_module()


class FakeTokenizer:
    def __init__(self) -> None:
        self.mapping = {
            " cat": [11],
            "cat": [11],
            " algorithm": [21],
            "algorithm": [21],
            " justice": [31],
            "justice": [31],
            " hello": [41],
            "hello": [41],
            "value": [77, 88],
        }

    def encode(self, text, add_special_tokens=False):
        return list(self.mapping.get(text, []))

    def decode(self, ids, clean_up_tokenization_spaces=False):
        rev = {
            11: " cat",
            21: " algorithm",
            31: " justice",
            41: " hello",
        }
        return rev[ids[0]]


def test_normalize_term_filters_noise() -> None:
    assert mod.normalize_term(" cat") == "cat"
    assert mod.normalize_term("RESET") is None
    assert mod.normalize_term("do") is None
    assert mod.normalize_term("hello!") is None
    assert mod.normalize_term("there") is None


def test_resolve_single_token_id_prefers_valid_single_token() -> None:
    tok = FakeTokenizer()
    assert mod.resolve_single_token_id(tok, "cat") == 11
    assert mod.resolve_single_token_id(tok, "algorithm") == 21
    assert mod.resolve_single_token_id(tok, "value") is None


def test_select_balanced_inventory_applies_thresholds() -> None:
    rows = [
        {"term": "cat", "top_category": "animal", "top_score": 0.42, "margin": 0.09},
        {"term": "dog", "top_category": "animal", "top_score": 0.40, "margin": 0.07},
        {"term": "bus", "top_category": "vehicle", "top_score": 0.39, "margin": 0.02},
        {"term": "weak", "top_category": "vehicle", "top_score": 0.11, "margin": 0.08},
    ]
    out = mod.select_balanced_inventory(rows, per_category_target=1, min_score=0.20, min_margin=0.03)
    assert [(row["term"], row["top_category"]) for row in out] == [("cat", "animal")]
