from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_relation_discovery_deprioritized import discover_relation_mode, summarize  # noqa: E402


def test_discover_relation_mode_keeps_king_patch_local_without_family() -> None:
    row = discover_relation_mode(
        {
            "family": "gender_role_swap",
            "word_class": "noun",
            "category": "human",
            "items": ["king", "man", "woman", "queen"],
            "local_linear_score": 0.66,
            "path_bundle_score": 0.30,
            "interpretation": "local_linear",
        },
        shared_closure_categories=["fruit", "action"],
        weak_frontier_categories=["human", "tech"],
    )
    assert row["discovered_mode"] == "discovered_local_patch"
    assert row["agrees_with_prior_interpretation"] is True


def test_discover_relation_mode_pushes_protocol_like_row_to_bundle() -> None:
    row = discover_relation_mode(
        {
            "family": "protocol_role",
            "word_class": "concept",
            "category": "tech",
            "items": ["protocol", "client", "thread", "algorithm"],
            "local_linear_score": 0.19,
            "path_bundle_score": 0.90,
            "interpretation": "path_bundle",
        },
        shared_closure_categories=["fruit", "action"],
        weak_frontier_categories=["human", "tech"],
    )
    assert row["discovered_mode"] == "discovered_path_bundle"


def test_summarize_reports_agreement_ratio_and_disagreement_list() -> None:
    summary = summarize(
        [
            {
                "discovered_mode": "discovered_local_patch",
                "word_class": "noun",
                "category": "human",
                "certainty_without_family": 0.4,
                "agrees_with_prior_interpretation": True,
            },
            {
                "discovered_mode": "discovered_path_bundle",
                "word_class": "verb",
                "category": "action",
                "certainty_without_family": 0.5,
                "agrees_with_prior_interpretation": False,
                "items": ["create", "destroy", "build"],
                "family": "verb_antonym",
                "interpretation": "local_linear",
                "margin_without_family": -0.2,
            },
        ]
    )
    assert summary["group_count"] == 2
    assert abs(summary["prior_agreement_ratio"] - 0.5) < 1e-9
    assert summary["word_class_frontier"]["noun"] == "discovered_local_patch"
    assert summary["top_disagreements"][0]["family"] == "verb_antonym"
