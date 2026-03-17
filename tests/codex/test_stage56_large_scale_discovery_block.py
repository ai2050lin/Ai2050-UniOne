from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_large_scale_discovery_block import (  # noqa: E402
    build_aggregator_command,
    build_inventory_command,
    build_pipeline_command,
)


def make_args(**overrides):
    defaults = {
        "models": "Qwen/Qwen3-4B",
        "python_exe": "python",
        "source_file": "tests/codex/deepseek7b_nouns_english_520_clean.csv",
        "categories": "",
        "terms_per_category": 9,
        "require_category_word": False,
        "items_file": "tests/codex_temp/stage56_large_scale_discovery_items.csv",
        "inventory_manifest_file": "tests/codex_temp/stage56_large_scale_discovery_manifest.json",
        "inventory_report_file": "tests/codex_temp/stage56_large_scale_discovery_report.md",
        "output_root": "tempdata/stage56_large_scale_discovery_test",
        "survey_per_category": 9,
        "deep_per_category": 6,
        "closure_per_category": 4,
        "anchors_per_category": 2,
        "challengers_per_category": 3,
        "supports_per_category": 2,
        "use_stage2_cleanup": False,
        "family_count": 10,
        "terms_per_family": 4,
        "shared_k": 48,
        "specific_k": 24,
        "signature_top_k": 256,
        "subset_sizes": "48,32,24,16,12,8,6,4",
        "stage5_max_candidates": 30,
        "stage5_per_category_limit": 3,
        "stage5_max_neurons_per_candidate": 12,
        "stage5_max_neurons_per_layer": 4,
        "stage5_prototype_term_mode": "any",
        "stage5_disable_prototype_proxy": False,
        "stage5_margin_adv_threshold": 0.0,
        "stage5_margin_adv_penalty": 0.02,
        "stage6_max_instance_terms_per_category": 3,
        "stage6_strict_synergy_threshold": 0.0,
        "score_alpha": 256.0,
        "candidate_overlap_penalty": 0.15,
        "max_candidate_overlap": 1.0,
        "dtype": "bfloat16",
        "device": "cuda",
        "seed": 42,
        "progress_every": 25,
        "resume": False,
        "dry_run": True,
        "inventory_only": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_inventory_command_writes_codex_temp_targets():
    args = make_args()
    command = build_inventory_command(args)
    assert "--output-file" in command
    assert any(str(value).endswith("stage56_large_scale_discovery_items.csv") for value in command)
    assert "--terms-per-category" in command
    assert "9" in command


def test_build_pipeline_command_uses_large_scale_defaults():
    args = make_args(stage5_disable_prototype_proxy=True, use_stage2_cleanup=True)
    command = build_pipeline_command(args)
    assert "--family-count" in command
    assert "10" in command
    assert "--stage5-per-category-limit" in command
    assert "3" in command
    assert "--stage5-prototype-term-mode" in command
    assert "any" in command
    assert "--stage5-disable-prototype-proxy" in command
    assert "--use-stage2-cleanup" in command
    assert "--dry-run" in command
    assert "--max-candidate-overlap" in command
    assert "1.0" in command


def test_build_aggregator_command_targets_output_root():
    args = make_args(output_root="tempdata/stage56_large_scale_discovery_test")
    command = build_aggregator_command(args)
    assert str(Path("tempdata/stage56_large_scale_discovery_test") / "discovery_summary.json") in command
