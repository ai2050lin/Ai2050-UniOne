from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_multimodel_sequential_pipeline import (  # noqa: E402
    build_command_plan,
    build_execution_plan,
    build_stage_dirs,
    model_tag,
)


def make_args(**overrides):
    defaults = {
        "models": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "Qwen/Qwen3-4B"],
        "python_exe": "python",
        "items_file": "tests/codex_temp/stage56_smoke_items.csv",
        "output_root": "tempdata/stage56_seq_smoke",
        "max_items": 0,
        "survey_per_category": 1,
        "deep_per_category": 1,
        "closure_per_category": 1,
        "anchors_per_category": 1,
        "challengers_per_category": 1,
        "supports_per_category": 0,
        "use_stage2_cleanup": False,
        "cleanup_source_items_file": "tempdata/deepseek7b_tokenizer_vocab_expander_1500_20260317/combined_seed_plus_expanded.csv",
        "cleanup_seed_file": "tests/codex/deepseek7b_nouns_english_520_clean.csv",
        "cleanup_candidate_metadata_file": "tempdata/deepseek7b_tokenizer_vocab_expander_1500_20260317/all_candidates.jsonl",
        "family_count": 4,
        "terms_per_family": 2,
        "shared_k": 8,
        "specific_k": 4,
        "signature_top_k": 64,
        "source_kinds": "family_shared,combined",
        "subset_sizes": "8,4",
        "margin_preserve_threshold": 0.8,
        "global_common_max_fraction": 1.1,
        "stage5_max_candidates": 4,
        "stage5_per_category_limit": 1,
        "stage5_max_neurons_per_candidate": 6,
        "stage5_max_neurons_per_layer": 3,
        "stage5_prototype_term_mode": "any",
        "stage5_disable_prototype_proxy": False,
        "stage5_margin_adv_threshold": 0.0,
        "stage5_margin_adv_penalty": 0.0,
        "stage6_max_instance_terms_per_category": 1,
        "stage6_strict_synergy_threshold": 0.0,
        "score_alpha": 256.0,
        "candidate_overlap_penalty": 0.15,
        "max_candidate_overlap": 0.80,
        "require_category_coverage": True,
        "dtype": "bfloat16",
        "device": "cuda",
        "seed": 42,
        "progress_every": 10,
        "dry_run": True,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_model_tag_supports_known_models():
    assert model_tag("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B") == "deepseek_7b"
    assert model_tag("Qwen/Qwen3-4B") == "qwen3_4b"


def test_build_stage_dirs_scopes_by_model():
    dirs = build_stage_dirs(Path("tempdata/out"), "Qwen/Qwen3-4B")
    assert dirs["stage2_cleanup"] == Path("tempdata/out/qwen3_4b/stage2_focus_cleanup")
    assert dirs["stage5_prototype"] == Path("tempdata/out/qwen3_4b/stage5_prototype")
    assert dirs["stage6"] == Path("tempdata/out/qwen3_4b/stage6_prototype_instance_decomposition")


def test_build_command_plan_orders_stage5_before_stage6():
    args = make_args(models=["Qwen/Qwen3-4B"])
    dirs = build_stage_dirs(Path(args.output_root), "Qwen/Qwen3-4B")
    plan = build_command_plan(args, "Qwen/Qwen3-4B", dirs)
    names = [step["name"] for step in plan]
    assert names == [
        "stage1_three_pool",
        "stage2_focus_builder",
        "stage3_causal_closure",
        "stage4_minimal_circuit",
        "stage5_prototype",
        "stage5_instance",
        "stage6_prototype_instance_decomposition",
    ]
    stage6_cmd = plan[-1]["command"]
    assert "--prototype-candidates" in stage6_cmd
    assert str(dirs["stage5_prototype"] / "candidates.jsonl") in stage6_cmd
    assert str(dirs["stage5_instance"] / "candidates.jsonl") in stage6_cmd


def test_build_command_plan_passes_real_category_prototype_constraints():
    args = make_args(
        models=["Qwen/Qwen3-4B"],
        stage5_prototype_term_mode="category_only",
        stage5_disable_prototype_proxy=True,
        stage5_margin_adv_penalty=0.05,
        stage6_strict_synergy_threshold=0.0,
    )
    dirs = build_stage_dirs(Path(args.output_root), "Qwen/Qwen3-4B")
    plan = build_command_plan(args, "Qwen/Qwen3-4B", dirs)
    stage5_proto_cmd = next(step["command"] for step in plan if step["name"] == "stage5_prototype")
    assert "--prototype-term-mode" in stage5_proto_cmd
    assert "category_only" in stage5_proto_cmd
    assert "--disable-prototype-proxy" in stage5_proto_cmd
    assert "--margin-adv-penalty" in stage5_proto_cmd
    assert "0.05" in stage5_proto_cmd
    stage6_cmd = next(step["command"] for step in plan if step["name"] == "stage6_prototype_instance_decomposition")
    assert "--strict-synergy-threshold" in stage6_cmd


def test_build_command_plan_includes_cleanup_when_enabled():
    args = make_args(models=["Qwen/Qwen3-4B"], use_stage2_cleanup=True)
    dirs = build_stage_dirs(Path(args.output_root), "Qwen/Qwen3-4B")
    plan = build_command_plan(args, "Qwen/Qwen3-4B", dirs)
    names = [step["name"] for step in plan]
    assert names[0:3] == [
        "stage1_three_pool",
        "stage2_focus_builder",
        "stage2_focus_cleanup",
    ]
    stage3_cmd = plan[3]["command"]
    assert str(dirs["stage2_cleanup"] / "cleaned_focus_manifest.json") in stage3_cmd


def test_build_execution_plan_is_model_serial_not_interleaved():
    args = make_args()
    plan = build_execution_plan(args)
    model_steps = [(step["model_tag"], step["name"]) for step in plan]
    assert model_steps[:7] == [
        ("deepseek_7b", "stage1_three_pool"),
        ("deepseek_7b", "stage2_focus_builder"),
        ("deepseek_7b", "stage3_causal_closure"),
        ("deepseek_7b", "stage4_minimal_circuit"),
        ("deepseek_7b", "stage5_prototype"),
        ("deepseek_7b", "stage5_instance"),
        ("deepseek_7b", "stage6_prototype_instance_decomposition"),
    ]
    assert model_steps[7:14] == [
        ("qwen3_4b", "stage1_three_pool"),
        ("qwen3_4b", "stage2_focus_builder"),
        ("qwen3_4b", "stage3_causal_closure"),
        ("qwen3_4b", "stage4_minimal_circuit"),
        ("qwen3_4b", "stage5_prototype"),
        ("qwen3_4b", "stage5_instance"),
        ("qwen3_4b", "stage6_prototype_instance_decomposition"),
    ]
