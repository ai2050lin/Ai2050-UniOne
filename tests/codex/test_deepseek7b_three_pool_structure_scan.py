from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from deepseek7b_three_pool_structure_scan import (  # noqa: E402
    SCHEMA_VERSION,
    LexemeItem,
    PoolTask,
    analyze_task,
    build_closure_candidates,
    build_family_prototypes,
    build_pool_tasks,
    export_bundle,
    finite_stats,
    load_items,
)


def test_build_pool_tasks_nested_membership(tmp_path):
    csv_path = tmp_path / "items.csv"
    csv_path.write_text(
        "\n".join(
            [
                "# noun,category",
                "apple,fruit",
                "banana,fruit",
                "orange,fruit",
                "cat,animal",
                "dog,animal",
                "horse,animal",
            ]
        ),
        encoding="utf-8",
    )

    items = load_items(str(csv_path), max_items=0)
    tasks = build_pool_tasks(items, survey_per_category=3, deep_per_category=2, closure_per_category=1, seed=7)
    by_pool = {}
    for task in tasks:
        by_pool.setdefault(task.pool, []).append(task.item.term)

    assert len(by_pool["survey"]) == 6
    assert len(by_pool["deep"]) == 4
    assert len(by_pool["closure"]) == 2


def test_export_bundle_writes_stable_format(tmp_path):
    records = [
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "pool_record",
            "item": {"term": "apple", "category": "fruit", "language": "ascii"},
            "pool": "deep",
            "prompt_count": 3,
            "signature_top_k": 4,
            "signature_top_indices": [1, 2, 3, 4],
            "signature_top_values": [1.0, 0.9, 0.8, 0.7],
            "signature_layer_distribution": {"0": 2, "1": 2},
            "aggregate": {
                "prompt_stability_jaccard_mean": 0.8,
                "mean_prompt_l2_norm": 4.0,
                "mean_prompt_activation": 0.1,
                "top3_layer_ratio": 1.0,
            },
            "prompt_records": [],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "pool_record",
            "item": {"term": "banana", "category": "fruit", "language": "ascii"},
            "pool": "deep",
            "prompt_count": 3,
            "signature_top_k": 4,
            "signature_top_indices": [1, 2, 3, 8],
            "signature_top_values": [1.0, 0.9, 0.8, 0.6],
            "signature_layer_distribution": {"0": 3, "1": 1},
            "aggregate": {
                "prompt_stability_jaccard_mean": 0.75,
                "mean_prompt_l2_norm": 4.2,
                "mean_prompt_activation": 0.11,
                "top3_layer_ratio": 1.0,
            },
            "prompt_records": [],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "pool_record",
            "item": {"term": "cat", "category": "animal", "language": "ascii"},
            "pool": "deep",
            "prompt_count": 3,
            "signature_top_k": 4,
            "signature_top_indices": [20, 21, 22, 23],
            "signature_top_values": [1.0, 0.9, 0.8, 0.7],
            "signature_layer_distribution": {"0": 1, "1": 3},
            "aggregate": {
                "prompt_stability_jaccard_mean": 0.7,
                "mean_prompt_l2_norm": 4.1,
                "mean_prompt_activation": 0.09,
                "top3_layer_ratio": 1.0,
            },
            "prompt_records": [],
        },
    ]
    families = build_family_prototypes(records)
    closure = build_closure_candidates(records, families)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "manifest",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "device": "cuda:0",
        "runtime_sec": 1.23,
        "counts": {"input_items": 3, "tasks": 3},
        "files": {
            "manifest": "manifest.json",
            "records": "records.jsonl",
            "families": "families.jsonl",
            "closure_candidates": "closure_candidates.jsonl",
            "summary": "summary.json",
            "format": "FORMAT.md",
        },
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "summary",
        "headline_metrics": {
            "record_count": 3,
            "family_count": 2,
            "closure_candidate_count": 3,
            "survey_records": 0,
            "deep_records": 3,
            "closure_records": 0,
            "records_with_nonfinite_prompts": 0,
            "nonfinite_prompt_count_total": 0,
            "mean_prompt_stability_survey": 0.0,
            "mean_prompt_stability_deep": 0.75,
            "mean_prompt_stability_closure": 0.0,
        },
        "category_coverage_survey": {},
    }

    export_bundle(tmp_path, manifest, records, families, closure, summary)

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "records.jsonl").exists()
    assert (tmp_path / "families.jsonl").exists()
    assert (tmp_path / "closure_candidates.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "FORMAT.md").exists()

    loaded_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert loaded_manifest["schema_version"] == SCHEMA_VERSION

    record_lines = (tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()
    family_lines = (tmp_path / "families.jsonl").read_text(encoding="utf-8").splitlines()
    closure_lines = (tmp_path / "closure_candidates.jsonl").read_text(encoding="utf-8").splitlines()

    assert len(record_lines) == 3
    assert len(family_lines) == 2
    assert len(closure_lines) == 3


def test_finite_stats_reports_bad_layer():
    import numpy as np

    vec = np.array([1.0, 2.0, np.nan, np.inf, 5.0, 6.0], dtype=np.float32)
    stats = finite_stats(vec, d_ff=3)

    assert stats["nonfinite_count"] == 2
    assert stats["nan_count"] == 1
    assert stats["posinf_count"] == 1
    assert stats["bad_layers"] == [
        {"layer": 0, "nonfinite_count": 1},
        {"layer": 1, "nonfinite_count": 1},
    ]


def test_analyze_task_raises_on_nonfinite(monkeypatch):
    import numpy as np

    class FakeCollector:
        n_layers = 2
        d_ff = 3
        total_neurons = 6

        def reset(self):
            return None

        def get_flat(self):
            return np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0], dtype=np.float32)

    monkeypatch.setattr(
        "deepseek7b_three_pool_structure_scan.run_prompt",
        lambda model, tok, text: None,
    )

    task = PoolTask(item=LexemeItem(term="apple", category="fruit", language="ascii"), pool="survey")
    try:
        analyze_task(task, model=None, tok=None, collector=FakeCollector(), top_k=4, nonfinite_policy="raise")
    except RuntimeError as exc:
        assert "non-finite activations detected" in str(exc)
        assert "apple" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for non-finite activations")
