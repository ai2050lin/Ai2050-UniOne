from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_large_scale_discovery_aggregator import aggregate_output_root  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_aggregate_output_root_summarizes_models_and_categories(tmp_path):
    model_root = tmp_path / "qwen3_4b"
    write_json(model_root / "stage3_causal_closure" / "summary.json", {"selected_categories": ["animal", "human"]})
    write_json(
        model_root / "stage5_prototype" / "summary.json",
        {
            "candidate_count": 2,
            "mean_candidate_full_strict_joint_adv": 0.1,
            "strict_positive_micro_circuit_count": 1,
        },
    )
    write_jsonl(
        model_root / "stage5_prototype" / "candidates.jsonl",
        [
            {"candidate_layer_distribution": {"10": 2, "11": 1}},
            {"candidate_layer_distribution": {"10": 1, "12": 3}},
        ],
    )
    write_json(
        model_root / "stage5_instance" / "summary.json",
        {
            "candidate_count": 3,
            "mean_candidate_full_strict_joint_adv": 0.05,
            "strict_positive_micro_circuit_count": 0,
        },
    )
    write_jsonl(
        model_root / "stage5_instance" / "candidates.jsonl",
        [
            {"candidate_layer_distribution": {"12": 1}},
            {"candidate_layer_distribution": {"13": 2}},
        ],
    )
    write_json(
        model_root / "stage6_prototype_instance_decomposition" / "summary.json",
        {
            "model_id": "Qwen/Qwen3-4B",
            "pair_count": 2,
            "strict_positive_synergy_pair_count": 1,
            "strict_positive_synergy_categories": ["human"],
            "mean_union_joint_adv": 0.2,
            "mean_union_synergy_joint": 0.03,
        },
    )
    write_jsonl(
        model_root / "stage6_prototype_instance_decomposition" / "results.jsonl",
        [
            {
                "category": "human",
                "instance_term": "teacher",
                "prototype_neuron_count": 6,
                "instance_neuron_count": 4,
                "union_neuron_count": 10,
                "overlap_neuron_count": 0,
                "union_joint_adv": 0.3,
                "union_synergy_joint": 0.04,
                "union_margin_adv": 0.01,
                "strict_positive_synergy": True,
            },
            {
                "category": "animal",
                "instance_term": "rabbit",
                "prototype_neuron_count": 6,
                "instance_neuron_count": 6,
                "union_neuron_count": 9,
                "overlap_neuron_count": 3,
                "union_joint_adv": 0.1,
                "union_synergy_joint": -0.01,
                "union_margin_adv": 0.0,
                "strict_positive_synergy": False,
            },
        ],
    )

    payload = aggregate_output_root(tmp_path)
    assert payload["summary"]["model_count"] == 1
    assert payload["summary"]["total_pair_count"] == 2
    assert payload["summary"]["total_strict_positive_pair_count"] == 1
    assert payload["summary"]["categories_with_any_strict_positive"] == ["human"]
    assert payload["per_model_rows"][0]["top_prototype_layers"][0] == {"layer": 10, "count": 3}
    assert payload["per_category_rows"][0]["record_type"] == "stage56_large_scale_discovery_category_summary"
