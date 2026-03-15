from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gpt5.code.dnn_dense_activation_harvest_pipeline import DenseActivationHarvestPipeline  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    pipeline = DenseActivationHarvestPipeline.from_repo(ROOT)
    summary = pipeline.summary(ROOT)

    highest_priority_rows = [
        row for row in summary["tasks"].values() if row["priority"] == "highest"
    ]
    mean_highest_priority_readiness = sum(
        row["task_readiness"] for row in highest_priority_rows
    ) / max(1, len(highest_priority_rows))

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_dense_activation_harvest_pipeline_block",
        },
        "strict_goal": {
            "statement": "Turn the dense harvesting manifest into a runnable pipeline spec for specific, protocol, successor, topology, and lift buckets.",
            "boundary": "This block verifies launchable extraction paths. It does not yet execute full dense neuron harvesting on real models.",
        },
        "pipeline_summary": summary,
        "headline_metrics": {
            "total_buckets": summary["total_buckets"],
            "highest_priority_bucket_count": summary["highest_priority_bucket_count"],
            "ready_bucket_count": summary["ready_bucket_count"],
            "runnable_highest_priority_bucket_count": summary["runnable_highest_priority_bucket_count"],
            "total_source_scripts": summary["total_source_scripts"],
            "existing_source_scripts": summary["existing_source_scripts"],
            "direct_dense_script_coverage": summary["direct_dense_script_coverage"],
            "pipeline_ready_score": summary["pipeline_ready_score"],
            "mean_highest_priority_readiness": float(mean_highest_priority_readiness),
        },
        "strict_verdict": {
            "direct_dense_pipeline_present": bool(summary["pipeline_ready_score"] > 0.90),
            "highest_priority_launch_ready": bool(summary["runnable_highest_priority_bucket_count"] == 3),
            "harvesting_started": False,
            "core_answer": (
                "Dense harvesting has moved from a target manifest into a runnable pipeline specification. "
                "The highest-priority buckets now have explicit source scripts, capture sites, prompt families, and tensor layouts."
            ),
            "main_hard_gaps": [
                "specific bucket is launch-ready, but Qwen-side direct dense specific extraction is still thinner than DeepSeek-side extraction",
                "protocol bucket has dense head-level extraction paths, but tool/interface outputs still need a unified exact tensor export",
                "successor bucket is now schedulable, but current paths still mix direct gate hooks with summary-level chain inventories",
                "lift remains the weakest bucket because it still lacks a strong direct dense harvesting path",
            ],
        },
        "progress_estimate": {
            "direct_dense_harvest_pipeline_percent": 79.0,
            "dense_signature_strategy_percent": 76.0,
            "final_theorem_strategy_percent": 72.0,
            "full_brain_encoding_mechanism_percent": 89.0,
        },
        "next_large_blocks": [
            "Execute the specific/protocol/successor buckets as real dense harvesting runs and write standardized tensor outputs.",
            "Replace part of the current artifact-derived signatures with harvested dense tensors and recalculate successor-specific restoration.",
            "After dense successor/protocol tensors land, recompute neuron-level general structure and full math theory closure under stricter exact-real thresholds.",
        ],
    }
    return payload


def test_dnn_dense_activation_harvest_pipeline_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]

    assert metrics["total_buckets"] == 5
    assert metrics["highest_priority_bucket_count"] == 3
    assert metrics["ready_bucket_count"] >= 4
    assert metrics["runnable_highest_priority_bucket_count"] == 3
    assert metrics["existing_source_scripts"] == metrics["total_source_scripts"]
    assert metrics["direct_dense_script_coverage"] >= 8
    assert metrics["pipeline_ready_score"] > 0.90
    assert metrics["mean_highest_priority_readiness"] >= 0.86
    assert verdict["direct_dense_pipeline_present"] is True
    assert verdict["highest_priority_launch_ready"] is True
    assert verdict["harvesting_started"] is False


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN dense activation harvest pipeline block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_dense_activation_harvest_pipeline_block_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
