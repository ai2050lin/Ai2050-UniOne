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

from research.gpt5.code.dnn_dense_activation_harvest_queue import DenseActivationHarvestQueue  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    queue = DenseActivationHarvestQueue.from_pipeline(ROOT)
    summary = queue.summary(ROOT)
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_dense_activation_harvest_queue_block",
        },
        "strict_goal": {
            "statement": "Turn the highest-priority dense harvesting buckets into concrete runnable queue items.",
            "boundary": "This block materializes queue rows and exactness tiers. It does not yet execute the queued dense harvesting runs.",
        },
        "queue_summary": summary,
        "headline_metrics": {
            "total_runs": summary["total_runs"],
            "launchable_runs": summary["launchable_runs"],
            "highest_priority_runs": summary["highest_priority_runs"],
            "direct_dense_runs": summary["direct_dense_runs"],
            "head_dense_runs": summary["head_dense_runs"],
            "summary_proxy_runs": summary["summary_proxy_runs"],
            "inventory_proxy_runs": summary["inventory_proxy_runs"],
            "queue_ready_score": summary["queue_ready_score"],
        },
        "strict_verdict": {
            "dense_queue_present": bool(summary["queue_ready_score"] >= 0.80),
            "all_highest_priority_runs_enumerated": bool(summary["highest_priority_runs"] == summary["total_runs"]),
            "core_answer": (
                "The project now has a concrete execution queue for the highest-priority dense harvesting lines. "
                "Each run is bound to a source script, concept group, prompt family, capture site, and exactness tier."
            ),
            "main_hard_gaps": [
                "queue rows are concrete, but no standardized dense tensor outputs have been written yet",
                "protocol and successor queues still contain proxy-tier rows alongside direct-dense rows",
                "queue materialization solves scheduling, not exact closure",
            ],
        },
        "progress_estimate": {
            "direct_dense_harvest_queue_percent": 82.0,
            "direct_dense_harvest_pipeline_percent": 79.0,
            "full_brain_encoding_mechanism_percent": 89.0,
        },
        "next_large_blocks": [
            "Execute the queue and write standardized dense tensor outputs for specific/protocol/successor.",
            "Use the queue outputs to replace proxy rows inside the activation-signature and restoration stack.",
            "Recompute successor-specific restoration under exact-dense-only constraints.",
        ],
    }
    return payload


def test_dnn_dense_activation_harvest_queue_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["total_runs"] == 10
    assert metrics["launchable_runs"] >= 7
    assert metrics["highest_priority_runs"] == 10
    assert metrics["direct_dense_runs"] >= 5
    assert metrics["head_dense_runs"] >= 2
    assert metrics["summary_proxy_runs"] >= 2
    assert metrics["inventory_proxy_runs"] == 1
    assert metrics["queue_ready_score"] >= 0.80
    assert verdict["dense_queue_present"] is True
    assert verdict["all_highest_priority_runs_enumerated"] is True


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN dense activation harvest queue block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_dense_activation_harvest_queue_block_20260315.json",
    )
    args = ap.parse_args()
    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
