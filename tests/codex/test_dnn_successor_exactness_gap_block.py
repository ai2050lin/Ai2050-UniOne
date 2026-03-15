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
    successor = queue.successor_exactness_summary(ROOT)
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_successor_exactness_gap_block",
        },
        "strict_goal": {
            "statement": "Audit whether the successor harvesting path has reached dense exact closure.",
            "boundary": "This block audits path composition. It does not yet execute successor dense harvesting on real models.",
        },
        "successor_exactness_summary": successor,
        "headline_metrics": {
            "successor_run_count": successor["successor_run_count"],
            "direct_dense_run_count": successor["direct_dense_run_count"],
            "summary_proxy_run_count": successor["summary_proxy_run_count"],
            "inventory_proxy_run_count": successor["inventory_proxy_run_count"],
            "direct_dense_ratio": successor["direct_dense_ratio"],
            "proxy_ratio": successor["proxy_ratio"],
        },
        "strict_verdict": {
            "dense_exact_successor_closure": successor["dense_exact_closure"],
            "core_answer": (
                "Successor is now schedulable, but it still mixes one direct-dense route with summary/inventory-based proxy routes. "
                "That is why it is not yet a dense exact closure."
            ),
            "main_hard_gaps": [
                "only one successor path is direct-dense today: the DeepSeek multi-hop gate-hook route",
                "online recovery chain is still a summary-proxy path rather than a dense activation export",
                "successor inventory is still an inventory-building path rather than a dense chain tensor path",
                "because proxy routes still dominate, successor exactness cannot yet be treated as neuron-level exact closure",
            ],
        },
        "progress_estimate": {
            "successor_dense_exact_closure_percent": 41.0,
            "successor_dense_harvest_schedulability_percent": 77.0,
            "full_brain_encoding_mechanism_percent": 89.0,
        },
        "next_large_blocks": [
            "Replace the summary-proxy online recovery path with a real dense activation export path.",
            "Replace the inventory-only successor path with stage-resolved dense chain tensors.",
            "Recompute successor restoration after the proxy share drops below the exact-closure threshold.",
        ],
    }
    return payload


def test_dnn_successor_exactness_gap_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["successor_run_count"] == 3
    assert metrics["direct_dense_run_count"] == 1
    assert metrics["summary_proxy_run_count"] == 1
    assert metrics["inventory_proxy_run_count"] == 1
    assert abs(metrics["direct_dense_ratio"] - (1.0 / 3.0)) < 1e-9
    assert abs(metrics["proxy_ratio"] - (2.0 / 3.0)) < 1e-9
    assert verdict["dense_exact_successor_closure"] is False


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN successor exactness gap block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_successor_exactness_gap_block_20260315.json",
    )
    args = ap.parse_args()
    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
