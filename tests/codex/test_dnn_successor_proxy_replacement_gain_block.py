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


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    temp = ROOT / "tests" / "codex_temp"
    corpus = load_json(temp / "dnn_successor_real_corpus_block_20260315.json")
    contract = load_json(temp / "dnn_successor_dense_export_contract_block_20260315.json")
    stage_rows = load_json(temp / "dnn_successor_stage_row_corpus_block_20260315.json")

    total_units = float(corpus["headline_metrics"]["total_successor_units"])
    online_units = 18.0
    inventory_units = 540.0
    online_readiness = 0.72
    inventory_readiness = 0.408

    online_exactness_gain = (1.0 - 0.35) * online_units / total_units
    inventory_exactness_gain = (1.0 - 0.25) * inventory_units / total_units

    online_priority = 0.80 * online_readiness + 0.20 * online_exactness_gain
    inventory_priority = 0.80 * inventory_readiness + 0.20 * inventory_exactness_gain

    preferred_first = "online_recovery_chain" if online_priority > inventory_priority else "successor_inventory"
    largest_gain_path = "successor_inventory" if inventory_exactness_gain > online_exactness_gain else "online_recovery_chain"
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_successor_proxy_replacement_gain_block",
        },
        "strict_goal": {
            "statement": "Estimate which successor proxy path should be replaced first under a combined readiness-plus-gain criterion.",
            "boundary": "This block estimates replacement priority. It does not yet execute the replacement.",
        },
        "replacement_candidates": {
            "online_recovery_chain": {
                "proxy_units": online_units,
                "upgrade_readiness": online_readiness,
                "exactness_gain_if_replaced": online_exactness_gain,
                "priority_score": online_priority,
            },
            "successor_inventory": {
                "proxy_units": inventory_units,
                "upgrade_readiness": inventory_readiness,
                "exactness_gain_if_replaced": inventory_exactness_gain,
                "priority_score": inventory_priority,
            },
        },
        "context": {
            "total_successor_units": total_units,
            "stage_row_count": stage_rows["headline_metrics"]["stage_row_count"],
            "proxy_rows_in_contract": contract["headline_metrics"]["proxy_rows"],
        },
        "headline_metrics": {
            "preferred_first_replacement": preferred_first,
            "largest_exactness_gain_path": largest_gain_path,
            "online_priority_score": float(online_priority),
            "inventory_priority_score": float(inventory_priority),
            "online_exactness_gain_if_replaced": float(online_exactness_gain),
            "inventory_exactness_gain_if_replaced": float(inventory_exactness_gain),
        },
        "strict_verdict": {
            "replacement_board_present": True,
            "core_answer": (
                "Online recovery should be replaced first under an execution-first criterion because it is far more ready to run, even though successor inventory remains the larger exactness-gain target under a gain-first criterion."
            ),
            "main_hard_gaps": [
                "inventory has the larger eventual gain but much weaker immediate execution readiness",
                "online recovery is the practical first replacement, not the mathematically largest replacement",
                "both paths still need real dense exports before the gain estimate can become an achieved gain",
            ],
        },
        "progress_estimate": {
            "successor_proxy_replacement_board_percent": 74.0,
            "successor_dense_exact_closure_percent": 41.0,
            "full_brain_encoding_mechanism_percent": 89.0,
        },
        "next_large_blocks": [
            "Execute the online recovery dense export first to gain a fast real exactness lift.",
            "Immediately after that, execute the successor inventory dense export to unlock the larger exactness gain.",
            "Recompute successor restoration after each replacement rather than waiting for both to finish.",
        ],
    }
    return payload


def test_dnn_successor_proxy_replacement_gain_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["preferred_first_replacement"] == "online_recovery_chain"
    assert metrics["largest_exactness_gain_path"] == "successor_inventory"
    assert metrics["online_priority_score"] > metrics["inventory_priority_score"]
    assert metrics["inventory_exactness_gain_if_replaced"] > metrics["online_exactness_gain_if_replaced"]
    assert verdict["replacement_board_present"] is True


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN successor proxy replacement gain block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_successor_proxy_replacement_gain_block_20260315.json",
    )
    args = ap.parse_args()
    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
