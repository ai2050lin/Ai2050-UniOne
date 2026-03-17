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

from research.gpt5.code.dnn_local_extraction_completion_board import DnnLocalExtractionCompletionBoard  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    board = DnnLocalExtractionCompletionBoard.from_repo(ROOT)
    summary = board.summary()
    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "dnn_local_extraction_completion_board_block",
        },
        "strict_goal": {
            "statement": "Give a hard local-status verdict on how much of the noun extraction work is truly done under current offline constraints, and where the blocker has moved.",
            "boundary": "This block measures local completion and blocker position. It does not claim that real model harvest has already been completed.",
        },
        "headline_metrics": summary["headline_metrics"],
        "strict_conclusion": summary["strict_conclusion"],
    }


def test_dnn_local_extraction_completion_board_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    assert metrics["metric_lines_cn"][0].startswith("（")
    assert metrics["offline_preparation_score"] > 0.9
    assert metrics["real_harvest_completion"] == 0.0
    assert metrics["local_completion_ceiling"] >= 0.7


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN local extraction completion board block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_local_extraction_completion_board_block_20260316.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["headline_metrics"]["metric_lines_cn"]))


if __name__ == "__main__":
    main()
