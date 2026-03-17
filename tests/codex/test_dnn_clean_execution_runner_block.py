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

from research.gpt5.code.dnn_clean_execution_runner import DnnCleanExecutionRunner  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    runner = DnnCleanExecutionRunner.from_repo(ROOT)
    summary = runner.summary()
    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "dnn_clean_execution_runner_block",
        },
        "strict_goal": {
            "statement": "Track the execution status of the clean no-garble batch path, so the project can move from batch preparation to actual harvest completion.",
            "boundary": "This block tracks batch status. It does not itself guarantee that model inference has run successfully.",
        },
        "headline_metrics": summary["headline_metrics"],
        "run_rows": summary["run_rows"],
        "strict_conclusion": summary["strict_conclusion"],
    }


def test_dnn_clean_execution_runner_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    assert metrics["metric_lines_cn"][0].startswith("（")
    assert metrics["clean_total_batches"] >= 5
    assert len(payload["run_rows"]) == metrics["clean_total_batches"]


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN clean execution runner block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_clean_execution_runner_block_20260316.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["headline_metrics"]["metric_lines_cn"]))


if __name__ == "__main__":
    main()
