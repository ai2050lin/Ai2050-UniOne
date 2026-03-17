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

from research.gpt5.code.dnn_1000plus_dense_execution_bundle import Dnn1000PlusDenseExecutionBundle  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    bundle = Dnn1000PlusDenseExecutionBundle.from_repo(ROOT)
    summary = bundle.summary()
    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "dnn_1000plus_dense_execution_bundle_block",
        },
        "strict_goal": {
            "statement": "Wire the 1000+ noun source into launchable balanced batches that directly target mass noun scan and specific dense signature harvesting.",
            "boundary": "This block creates the execution bundle and batch artifacts. It does not run the expensive model harvest itself.",
        },
        "headline_metrics": summary["headline_metrics"],
        "task_rows": summary["task_rows"],
        "strict_conclusion": summary["strict_conclusion"],
    }


def test_dnn_1000plus_dense_execution_bundle_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    assert len(metrics["metric_lines_cn"]) >= 8
    assert metrics["metric_lines_cn"][0].startswith("（")
    assert metrics["batch_count"] >= 8
    assert metrics["total_batched_nouns"] >= 1000
    assert metrics["anchored_batch_count"] >= 3
    assert metrics["specific_schema_launchable"] == 1.0
    first_csv = ROOT / payload["task_rows"][0]["csv_relative_path"]
    assert first_csv.exists()


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN 1000+ dense execution bundle block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_1000plus_dense_execution_bundle_block_20260316.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["headline_metrics"]["metric_lines_cn"]))


if __name__ == "__main__":
    main()
