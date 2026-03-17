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

from research.gpt5.code.dnn_clean_english_execution_bundle import DnnCleanEnglishExecutionBundle  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    bundle = DnnCleanEnglishExecutionBundle.from_repo(ROOT)
    summary = bundle.summary()
    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "dnn_clean_english_execution_bundle_block",
        },
        "strict_goal": {
            "statement": "Build a no-garble practical execution path by extracting a clean English-only noun source and batching it for the existing mass noun scan pipeline.",
            "boundary": "This block repairs the practical execution path. It does not replace the long-term need for a truly clean multilingual 1000+ source.",
        },
        "headline_metrics": summary["headline_metrics"],
        "clean_source_relative_path": summary["clean_source_relative_path"],
        "batch_rows": summary["batch_rows"],
        "strict_conclusion": summary["strict_conclusion"],
    }


def test_dnn_clean_english_execution_bundle_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    assert metrics["metric_lines_cn"][0].startswith("（")
    assert metrics["clean_unique_english_nouns"] >= 500
    assert metrics["clean_batch_count"] >= 5
    assert metrics["clean_full_category_batch_count"] >= 4
    assert (ROOT / payload["clean_source_relative_path"]).exists()
    assert (ROOT / payload["batch_rows"][0]["csv_relative_path"]).exists()


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN clean English execution bundle block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_clean_english_execution_bundle_block_20260316.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["headline_metrics"]["metric_lines_cn"]))


if __name__ == "__main__":
    main()
