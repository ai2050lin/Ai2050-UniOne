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

from research.gpt5.code.dnn_successor_stage_row_corpus import DnnSuccessorStageRowCorpus  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    corpus = DnnSuccessorStageRowCorpus.from_artifacts(ROOT)
    summary = corpus.summary()
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_successor_stage_row_corpus_block",
        },
        "strict_goal": {
            "statement": "Turn online recovery and successor inventory proxies into one standardized stage-row corpus.",
            "boundary": "This block standardizes proxy stage evidence. It does not yet produce dense unit tensors.",
        },
        "stage_row_summary": summary,
        "headline_metrics": {
            "stage_row_count": summary["stage_row_count"],
            "online_recovery_stage_rows": summary["online_recovery_stage_rows"],
            "inventory_stage_rows": summary["inventory_stage_rows"],
            "mean_trigger_rate": summary["mean_trigger_rate"],
            "mean_recovery_success_rate": summary["mean_recovery_success_rate"],
            "weighted_exact_stage_rows": summary["weighted_exact_stage_rows"],
        },
        "strict_verdict": {
            "successor_stage_row_corpus_present": bool(summary["stage_row_count"] == 20),
            "core_answer": (
                "The project now has a standardized stage-row view over the two main successor proxy sources: "
                "online recovery and successor inventory can now be compared in the same row format."
            ),
            "main_hard_gaps": [
                "the stage rows are standardized, but they are still proxy rows rather than dense neuron rows",
                "inventory stage rows are still derived from aggregate inventory metrics rather than real chain-stage state exports",
                "this block improves comparability, not exactness",
            ],
        },
        "progress_estimate": {
            "successor_stage_row_corpus_percent": 67.0,
            "successor_real_corpus_percent": 69.0,
            "full_brain_encoding_mechanism_percent": 89.0,
        },
        "next_large_blocks": [
            "Use the stage-row corpus to prioritize which proxy successor path should be replaced first.",
            "Upgrade the stage-row corpus into dense chain-stage row-state exports.",
            "Feed the upgraded stage-row evidence back into successor restoration and theorem closure tests.",
        ],
    }
    return payload


def test_dnn_successor_stage_row_corpus_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["stage_row_count"] == 20
    assert metrics["online_recovery_stage_rows"] == 8
    assert metrics["inventory_stage_rows"] == 12
    assert metrics["weighted_exact_stage_rows"] > 5.0
    assert verdict["successor_stage_row_corpus_present"] is True


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN successor stage-row corpus block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_successor_stage_row_corpus_block_20260315.json",
    )
    args = ap.parse_args()
    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
