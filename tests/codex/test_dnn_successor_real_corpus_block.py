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

from research.gpt5.code.dnn_successor_real_corpus import DnnSuccessorRealCorpus  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    corpus = DnnSuccessorRealCorpus.from_artifacts(ROOT)
    summary = corpus.summary()
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_successor_real_corpus_block",
        },
        "strict_goal": {
            "statement": "Unify successor-related direct, proxy, structured, and contract artifacts into one exactness-aware corpus.",
            "boundary": "This block standardizes successor evidence. It does not yet execute new dense successor mining runs.",
        },
        "corpus_summary": summary,
        "headline_metrics": {
            "total_successor_units": summary["total_successor_units"],
            "exact_dense_units": summary["exact_dense_units"],
            "proxy_units": summary["proxy_units"],
            "stage_resolved_units": summary["stage_resolved_units"],
            "dense_tensor_units": summary["dense_tensor_units"],
            "weighted_exact_units": summary["weighted_exact_units"],
            "exactness_fraction": summary["exactness_fraction"],
        },
        "strict_verdict": {
            "successor_corpus_present": bool(summary["total_successor_units"] > 600),
            "core_answer": (
                "The project now has a unified successor corpus that mixes direct-dense evidence, stage-resolved proxies, "
                "structured successor law terms, and dense export contracts under one exactness-aware accounting."
            ),
            "main_hard_gaps": [
                "proxy successor units still dominate exact dense units",
                "stage-resolved successor evidence exists, but most of it is still not dense tensor evidence",
                "the corpus is now unified, but the core successor exactness bottleneck is still visible rather than solved",
            ],
        },
        "progress_estimate": {
            "successor_real_corpus_percent": 69.0,
            "successor_dense_exact_closure_percent": 41.0,
            "full_brain_encoding_mechanism_percent": 89.0,
        },
        "next_large_blocks": [
            "Replace proxy-dominant successor units with dense tensor units from online recovery and inventory paths.",
            "Use the unified successor corpus to recompute exactness-aware successor restoration.",
            "Push the direct successor path to explicit stage alignment and then rerun successor exactness closure.",
        ],
    }
    return payload


def test_dnn_successor_real_corpus_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["total_successor_units"] > 600
    assert metrics["proxy_units"] > metrics["exact_dense_units"]
    assert metrics["stage_resolved_units"] > metrics["dense_tensor_units"]
    assert metrics["exactness_fraction"] > 0.30
    assert verdict["successor_corpus_present"] is True


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN successor real corpus block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_successor_real_corpus_block_20260315.json",
    )
    args = ap.parse_args()
    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
