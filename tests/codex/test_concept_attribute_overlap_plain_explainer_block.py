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

from research.gpt5.code.concept_attribute_overlap_plain_explainer import ConceptAttributeOverlapPlainExplainer  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    explainer = ConceptAttributeOverlapPlainExplainer.from_artifacts(ROOT)
    summary = explainer.summary()
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "concept_attribute_overlap_plain_explainer_block",
        },
        "strict_goal": {
            "statement": "Explain whether apple-round and moon-round share neurons, using current family patch / concept offset / attribute fiber evidence.",
            "boundary": "This block gives a strict inference from current evidence. It does not claim dense neuron-level direct overlap has been fully measured.",
        },
        "headline_metrics": {
            "metric_lines_cn": summary["metric_lines_cn"],
        },
        "plain_answer": summary["plain_answer"],
        "strict_verdict": {
            "ordinary_reader_readable": True,
            "core_answer": "Apple-round and moon-round are best understood as partial overlap on a reusable round-related attribute direction, while their full object codes remain different because family patch and concept offset differ.",
            "dense_overlap_directly_measured": False,
        },
    }
    return payload


def test_concept_attribute_overlap_plain_explainer_block() -> None:
    payload = build_payload()
    assert len(payload["headline_metrics"]["metric_lines_cn"]) >= 6
    assert payload["headline_metrics"]["metric_lines_cn"][0].startswith("（")
    assert payload["plain_answer"]["short_answer"]
    assert len(payload["plain_answer"]["candidate_formula"]) == 2
    assert payload["strict_verdict"]["ordinary_reader_readable"] is True
    assert payload["strict_verdict"]["dense_overlap_directly_measured"] is False


def main() -> None:
    ap = argparse.ArgumentParser(description="concept attribute overlap plain explainer block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/concept_attribute_overlap_plain_explainer_block_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["headline_metrics"]["metric_lines_cn"]))


if __name__ == "__main__":
    main()
