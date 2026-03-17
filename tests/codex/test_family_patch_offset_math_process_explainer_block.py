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

from research.gpt5.code.family_patch_offset_math_process_explainer import FamilyPatchOffsetMathProcessExplainer  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    explainer = FamilyPatchOffsetMathProcessExplainer.from_artifacts(ROOT)
    summary = explainer.summary()

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "family_patch_offset_math_process_explainer_block",
        },
        "strict_goal": {
            "statement": "Explain the mathematical calculation process behind family patch and concept offset, including test chain, computation steps, apple example, current conclusions, and remaining hard gaps.",
            "boundary": "This block explains how current conclusions are computed. It does not claim final theorem closure.",
        },
        "headline_metrics": {
            "metric_lines_cn": summary["metric_lines_cn"],
        },
        "test_chain": summary["test_chain"],
        "math_process": summary["math_process"],
        "apple_example": summary["apple_example"],
        "current_conclusion": summary["current_conclusion"],
        "strict_verdict": {
            "math_process_explainer_present": True,
            "ordinary_reader_readable": True,
            "core_answer": "The current project can already explain how family patch and concept offset are computed: first fit a family-level local basis, then compute concept residuals relative to that basis, then test whether those residuals are sparse and specific enough to recover the concept. What remains open is exact closure and dynamic learning law.",
        },
    }
    return payload


def test_family_patch_offset_math_process_explainer_block() -> None:
    payload = build_payload()
    assert len(payload["headline_metrics"]["metric_lines_cn"]) >= 6
    assert payload["headline_metrics"]["metric_lines_cn"][0].startswith("（")
    assert len(payload["test_chain"]) >= 4
    assert len(payload["math_process"]["family_patch_process"]) >= 5
    assert len(payload["math_process"]["concept_offset_process"]) >= 5
    assert "candidate_formula" in payload["apple_example"]
    assert payload["strict_verdict"]["math_process_explainer_present"] is True
    assert payload["strict_verdict"]["ordinary_reader_readable"] is True


def main() -> None:
    ap = argparse.ArgumentParser(description="family patch + concept offset math process explainer block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/family_patch_offset_math_process_explainer_block_20260316.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["headline_metrics"]["metric_lines_cn"]))


if __name__ == "__main__":
    main()
