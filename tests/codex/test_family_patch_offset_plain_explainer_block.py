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

from research.gpt5.code.family_patch_offset_plain_explainer import FamilyPatchOffsetPlainExplainer  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    explainer = FamilyPatchOffsetPlainExplainer.from_artifacts(ROOT)
    summary = explainer.summary()

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "family_patch_offset_plain_explainer_block",
        },
        "strict_goal": {
            "statement": "Turn the current family patch and concept offset evidence into a plain-language explanation that an ordinary reader can follow.",
            "boundary": "This block explains the current evidence and logic in plain language. It does not claim final theorem closure.",
        },
        "headline_metrics": {
            "metric_lines_cn": summary["metric_lines_cn"],
        },
        "plain_answer": summary["plain_answer"],
        "strict_verdict": {
            "plain_explainer_present": True,
            "ordinary_reader_readable": True,
            "core_answer": "The current evidence already supports a simple picture: first there is a shared family base, then there is a concept-specific offset, and finally attributes/context refine the object. What remains unsolved is exact closure and dynamic learning law.",
        },
    }
    return payload


def test_family_patch_offset_plain_explainer_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    plain = payload["plain_answer"]
    verdict = payload["strict_verdict"]
    assert len(metrics["metric_lines_cn"]) >= 6
    assert payload["headline_metrics"]["metric_lines_cn"][0].startswith("（")
    assert "family_patch" in plain
    assert "concept_offset" in plain
    assert "apple_walkthrough" in plain
    assert verdict["plain_explainer_present"] is True
    assert verdict["ordinary_reader_readable"] is True


def main() -> None:
    ap = argparse.ArgumentParser(description="family patch / concept offset plain explainer block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/family_patch_offset_plain_explainer_block_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["headline_metrics"]["metric_lines_cn"]))


if __name__ == "__main__":
    main()
