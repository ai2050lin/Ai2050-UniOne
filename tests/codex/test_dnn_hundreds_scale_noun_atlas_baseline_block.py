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

from research.gpt5.code.dnn_hundreds_scale_noun_atlas_baseline import DnnHundredsScaleNounAtlasBaseline  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    baseline = DnnHundredsScaleNounAtlasBaseline.from_repo(ROOT)
    summary = baseline.summary()
    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "dnn_hundreds_scale_noun_atlas_baseline_block",
        },
        "strict_goal": {
            "statement": "Establish a real hundreds-scale noun atlas baseline for family patch plus concept offset, using the current 280 unique nouns, mass scans, and large-scale inventory.",
            "boundary": "This block establishes the hundreds-scale baseline. It does not claim thousands-scale or dense theorem closure.",
        },
        "headline_metrics": {
            "metric_lines_cn": summary["metric_lines_cn"],
        },
        "baseline": summary["baseline"],
        "strict_conclusion": summary["strict_conclusion"],
    }


def test_dnn_hundreds_scale_noun_atlas_baseline_block() -> None:
    payload = build_payload()
    assert len(payload["headline_metrics"]["metric_lines_cn"]) >= 6
    assert payload["headline_metrics"]["metric_lines_cn"][0].startswith("（")
    assert payload["baseline"]["hundreds_unique_nouns"] == 280
    assert payload["baseline"]["hundreds_mass_records"] == 600
    assert payload["baseline"]["inventory_concepts"] >= 300
    assert payload["baseline"]["hundreds_category_count"] >= 10


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN hundreds-scale noun atlas baseline block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_hundreds_scale_noun_atlas_baseline_block_20260316.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["headline_metrics"]["metric_lines_cn"]))


if __name__ == "__main__":
    main()
