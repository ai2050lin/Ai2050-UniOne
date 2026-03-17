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

from research.gpt5.code.dnn_1000plus_family_patch_offset_stage_target import Dnn1000PlusFamilyPatchOffsetStageTarget  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    target = Dnn1000PlusFamilyPatchOffsetStageTarget.from_repo(ROOT)
    summary = target.summary()
    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "dnn_1000plus_family_patch_offset_stage_target_block",
        },
        "strict_goal": {
            "statement": "Project the next family patch and concept offset stage targets after wiring the 1000+ noun source into a launchable dense execution bundle.",
            "boundary": "This block estimates the next stage target. It does not claim that the projected gains have already been measured by heavy harvest runs.",
        },
        "headline_metrics": summary["headline_metrics"],
        "strict_conclusion": summary["strict_conclusion"],
    }


def test_dnn_1000plus_family_patch_offset_stage_target_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    assert len(metrics["metric_lines_cn"]) >= 8
    assert metrics["metric_lines_cn"][0].startswith("（")
    assert metrics["projected_family_fit_strength"] > metrics["current_family_fit_strength"]
    assert metrics["projected_exact_specific_closure"] > metrics["current_exact_specific_closure"]
    assert metrics["projected_specific_closure_gain"] > 0.05


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN 1000+ family patch offset stage target block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_1000plus_family_patch_offset_stage_target_block_20260316.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["headline_metrics"]["metric_lines_cn"]))


if __name__ == "__main__":
    main()
