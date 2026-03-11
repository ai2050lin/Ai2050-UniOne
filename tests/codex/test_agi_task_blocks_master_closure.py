#!/usr/bin/env python
"""
Build a master closure view over task blocks 1-4.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Master closure view for task blocks 1-4")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/agi_task_blocks_master_closure_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    task1 = load_json(ROOT / "tests" / "codex_temp" / "unified_mechanism_causal_homology_20260311.json")
    task2 = load_json(ROOT / "tests" / "codex_temp" / "task_block_2_unified_training_closure_20260311.json")
    task3 = load_json(ROOT / "tests" / "codex_temp" / "task_block_3_real_model_task_bridge_closure_20260311.json")
    task4 = load_json(ROOT / "tests" / "codex_temp" / "task_block_4_brain_constraint_closure_20260311.json")

    blocks = {
        "task_block_1_causal_homology": {
            "score": float(task1["headline_metrics"]["overall_causal_homology_score"]),
            "status": "completed" if bool(task1["hypotheses"]["H6_unified_causal_homology_is_moderately_supported"]) else "in_progress",
        },
        "task_block_2_training_closure": {
            "score": float(task2["headline_metrics"]["overall_task_block_2_score"]),
            "status": "completed" if bool(task2["hypotheses"]["H6_task_block_2_is_moderately_closed"]) else "in_progress",
        },
        "task_block_3_real_model_bridge": {
            "score": float(task3["headline_metrics"]["overall_task_block_3_score"]),
            "status": "completed" if bool(task3["hypotheses"]["H5_task_block_3_is_moderately_closed"]) else "in_progress",
        },
        "task_block_4_brain_constraints": {
            "score": float(task4["headline_metrics"]["overall_task_block_4_score"]),
            "status": "completed" if bool(task4["hypotheses"]["H5_task_block_4_is_moderately_closed"]) else "in_progress",
        },
    }

    overall_progress = 0.25 * sum(block["score"] for block in blocks.values())
    completed_count = sum(1 for block in blocks.values() if block["status"] == "completed")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "master_closure_for_task_blocks_1_to_4",
        },
        "blocks": blocks,
        "headline_metrics": {
            "overall_progress_score": float(overall_progress),
            "completed_block_count": int(completed_count),
            "total_block_count": 4,
        },
        "project_readout": {
            "summary": (
                "This master view treats task blocks 1-4 as the current stage. If all four are at least moderately "
                "closed, the project should stop adding bridge dashboards and move to the next hard stage: turning "
                "brain-side and structure-side evidence into a trainable unified law under online pressure."
            ),
            "next_question": (
                "The next stage should fuse task blocks 2-4: use the unified law as the base, add brain-side penalties, "
                "and test whether online real-model metrics remain stable rather than being optimized separately."
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["blocks"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
