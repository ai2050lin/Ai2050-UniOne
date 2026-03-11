#!/usr/bin/env python
"""
Build a master closure view over stage 6 and fold it into the full project view.
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
    ap = argparse.ArgumentParser(description="Stage 6 master closure view")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage6_master_closure_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage5_master = load_json(ROOT / "tests" / "codex_temp" / "stage5_master_closure_20260311.json")
    stage6a = load_json(ROOT / "tests" / "codex_temp" / "stage6a_causal_core_compression_20260311.json")
    stage6b = load_json(ROOT / "tests" / "codex_temp" / "stage6b_real_training_loop_closure_20260311.json")
    stage6c = load_json(
        ROOT / "tests" / "codex_temp" / "stage6c_long_horizon_open_environment_closure_20260311.json"
    )
    stage6d = load_json(
        ROOT / "tests" / "codex_temp" / "stage6d_brain_constraint_core_reduction_20260311.json"
    )

    stage6_blocks = {
        "stage6a_causal_core_compression": {
            "score": float(stage6a["headline_metrics"]["overall_stage6a_score"]),
            "status": "completed"
            if bool(stage6a["hypotheses"]["H5_stage6a_causal_core_compression_is_moderately_closed"])
            else "in_progress",
        },
        "stage6b_real_training_loop": {
            "score": float(stage6b["headline_metrics"]["overall_stage6b_score"]),
            "status": "completed"
            if bool(stage6b["hypotheses"]["H6_stage6b_real_training_loop_is_moderately_closed"])
            else "in_progress",
        },
        "stage6c_open_environment": {
            "score": float(stage6c["headline_metrics"]["overall_stage6c_score"]),
            "status": "completed"
            if bool(stage6c["hypotheses"]["H5_stage6c_long_horizon_open_environment_is_moderately_closed"])
            else "in_progress",
        },
        "stage6d_brain_constraint_core_reduction": {
            "score": float(stage6d["headline_metrics"]["overall_stage6d_score"]),
            "status": "completed"
            if bool(stage6d["hypotheses"]["H5_stage6d_brain_constraint_core_reduction_is_moderately_closed"])
            else "in_progress",
        },
    }

    stage6_score = sum(block["score"] for block in stage6_blocks.values()) / 4.0
    completed_stage6 = sum(1 for block in stage6_blocks.values() if block["status"] == "completed")

    project_blocks = dict(stage5_master["project_blocks"])
    project_blocks["stage6_compressed_core_phase"] = {
        "score": float(stage6_score),
        "status": "completed" if completed_stage6 == 4 else "in_progress",
    }
    overall_progress = sum(block["score"] for block in project_blocks.values()) / float(len(project_blocks))
    completed_project = sum(1 for block in project_blocks.values() if block["status"] == "completed")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage6_master_closure",
        },
        "stage6_blocks": stage6_blocks,
        "stage6_headline_metrics": {
            "overall_stage6_score": float(stage6_score),
            "completed_stage6_block_count": int(completed_stage6),
            "total_stage6_block_count": 4,
        },
        "project_blocks": project_blocks,
        "project_headline_metrics": {
            "overall_project_progress_score": float(overall_progress),
            "completed_project_block_count": int(completed_project),
            "total_project_block_count": len(project_blocks),
        },
        "project_readout": {
            "summary": (
                "Stage 6 is complete only if compressed-core formation, real training loop, long-horizon open "
                "environment behavior, and brain-side freedom reduction all stay at least moderately closed."
            ),
            "next_question": (
                "If stage 6 is complete, the next large block should turn from closure-building to direct "
                "mechanism identification: a stage-7 attempt to guess the coding law itself."
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["stage6_headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["project_headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
