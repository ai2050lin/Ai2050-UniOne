#!/usr/bin/env python
"""
Build a master closure view over stage 5 and then fold it into the full project
progress view.
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
    ap = argparse.ArgumentParser(description="Stage 5 master closure view")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage5_master_closure_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    prior_master = load_json(ROOT / "tests" / "codex_temp" / "agi_task_blocks_master_closure_20260311.json")
    stage5_proxy = load_json(ROOT / "tests" / "codex_temp" / "stage5_fused_unified_law_objective_20260311.json")
    stage5a = load_json(ROOT / "tests" / "codex_temp" / "stage5a_real_fused_loss_closure_20260311.json")
    stage5b = load_json(ROOT / "tests" / "codex_temp" / "stage5b_structure_reinforcement_closure_20260311.json")
    stage5c = load_json(
        ROOT / "tests" / "codex_temp" / "stage5c_online_failure_integrated_training_closure_20260311.json"
    )
    stage5d = load_json(
        ROOT / "tests" / "codex_temp" / "stage5d_cross_model_unified_calibration_closure_20260311.json"
    )

    stage5_blocks = {
        "stage5_proxy_objective": {
            "score": float(stage5_proxy["headline_metrics"]["best_fused_score"]),
            "status": "completed"
            if bool(stage5_proxy["hypotheses"]["H5_stage5_proxy_objective_is_viable"])
            else "in_progress",
        },
        "stage5a_real_fused_loss": {
            "score": float(stage5a["headline_metrics"]["overall_stage5a_score"]),
            "status": "completed"
            if bool(stage5a["hypotheses"]["H5_stage5a_real_fused_loss_is_moderately_closed"])
            else "in_progress",
        },
        "stage5b_structure_reinforcement": {
            "score": float(stage5b["headline_metrics"]["overall_stage5b_score"]),
            "status": "completed"
            if bool(stage5b["hypotheses"]["H5_stage5b_structure_reinforcement_is_moderately_closed"])
            else "in_progress",
        },
        "stage5c_online_failure_integration": {
            "score": float(stage5c["headline_metrics"]["overall_stage5c_score"]),
            "status": "completed"
            if bool(stage5c["hypotheses"]["H5_stage5c_online_failure_integration_is_moderately_closed"])
            else "in_progress",
        },
        "stage5d_cross_model_calibration": {
            "score": float(stage5d["headline_metrics"]["overall_stage5d_score"]),
            "status": "completed"
            if bool(stage5d["hypotheses"]["H5_stage5d_cross_model_calibration_is_moderately_closed"])
            else "in_progress",
        },
    }

    stage5_score = sum(block["score"] for block in stage5_blocks.values()) / 5.0
    completed_stage5 = sum(1 for block in stage5_blocks.values() if block["status"] == "completed")

    project_blocks = dict(prior_master["blocks"])
    project_blocks["stage5_fused_unified_law"] = {
        "score": float(stage5_score),
        "status": "completed" if completed_stage5 == 5 else "in_progress",
    }
    overall_progress = sum(block["score"] for block in project_blocks.values()) / float(len(project_blocks))
    completed_project = sum(1 for block in project_blocks.values() if block["status"] == "completed")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage5_master_closure",
        },
        "stage5_blocks": stage5_blocks,
        "stage5_headline_metrics": {
            "overall_stage5_score": float(stage5_score),
            "completed_stage5_block_count": int(completed_stage5),
            "total_stage5_block_count": 5,
        },
        "project_blocks": project_blocks,
        "project_headline_metrics": {
            "overall_project_progress_score": float(overall_progress),
            "completed_project_block_count": int(completed_project),
            "total_project_block_count": len(project_blocks),
        },
        "project_readout": {
            "summary": (
                "Stage 5 is treated as complete only if proxy fusion, real fused loss alignment, structure "
                "reinforcement, online failure integration, and cross-model calibration all stay at least moderately closed."
            ),
            "next_question": (
                "If stage 5 is complete, the next major block should stop expanding dashboards and instead "
                "compress the unified law into a smaller causal core with fewer moving parts."
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["stage5_headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["project_headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
