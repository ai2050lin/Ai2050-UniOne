#!/usr/bin/env python
"""
Build the full stage-8 master view over adversarial search, precision editing,
cross-model invariants, and brain-side high-risk falsification.
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
    ap = argparse.ArgumentParser(description="Stage 8 master closure")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage8_master_closure_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage8ab = load_json(ROOT / "tests" / "codex_temp" / "stage8ab_adversarial_precision_master_20260311.json")
    stage8c = load_json(ROOT / "tests" / "codex_temp" / "stage8c_cross_model_task_invariants_20260311.json")
    stage8d = load_json(ROOT / "tests" / "codex_temp" / "stage8d_brain_high_risk_falsification_20260311.json")

    stage8_blocks = {
        "stage8a_adversarial_counterexample_search": stage8ab["stage8_blocks"][
            "stage8a_adversarial_counterexample_search"
        ],
        "stage8b_high_resolution_precision_editing": stage8ab["stage8_blocks"][
            "stage8b_high_resolution_precision_editing"
        ],
        "stage8c_cross_model_task_invariants": {
            "score": float(stage8c["headline_metrics"]["overall_stage8c_score"]),
            "status": "completed"
            if bool(stage8c["hypotheses"]["H5_stage8c_cross_model_task_invariants_are_moderately_supported"])
            else "in_progress",
        },
        "stage8d_brain_high_risk_falsification": {
            "score": float(stage8d["headline_metrics"]["overall_stage8d_score"]),
            "status": "completed"
            if bool(stage8d["hypotheses"]["H5_stage8d_brain_high_risk_falsification_is_moderately_supported"])
            else "in_progress",
        },
    }

    stage8_score = sum(block["score"] for block in stage8_blocks.values()) / 4.0
    completed_stage8 = sum(1 for block in stage8_blocks.values() if block["status"] == "completed")

    project_blocks = dict(stage8ab["project_blocks"])
    project_blocks["stage8_adversarial_precision_phase"] = {
        "score": float(stage8_score),
        "status": "completed" if completed_stage8 == 4 else "in_progress",
    }
    overall_progress = sum(block["score"] for block in project_blocks.values()) / float(len(project_blocks))
    completed_project = sum(1 for block in project_blocks.values() if block["status"] == "completed")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage8_master_closure",
        },
        "stage8_blocks": stage8_blocks,
        "stage8_headline_metrics": {
            "overall_stage8_score": float(stage8_score),
            "completed_stage8_block_count": int(completed_stage8),
            "total_stage8_block_count": 4,
        },
        "project_blocks": project_blocks,
        "project_headline_metrics": {
            "overall_project_progress_score": float(overall_progress),
            "completed_project_block_count": int(completed_project),
            "total_project_block_count": len(project_blocks),
        },
        "hypotheses": {
            "H1_stage8_has_counterexample_map": bool(
                stage8_blocks["stage8a_adversarial_counterexample_search"]["status"] == "completed"
            ),
            "H2_stage8_has_narrow_precision_policy": bool(
                stage8_blocks["stage8b_high_resolution_precision_editing"]["status"] == "completed"
            ),
            "H3_stage8_has_cross_model_task_invariants": bool(
                stage8_blocks["stage8c_cross_model_task_invariants"]["status"] == "completed"
            ),
            "H4_stage8_has_brain_high_risk_falsifiers": bool(
                stage8_blocks["stage8d_brain_high_risk_falsification"]["status"] == "completed"
            ),
            "H5_stage8_is_complete": bool(completed_stage8 == 4),
        },
        "project_readout": {
            "summary": (
                "Stage 8 is complete only if the project can map counterexamples, compress precision edits, identify "
                "cross-model/task invariants, and specify sharp brain-side falsifiers."
            ),
            "next_question": (
                "If stage 8 is complete, the next block should stop aggregating support and directly try to break the "
                "candidate coding law with targeted adversarial mechanism tests."
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["stage8_headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["project_headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
