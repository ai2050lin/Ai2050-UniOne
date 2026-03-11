#!/usr/bin/env python
"""
Build a partial stage-9 master view over mechanism-break tests and residual decomposition.
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
    ap = argparse.ArgumentParser(description="Stage 9AC mechanism residual master")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage9ac_mechanism_residual_master_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage8_master = load_json(ROOT / "tests" / "codex_temp" / "stage8_master_closure_20260311.json")
    stage9a = load_json(ROOT / "tests" / "codex_temp" / "stage9a_mechanism_adversarial_break_test_20260311.json")
    stage9c = load_json(
        ROOT / "tests" / "codex_temp" / "stage9c_unified_law_residual_decomposition_20260311.json"
    )

    stage9_blocks = {
        "stage9a_mechanism_adversarial_break_test": {
            "score": float(stage9a["headline_metrics"]["overall_stage9a_score"]),
            "status": "completed"
            if bool(stage9a["hypotheses"]["H5_stage9a_mechanism_break_test_is_moderately_supported"])
            else "in_progress",
        },
        "stage9b_strong_precision_closure": {
            "score": 0.0,
            "status": "pending",
        },
        "stage9c_unified_law_residual_decomposition": {
            "score": float(stage9c["headline_metrics"]["overall_stage9c_score"]),
            "status": "completed"
            if bool(stage9c["hypotheses"]["H5_stage9c_residual_decomposition_is_moderately_supported"])
            else "in_progress",
        },
        "stage9d_forward_brain_prediction": {
            "score": 0.0,
            "status": "pending",
        },
    }

    active_scores = [
        block["score"] for block in stage9_blocks.values() if block["status"] != "pending"
    ]
    stage9_partial_score = sum(active_scores) / max(1, len(active_scores))
    completed_stage9 = sum(1 for block in stage9_blocks.values() if block["status"] == "completed")

    project_blocks = dict(stage8_master["project_blocks"])
    project_blocks["stage9_mechanism_residual_phase"] = {
        "score": float(stage9_partial_score),
        "status": "in_progress" if completed_stage9 < 4 else "completed",
    }
    overall_progress = sum(block["score"] for block in project_blocks.values()) / float(len(project_blocks))
    completed_project = sum(1 for block in project_blocks.values() if block["status"] == "completed")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage9ac_mechanism_residual_master",
        },
        "stage9_blocks": stage9_blocks,
        "stage9_headline_metrics": {
            "current_stage9_score": float(stage9_partial_score),
            "completed_stage9_block_count": int(completed_stage9),
            "total_stage9_block_count": 4,
        },
        "project_blocks": project_blocks,
        "project_headline_metrics": {
            "overall_project_progress_score": float(overall_progress),
            "completed_project_block_count": int(completed_project),
            "total_project_block_count": len(project_blocks),
        },
        "hypotheses": {
            "H1_stage9a_is_established": bool(
                stage9_blocks["stage9a_mechanism_adversarial_break_test"]["status"] == "completed"
            ),
            "H2_stage9c_is_established": bool(
                stage9_blocks["stage9c_unified_law_residual_decomposition"]["status"] == "completed"
            ),
            "H3_stage9_is_not_complete_yet": bool(completed_stage9 < 4),
        },
        "project_readout": {
            "summary": (
                "This partial stage-9 master view is positive only if mechanism-break tests and residual decomposition "
                "are already established, even though strong precision closure and forward brain prediction remain pending."
            ),
            "next_question": (
                "If this stage holds, the next move should target the dominant residual source and push precision "
                "editing toward a stronger closure regime."
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["stage9_headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["project_headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
