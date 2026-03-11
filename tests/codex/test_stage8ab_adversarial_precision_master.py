#!/usr/bin/env python
"""
Build a master view over stage 8A and 8B and fold them into the full project.
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
    ap = argparse.ArgumentParser(description="Stage 8AB adversarial precision master")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage8ab_adversarial_precision_master_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage7d = load_json(ROOT / "tests" / "codex_temp" / "stage7d_coding_law_verdict_master_20260311.json")
    stage8a = load_json(ROOT / "tests" / "codex_temp" / "stage8a_adversarial_counterexample_search_20260311.json")
    stage8b = load_json(ROOT / "tests" / "codex_temp" / "stage8b_high_resolution_precision_editing_20260311.json")

    stage8_blocks = {
        "stage8a_adversarial_counterexample_search": {
            "score": float(stage8a["headline_metrics"]["overall_stage8a_score"]),
            "status": "completed"
            if bool(stage8a["hypotheses"]["H5_stage8a_adversarial_counterexample_map_is_established"])
            else "in_progress",
        },
        "stage8b_high_resolution_precision_editing": {
            "score": float(stage8b["headline_metrics"]["overall_stage8b_score"]),
            "status": "completed"
            if bool(stage8b["hypotheses"]["H5_stage8b_high_resolution_precision_editing_is_moderately_closed"])
            else "in_progress",
        },
    }

    stage8_score = sum(block["score"] for block in stage8_blocks.values()) / 2.0
    completed_stage8 = sum(1 for block in stage8_blocks.values() if block["status"] == "completed")

    project_blocks = dict(stage7d["project_blocks"])
    project_blocks["stage8_adversarial_precision_phase"] = {
        "score": float(stage8_score),
        "status": "completed" if completed_stage8 == 2 else "in_progress",
    }
    overall_progress = sum(block["score"] for block in project_blocks.values()) / float(len(project_blocks))
    completed_project = sum(1 for block in project_blocks.values() if block["status"] == "completed")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage8ab_adversarial_precision_master",
        },
        "stage8_blocks": stage8_blocks,
        "stage8_headline_metrics": {
            "overall_stage8_score": float(stage8_score),
            "completed_stage8_block_count": int(completed_stage8),
            "total_stage8_block_count": 2,
        },
        "project_blocks": project_blocks,
        "project_headline_metrics": {
            "overall_project_progress_score": float(overall_progress),
            "completed_project_block_count": int(completed_project),
            "total_project_block_count": len(project_blocks),
        },
        "hypotheses": {
            "H1_stage8_has_a_real_counterexample_map": bool(
                stage8a["headline_metrics"]["overall_stage8a_score"] >= 0.74
            ),
            "H2_stage8_has_a_narrower_precision_policy": bool(
                stage8b["headline_metrics"]["overall_stage8b_score"] >= 0.74
            ),
            "H3_stage8_still_has_unresolved_residual_risk": bool(
                stage8b["headline_metrics"]["residual_risk_score"] >= 0.70
            ),
            "H4_stage8_is_completed_so_far": bool(completed_stage8 == 2),
        },
        "project_readout": {
            "summary": (
                "This master view is positive only if stage 8 has already done two things: mapped where the current "
                "coding-law candidate is vulnerable, and compressed current editing results into a narrower precision policy."
            ),
            "next_question": (
                "If this stage holds, the remaining stage-8 blocks should move from local attacks to invariance tests "
                "across more models, tasks, and sharper brain-side falsification targets."
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
