#!/usr/bin/env python
"""
Build a stage-7 master verdict over the explicit coding-law candidate, including
its practical tuning utility and brain-side falsifiable support.
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
    ap = argparse.ArgumentParser(description="Stage 7D coding-law verdict master")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage7d_coding_law_verdict_master_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage6_master = load_json(ROOT / "tests" / "codex_temp" / "stage6_master_closure_20260311.json")
    stage7a = load_json(ROOT / "tests" / "codex_temp" / "stage7a_explicit_coding_law_candidate_20260311.json")
    stage7b = load_json(
        ROOT / "tests" / "codex_temp" / "stage7b_precision_tuning_and_cross_model_prediction_20260311.json"
    )
    stage7c = load_json(ROOT / "tests" / "codex_temp" / "stage7c_brain_falsifiable_predictions_20260311.json")

    stage7_blocks = {
        "stage7a_explicit_candidate": {
            "score": float(stage7a["headline_metrics"]["overall_stage7a_score"]),
            "status": "completed"
            if bool(stage7a["hypotheses"]["H5_stage7a_explicit_coding_law_is_moderately_supported"])
            else "in_progress",
        },
        "stage7b_precision_tuning_and_prediction": {
            "score": float(stage7b["headline_metrics"]["overall_stage7b_score"]),
            "status": "completed"
            if bool(stage7b["hypotheses"]["H5_stage7b_precision_tuning_and_prediction_is_moderately_supported"])
            else "in_progress",
        },
        "stage7c_brain_falsifiable_predictions": {
            "score": float(stage7c["headline_metrics"]["overall_stage7c_score"]),
            "status": "completed"
            if bool(stage7c["hypotheses"]["H5_stage7c_brain_falsifiable_predictions_are_moderately_supported"])
            else "in_progress",
        },
    }

    stage7_score = sum(block["score"] for block in stage7_blocks.values()) / 3.0
    completed_stage7 = sum(1 for block in stage7_blocks.values() if block["status"] == "completed")

    verdict_pillars = {
        "explicitness_and_fit": float(stage7a["headline_metrics"]["overall_stage7a_score"]),
        "practical_prediction_and_tuning": float(stage7b["headline_metrics"]["overall_stage7b_score"]),
        "brain_falsifiability": float(stage7c["headline_metrics"]["overall_stage7c_score"]),
        "remaining_precision_gap": float(1.0 - stage7b["headline_metrics"]["precise_tuning_score"]),
    }
    verdict_support_score = (
        verdict_pillars["explicitness_and_fit"]
        + verdict_pillars["practical_prediction_and_tuning"]
        + verdict_pillars["brain_falsifiability"]
        + (1.0 - verdict_pillars["remaining_precision_gap"])
    ) / 4.0

    project_blocks = dict(stage6_master["project_blocks"])
    project_blocks["stage7_candidate_coding_law_identification"] = {
        "score": float(stage7_score),
        "status": "completed" if completed_stage7 == 3 else "in_progress",
    }
    overall_progress = sum(block["score"] for block in project_blocks.values()) / float(len(project_blocks))
    completed_project = sum(1 for block in project_blocks.values() if block["status"] == "completed")

    verdict = "supported_but_not_final"
    if verdict_support_score >= 0.80 and bool(
        stage7b["hypotheses"]["H1_explicit_law_guides_precise_local_editing"]
    ):
        verdict = "strong_candidate"

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage7d_coding_law_verdict_master",
        },
        "stage7_blocks": stage7_blocks,
        "stage7_headline_metrics": {
            "overall_stage7_score": float(stage7_score),
            "completed_stage7_block_count": int(completed_stage7),
            "total_stage7_block_count": 3,
        },
        "verdict": {
            "truth_status": verdict,
            "verdict_support_score": float(verdict_support_score),
            "best_current_guess": stage7a["candidate_coding_law"]["verbal_guess"],
            "practical_tuning_policy": stage7b["recommended_policy"],
            "brain_side_key_prediction": stage7c["observed_predictions"]["law_prediction"],
        },
        "verdict_pillars": verdict_pillars,
        "project_blocks": project_blocks,
        "project_headline_metrics": {
            "overall_project_progress_score": float(overall_progress),
            "completed_project_block_count": int(completed_project),
            "total_project_block_count": len(project_blocks),
        },
        "hypotheses": {
            "H1_explicit_candidate_has_multiaxis_support": bool(
                stage7a["headline_metrics"]["overall_stage7a_score"] >= 0.75
            ),
            "H2_candidate_has_real_predictive_and_tuning_value": bool(
                stage7b["headline_metrics"]["overall_stage7b_score"] >= 0.72
            ),
            "H3_candidate_survives_brain_side_falsifiable_tests": bool(
                stage7c["headline_metrics"]["overall_stage7c_score"] >= 0.76
            ),
            "H4_coding_law_is_fully_proven": False,
            "H5_stage7_verdict_supports_best_current_guess_status": bool(
                verdict_support_score >= 0.74 and stage7_score >= 0.76
            ),
        },
        "project_readout": {
            "summary": (
                "Stage 7 is complete only if the project can write down an explicit candidate coding law, show that it "
                "has practical predictive value for model tuning, and survive brain-side directional falsifiability tests."
            ),
            "next_question": (
                "If this stage holds, the next block should stop scoring candidates and start trying to break the law: "
                "direct adversarial tests, tighter precision editing, and higher-resolution brain-side predictions."
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["stage7_headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["project_headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
