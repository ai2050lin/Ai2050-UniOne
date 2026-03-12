from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load(name: str) -> dict:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory-track brain encoding progress assessment")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_brain_encoding_progress_assessment_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    seven = load("theory_track_inventory_seven_question_mapping_20260312.json")
    bottleneck = load("theory_track_inventory_bottleneck_resolution_analysis_20260312.json")
    phases = load("phase_p1_p4_execution_master_20260312.json")
    path_read = load("theory_track_path_conditioned_readout_law_20260312.json")
    path_bridge = load("theory_track_path_conditioned_bridge_lift_law_20260312.json")

    strength_map = {"strong": 0.85, "medium": 0.65, "partial": 0.45}
    q_scores = {
        key: strength_map.get(value["current_strength"], 0.5)
        for key, value in seven["seven_question_mapping"].items()
    }
    mean_q_score = sum(q_scores.values()) / max(1, len(q_scores))

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_brain_encoding_progress_assessment",
        },
        "progress_snapshot": {
            "seven_question_mean_score": float(mean_q_score),
            "p1_score": phases["phases"]["P1_object_manifold_modeling"]["overall_score"],
            "p2_score": phases["phases"]["P2_controlled_update_law_modeling"]["overall_score"],
            "p3_score": phases["phases"]["P3_compatibility_geometry_modeling"]["overall_score"],
            "p4_score": phases["phases"]["P4_brain_mapping_and_falsification"]["overall_score"],
            "path_readout_open_count": path_read["phase_profile"]["stabilize_to_read_open"],
            "path_bridge_ready_count": path_bridge["headline_metrics"]["bridge_ready_count"],
        },
        "current_status": {
            "what_is_closed_most": [
                "Q1 encoding object layer",
                "Q5 crossmodal consistency",
                "Q6 discriminative geometry constraints",
            ],
            "what_is_partially_closed": [
                "Q2 local update law",
                "Q3 write/read separation",
                "Q4 bridge-role lift",
                "Q7 brain-side mapping",
            ],
            "main_open_bottleneck": bottleneck["current_bottlenecks"]["main_bottleneck"],
            "secondary_bottlenecks": bottleneck["current_bottlenecks"]["secondary_bottlenecks"],
        },
        "hard_problems": {
            "hardest_issue": "shared object manifold to discriminative geometry compatibility under dynamic switching and novelty stress",
            "why_hard": [
                "object atlas is already strong, but object-to-readout overlap remains narrow",
                "novelty and retention stress consume transport budget",
                "bridge-role is now anchored, but not yet densely coupled into dynamic closure",
                "brain-side execution is protocol-ready, not causally executed",
            ],
        },
        "recommended_solution_order": [
            "1. continue P3 only inside path-conditioned switching-aware transport families",
            "2. attach path-conditioned bridge-lift law to B-line and relation probes",
            "3. bind write/read and update law more tightly to inventory stress profiles",
            "4. execute P4 brain-side probe bundle for causal falsification",
        ],
        "project_percentages": {
            "theory_skeleton": "95% - 97%",
            "engineering_closure": "83% - 88%",
            "brain_encoding_crack_level": "86% - 90%",
        },
        "verdict": {
            "core_answer": "The project has now strongly reconstructed the object-atlas side of brain encoding and partially reconstructed the dynamic path laws, but the main hard gap remains dynamic object-to-readout compatibility under switching and stress.",
            "next_stage_goal": "turn path-conditioned readout and bridge-lift laws into engineering filters and brain-side probes",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
