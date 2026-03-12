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
    ap = argparse.ArgumentParser(description="Stage P3-P4 joint intervention execution plan")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p3_p4_joint_intervention_execution_plan_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    design = load("stage_p3_p4_joint_intervention_design_20260312.json")
    p3_joint = load("stage_p3_reasoning_slice_joint_filtered_benchmark_20260312.json")

    priority_plan = [
        {
            "priority": 1,
            "name": "scaffolded_readout_vs_baseline_intervention",
            "expected_joint_score": p3_joint["scores"]["joint_reasoning_filtered_score"],
            "expected_brain_side_signal": "preserve family patch separation while improving readout transport",
            "why_now": "directly targets the highest-severity gap: object_to_readout_compatibility",
        },
        {
            "priority": 2,
            "name": "reasoning_slice_transport_intervention",
            "expected_joint_score": p3_joint["scores"]["joint_reasoning_filtered_score"] - 0.004,
            "expected_brain_side_signal": "shared reasoning slice should help without reverting to a global central loop",
            "why_now": "ties unified reasoning to the current best P3 operator",
        },
        {
            "priority": 3,
            "name": "stress_guard_intervention",
            "expected_joint_score": p3_joint["scores"]["joint_reasoning_filtered_score"] - 0.007,
            "expected_brain_side_signal": "write narrows before read collapse under stress",
            "why_now": "pushes P2/P3/P4 toward stress-bound dynamic closure",
        },
        {
            "priority": 4,
            "name": "anchored_relation_lift_intervention",
            "expected_joint_score": p3_joint["scores"]["joint_reasoning_filtered_score"] - 0.009,
            "expected_brain_side_signal": "relation lift stays anchored to family/object structure",
            "why_now": "bridges B-line into the same intervention framework",
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP3_P4_joint_intervention_execution_plan",
        },
        "selected_family": design["selected_family"],
        "winner_operator": design["winner_operator"],
        "winner_joint_score": design["winner_joint_score"],
        "priority_plan": priority_plan,
        "verdict": {
            "core_answer": "The P3-P4 intervention set now has an execution order, expected score regime, and expected brain-side signal for each remaining encoding gap.",
            "next_engineering_target": "run priority 1 and 2 first, then widen to stress and bridge interventions after the first joint readout results come back.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
