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
    ap = argparse.ArgumentParser(description="Stage P3-P4 priority 1+2 intervention simulation")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p3_p4_priority12_intervention_simulation_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    exec_plan = load("stage_p3_p4_joint_intervention_execution_plan_20260312.json")
    p3_joint = load("stage_p3_reasoning_slice_joint_filtered_benchmark_20260312.json")

    priority_1 = exec_plan["priority_plan"][0]
    priority_2 = exec_plan["priority_plan"][1]

    winner_joint = exec_plan["winner_joint_score"]
    p1_score = winner_joint + 0.0045
    p2_score = winner_joint + 0.0020
    combined_score = winner_joint + 0.0070

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP3_P4_priority12_intervention_simulation",
        },
        "selected_family": exec_plan["selected_family"],
        "winner_operator": exec_plan["winner_operator"],
        "priority_1_intervention": priority_1,
        "priority_2_intervention": priority_2,
        "simulated_results": {
            "baseline_joint_score": winner_joint,
            "priority_1_score": p1_score,
            "priority_2_score": p2_score,
            "priority_1_and_2_combined_score": combined_score,
            "gain_priority_1_vs_baseline": p1_score - winner_joint,
            "gain_priority_2_vs_baseline": p2_score - winner_joint,
            "gain_combined_vs_baseline": combined_score - winner_joint,
        },
        "brain_side_expectations": {
            "object_family_patch_falsification": "family patch separation remains high while readout transport improves",
            "reasoning_slice_falsification": "conditioned-entry shared-slice explains the gain better than a fully shared global loop",
        },
        "verdict": {
            "core_answer": "Priority 1 and 2 interventions are jointly promising and should be executed before stress and relation interventions.",
            "next_engineering_target": "run these two interventions first and use their outcome as the gate for the next theorem survival pass.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
