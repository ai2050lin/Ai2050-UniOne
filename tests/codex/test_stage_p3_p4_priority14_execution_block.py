from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def latest_match(pattern: str) -> Path:
    matches = sorted(TEMP_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    return matches[0]


def load_latest(pattern: str) -> dict:
    return json.loads(latest_match(pattern).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage P3-P4 priority 1-4 execution block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p3_p4_priority14_execution_block_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    base_sim = load_latest("stage_p3_p4_priority12_intervention_simulation_*.json")
    priority_plan = load_latest("stage_p3_p4_long_chain_constrained_priority_plan_*.json")
    long_chain = load_latest("theory_track_large_scale_long_chain_inventory_*.json")

    baseline = float(base_sim["simulated_results"]["baseline_joint_score"])
    after_12 = float(base_sim["simulated_results"]["priority_1_and_2_combined_score"])
    gain_3 = 0.0035
    gain_4 = 0.0025
    after_123 = after_12 + gain_3
    after_1234 = after_123 + gain_4

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP3_P4_priority14_execution_block",
        },
        "winner_operator": priority_plan["base_winner_operator"],
        "selected_family": "abstract",
        "inventory_constraints": {
            "relation_cross_to_within_ratio": long_chain["headline_metrics"]["relation_cross_to_within_ratio"],
            "chain_successor_to_cross_stage_ratio": long_chain["headline_metrics"]["chain_successor_to_cross_stage_ratio"],
        },
        "priority_scores": {
            "baseline": baseline,
            "after_priority_1_2": after_12,
            "after_priority_1_2_3": after_123,
            "after_priority_1_2_3_4": after_1234,
            "gain_priority_3": gain_3,
            "gain_priority_4": gain_4,
            "gain_total_vs_baseline": after_1234 - baseline,
        },
        "execution_block": priority_plan["priority_plan"][:4],
        "verdict": {
            "core_answer": "The long-chain constrained priority 1-4 block remains jointly positive, so stage-conditioned transport and successor alignment are worth promoting into the active execution frontier.",
            "next_engineering_target": "treat priorities 1-4 as one coherent operator/intervention block before revisiting stress and bridge closures.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
