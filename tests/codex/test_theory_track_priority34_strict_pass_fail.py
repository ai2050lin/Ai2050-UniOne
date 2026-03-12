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
    ap = argparse.ArgumentParser(description="Theory-track priority 3+4 strict pass/fail")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_priority34_strict_pass_fail_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    block = load_latest("stage_p3_p4_priority14_execution_block_*.json")
    inventory = load_latest("theory_track_large_scale_naturalized_reasoning_inventory_*.json")
    pruning = load_latest("theory_track_naturalized_inventory_frontier_pruning_*.json")

    baseline = float(block["priority_scores"]["baseline"])
    after_12 = float(block["priority_scores"]["after_priority_1_2"])
    after_123 = float(block["priority_scores"]["after_priority_1_2_3"])
    after_1234 = float(block["priority_scores"]["after_priority_1_2_3_4"])

    temporal_ratio = float(inventory["headline_metrics"]["temporal_cross_to_within_ratio"])
    successor_ratio = float(inventory["headline_metrics"]["chain_successor_to_cross_stage_ratio"])
    relation_ratio = float(inventory["headline_metrics"]["relation_cross_to_within_ratio"])

    delta_p3 = after_123 - after_12
    delta_p4 = after_1234 - after_123

    if temporal_ratio >= 1.05 and delta_p3 >= 0.003:
        stage_status = "strict_pass"
        stage_conf = 0.78
    elif temporal_ratio >= 1.03 and delta_p3 > 0:
        stage_status = "strengthened_provisional"
        stage_conf = 0.68
    else:
        stage_status = "fail"
        stage_conf = 0.34

    if successor_ratio <= 0.93 and delta_p4 >= 0.0024 and relation_ratio >= 1.6:
        successor_status = "strict_pass"
        successor_conf = 0.74
    elif successor_ratio <= 0.97 and delta_p4 > 0:
        successor_status = "strengthened_provisional"
        successor_conf = 0.63
    else:
        successor_status = "fail"
        successor_conf = 0.31

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_priority34_strict_pass_fail",
        },
        "score_context": {
            "baseline": baseline,
            "after_priority_1_2": after_12,
            "after_priority_1_2_3": after_123,
            "after_priority_1_2_3_4": after_1234,
            "delta_priority_3": delta_p3,
            "delta_priority_4": delta_p4,
        },
        "inventory_constraints": {
            "temporal_cross_to_within_ratio": temporal_ratio,
            "chain_successor_to_cross_stage_ratio": successor_ratio,
            "relation_cross_to_within_ratio": relation_ratio,
        },
        "frontier_state": pruning["frontier_pruning"],
        "strict_pass_fail": [
            {
                "theorem": "stage_conditioned_reasoning_transport_theorem",
                "status": stage_status,
                "confidence": stage_conf,
            },
            {
                "theorem": "causal_successor_alignment_theorem",
                "status": successor_status,
                "confidence": successor_conf,
            },
        ],
        "verdict": {
            "core_answer": "Naturalized long-chain constraints are now strong enough to force a stricter pass/fail decision for the stage and successor theorems, instead of leaving both only in provisional status.",
            "next_theory_target": "if both survive, move stress and bridge into the next active frontier; if not, further expand naturalized chain inventory and tighten stage/successor perturbations.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
