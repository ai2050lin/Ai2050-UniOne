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
    ap = argparse.ArgumentParser(description="Theory-track long-chain survival criteria")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_long_chain_survival_criteria_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    long_chain = load_latest("theory_track_large_scale_long_chain_inventory_*.json")
    survival = load_latest("theory_track_icspb_theorem_survival_report_*.json")
    extended = load_latest("theory_track_long_chain_extended_theorem_set_*.json")

    criteria = [
        {
            "theorem": "family_section_theorem",
            "status": "ready_now",
            "pass_condition": "family patch separation remains above current large-inventory floor under scaffolded readout intervention",
            "fail_condition": "cross-family mixing rises to same-family band under scaffolded readout",
        },
        {
            "theorem": "restricted_readout_transport_theorem",
            "status": "ready_now",
            "pass_condition": "reasoning-slice transport continues to beat readout-only contrast while preserving restricted overlap behavior",
            "fail_condition": "reasoning transport gain disappears or only survives under direct collapse",
        },
        {
            "theorem": "stage_conditioned_reasoning_transport_theorem",
            "status": "next_after_priority_1_2",
            "pass_condition": "stage-conditioned transport exceeds stage-free readout under long-chain constraints",
            "fail_condition": "stage labels do not improve transport once family and reasoning slice are controlled",
        },
        {
            "theorem": "causal_successor_alignment_theorem",
            "status": "next_after_priority_1_2",
            "pass_condition": "successor-aware transport preserves chain-local successor coherence better than chain-agnostic transport",
            "fail_condition": "successor-aware transport yields no coherence gain over chain-agnostic transport",
        },
        {
            "theorem": "stress_guarded_update_theorem",
            "status": "queued_later",
            "pass_condition": "write narrows before read collapse under stress while stage-aware admissibility remains valid",
            "fail_condition": "stress directly destroys readout without guarded-write asymmetry",
        },
        {
            "theorem": "anchored_bridge_lift_theorem",
            "status": "queued_later",
            "pass_condition": "relation lift remains family-anchored while causal successor structure is preserved",
            "fail_condition": "relation lift only survives after losing family anchoring or successor coherence",
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_long_chain_survival_criteria",
        },
        "inventory_constraints": {
            "family_cross_to_within_ratio": long_chain["headline_metrics"]["family_cross_to_within_ratio"],
            "relation_cross_to_within_ratio": long_chain["headline_metrics"]["relation_cross_to_within_ratio"],
            "chain_successor_to_cross_stage_ratio": long_chain["headline_metrics"]["chain_successor_to_cross_stage_ratio"],
            "ready_for_immediate_survival_test_count": survival["ready_for_immediate_survival_test_count"],
            "extended_theorem_count": len(extended["legacy_theorems"]) + len(extended["new_theorems"]),
        },
        "survival_criteria": criteria,
        "verdict": {
            "core_answer": "Long-chain inventory now supports an explicit six-theorem survival framework, with two ready now, two next under stage/chain-aware interventions, and two queued behind later dynamic closures.",
            "next_theory_target": "promote stage-conditioned transport and successor alignment into the next intervention priority block.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
