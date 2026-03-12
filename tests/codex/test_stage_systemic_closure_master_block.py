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
    ap = argparse.ArgumentParser(description="Stage systemic closure master block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_systemic_closure_master_block_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inv = load("theory_track_systemic_multiaxis_inventory_expansion_20260312.json")
    prune = load("theory_track_systemic_inventory_master_pruning_20260312.json")
    p2 = load("stage_p2_stress_coupled_update_pruned_search_20260312.json")
    bline = load("stage_b_path_conditioned_bridge_filtered_search_20260312.json")
    p4 = load("stage_p4_causal_falsification_bundle_20260312.json")
    v3 = load("theory_track_10round_excavation_loop_v3_20260312.json")

    scores = v3["ending_point"]["final_scores"]
    systemic_score = (
        scores["protocol_calling"]
        + scores["relation_chain"]
        + scores["successor_coherence"]
        + scores["brain_side_causal_closure"]
        + scores["theorem_pruning_strength"]
    ) / 5.0

    priority_block = [
        {
            "priority": 1,
            "block": "cross_model_real_long_chain_trace_capture",
            "why": "继续直打 successor 和 protocol 共同瓶颈",
            "targets": ["successor_coherence", "protocol_calling"],
        },
        {
            "priority": 2,
            "block": "protocol_bridge_transport_intervention",
            "why": "让 successor/readout 真正穿过 protocol bridge",
            "targets": ["object_to_readout_compatibility", "protocol_calling"],
        },
        {
            "priority": 3,
            "block": "P4_online_brain_causal_execution",
            "why": "把 brain-side causal closure 从状态机推进到在线执行",
            "targets": ["brain_side_causal_closure"],
        },
        {
            "priority": 4,
            "block": "stress_bridge_strict_survival",
            "why": "把 P2/B-line 的 theorem 拉进真正 strict frontier",
            "targets": ["stress_guarded_update_theorem", "anchored_bridge_lift_theorem"],
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "Stage_systemic_closure_master_block",
        },
        "systemic_sources": {
            "inventory_num_concepts": inv["headline_metrics"]["num_concepts"],
            "preserved_theorem_count": len(prune["master_pruning"]["preserved_theorems"]),
            "p2_kept_pillar_count": p2["bridge_to_engineering"]["kept_pillar_count"],
            "bline_family_count": bline["bridge_to_engineering"]["family_count"],
            "p4_falsification_block_count": len(p4["falsification_blocks"]),
        },
        "headline_metrics": {
            "systemic_closure_score": float(systemic_score),
            "protocol_calling": float(scores["protocol_calling"]),
            "relation_chain": float(scores["relation_chain"]),
            "successor_coherence": float(scores["successor_coherence"]),
            "brain_side_causal_closure": float(scores["brain_side_causal_closure"]),
            "theorem_pruning_strength": float(scores["theorem_pruning_strength"]),
        },
        "priority_block": priority_block,
        "verdict": {
            "core_answer": (
                "Systemic closure now has a unified block plan: one inventory, one master pruning layer, one engineering closure plan."
            ),
            "next_engineering_target": (
                "Execute the next large block around successor+protocol, then push brain-side causal execution and strict survival."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
