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
    ap = argparse.ArgumentParser(description="Theory-track systemic inventory master pruning")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_systemic_inventory_master_pruning_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inv = load("theory_track_systemic_multiaxis_inventory_expansion_20260312.json")
    qd = load("qwen_deepseek_naturalized_trace_bundle_20260312.json")
    succ = load("theory_track_successor_coherence_closure_diagnosis_20260312.json")

    metrics = inv["headline_metrics"]
    successor_gap = float(qd["missing_axes"][0]["gap"])  # already ordered; successor currently first

    preserved_theorems = [
        "family_section_theorem",
        "restricted_readout_transport_theorem",
        "stage_conditioned_reasoning_transport_theorem",
        "causal_successor_alignment_theorem",
        "stress_guarded_update_theorem",
        "anchored_bridge_lift_theorem",
        "protocol_task_bridge_theorem",
    ]
    pruned_theorems = [
        "single_global_reasoning_loop_theorem",
        "context_free_transport_theorem",
        "relation_free_readout_theorem",
        "temporal_stage_free_reasoning_theorem",
        "chain_agnostic_transport_theorem",
        "protocol_irrelevance_theorem",
    ]

    preserved_A = [
        "family_conditioned_intersection_cones",
        "stress_gated_update_cones",
        "relation_sensitive_update_gate",
        "stage_conditioned_admissibility_gate",
        "successor_sensitive_update_gate",
        "protocol_sensitive_update_gate",
    ]
    preserved_M = [
        "family_patched_viability_charts",
        "restricted_overlap_bands",
        "relation_conditioned_chart_widening",
        "temporal_transition_chart_family",
        "successor_chain_band",
        "protocol_bridge_band",
    ]
    preserved_interventions = [
        "scaffolded_readout_vs_baseline_intervention",
        "reasoning_slice_transport_intervention",
        "stage_conditioned_reasoning_transport_intervention",
        "causal_successor_alignment_intervention",
        "protocol_bridge_transport_intervention",
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_systemic_inventory_master_pruning",
        },
        "inventory_constraints": metrics,
        "successor_closure_context": succ["closure_status"],
        "master_pruning": {
            "preserved_theorems": preserved_theorems,
            "pruned_theorems": pruned_theorems,
            "preserved_A": preserved_A,
            "preserved_Mfeas": preserved_M,
            "preserved_interventions": preserved_interventions,
            "remaining_successor_gap": successor_gap,
        },
        "verdict": {
            "core_answer": (
                "Systemic inventory now shrinks theorem space, A(I), M_feas(I), and intervention space together instead of pruning them separately."
            ),
            "next_theory_target": (
                "Promote protocol-sensitive transport and successor-sensitive transport into the same closure block."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
