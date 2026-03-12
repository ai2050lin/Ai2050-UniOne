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
    ap = argparse.ArgumentParser(description="Theory-track long-chain extended theorem set")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_long_chain_extended_theorem_set_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    pruning = load_latest("theory_track_long_chain_inventory_theorem_pruning_*.json")
    amfeas = load_latest("theory_track_long_chain_inventory_to_A_Mfeas_pruning_*.json")
    intervention = load_latest("theory_track_long_chain_inventory_to_intervention_pruning_*.json")

    legacy_theorems = [
        "family_section_theorem",
        "restricted_readout_transport_theorem",
        "stress_guarded_update_theorem",
        "anchored_bridge_lift_theorem",
    ]
    new_theorems = [
        {
            "theorem": "stage_conditioned_reasoning_transport_theorem",
            "core_statement": "Reasoning transport must be stage-conditioned once temporal stage structure remains visible under long-chain inventory.",
            "bound_intervention": "stage_conditioned_reasoning_transport_intervention",
            "depends_on": [
                "stage_conditioned_admissibility_gate",
                "temporal_transition_chart_family",
            ],
        },
        {
            "theorem": "causal_successor_alignment_theorem",
            "core_statement": "Valid reasoning transport should preserve local successor coherence better than chain-agnostic transport alternatives.",
            "bound_intervention": "causal_successor_alignment_intervention",
            "depends_on": [
                "successor-aligned transport",
                "restricted_overlap_bands",
            ],
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_long_chain_extended_theorem_set",
        },
        "legacy_theorems": legacy_theorems,
        "new_theorems": new_theorems,
        "source_constraints": {
            "preserved_theorems_from_pruning": pruning["preserved_theorems"],
            "preserved_A_families": amfeas["A_pruning"]["preserved_families"],
            "preserved_Mfeas_families": amfeas["Mfeas_pruning"]["preserved_families"],
            "preserved_interventions": intervention["preserved_interventions"],
        },
        "verdict": {
            "core_answer": "Long-chain inventory now justifies extending ICSPB from four legacy theorem candidates to a six-theorem set that explicitly includes stage-conditioned transport and successor alignment.",
            "next_theory_target": "turn the two new long-chain theorems into explicit intervention-level survival criteria.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
