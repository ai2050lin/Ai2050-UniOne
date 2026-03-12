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
    ap = argparse.ArgumentParser(description="Theory-track path-conditioned readout law")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_path_conditioned_readout_law_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    path_law = load("theory_track_path_conditioned_encoding_law_20260312.json")
    switching = load("theory_track_switching_aware_readout_law_20260312.json")
    concept_transport = load("theory_track_inventory_stress_to_readout_transport_coupling_20260312.json")
    bottleneck = load("theory_track_inventory_bottleneck_resolution_analysis_20260312.json")

    profiles = concept_transport["concept_transport_profiles"]
    open_read = 0
    narrow_read = 0
    conditional_read = 0

    for row in path_law["concept_path_entries"].values():
        phase_status = row["phase_status"]
        if phase_status["stabilize_to_read"] == "open":
            open_read += 1
        if phase_status["novelty_to_read"] == "narrow":
            narrow_read += 1
        if phase_status["relation_to_read"] == "conditional":
            conditional_read += 1

    mean_transport_budget = float(
        sum(float(v["transport_budget"]) for v in profiles.values()) / max(1, len(profiles))
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_path_conditioned_readout_law",
        },
        "path_conditioned_readout_law": {
            "core_statement": "Readout is a path-conditioned operation that opens only when a concept remains inside admissible, viable, and switching-aware transport bands.",
            "formal_form": "Read(c, mode_1 -> mode_2) = Q(c) only if Pi_path(c, mode_1 -> mode_2) = 1",
            "expanded_form": "Pi_path = 1[Delta in A(I)] * 1[trajectory subset of M_feas(I)] * 1[Tau_readout(c, mode_1 -> mode_2) > 0]",
            "switching_support": switching["switching_aware_readout_law"]["formal_form"],
        },
        "phase_profile": {
            "stabilize_to_read_open": int(open_read),
            "novelty_to_read_narrow": int(narrow_read),
            "relation_to_read_conditional": int(conditional_read),
            "mean_transport_budget": mean_transport_budget,
        },
        "mathematical_meaning": {
            "core_answer": "Readout cannot be treated as a static head over object geometry. It is a constrained path query over the atlas, gated by admissibility, viability, and switching-aware transport.",
            "why_it_helps": [
                "turns Q6 from a static geometry problem into a transport-query problem",
                "explains why direct object-to-readout collapse keeps failing",
                "gives engineering a sharper acceptance criterion for P3",
            ],
            "main_open_gap": bottleneck["current_bottlenecks"]["main_bottleneck"],
        },
        "verdict": {
            "core_answer": "The readout law is now strong enough to be stated as path-conditioned rather than merely overlap-conditioned.",
            "next_theory_target": "extend the same path-conditioned formalism to relation and bridge lifts",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
