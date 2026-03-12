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
    ap = argparse.ArgumentParser(description="Theory-track phase-level transport operator")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_phase_level_transport_operator_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    family_transport = load("theory_track_family_level_transport_operator_20260312.json")
    overlap = load("theory_track_restricted_overlap_maps_20260312.json")
    limitations = load("theory_track_inventory_limitations_analysis_20260312.json")

    family_rows = family_transport["family_transport_operator"]["formal_family"]
    mean_family_budget = float(sum(float(row["mean_transport_budget"]) for row in family_rows.values()) / len(family_rows))
    mean_phase_overlap = float(sum(float(row["memory_phase_overlap"]) for row in overlap["restricted_overlap_maps"].values()) / len(overlap["restricted_overlap_maps"]))

    phase_operators = {
        "stabilize_to_read": {
            "formal_form": f"Tau_phase^(stabilize->read) = {mean_family_budget:.4f} + 0.5*{mean_phase_overlap:.4f}",
            "meaning": "stabilized memory can be opened for readout when family transport and phase overlap are both sufficient",
            "status": "candidate_open",
        },
        "novelty_to_read": {
            "formal_form": f"Tau_phase^(novelty->read) = {mean_family_budget:.4f} - 0.5*novelty_load",
            "meaning": "novelty-heavy phase should narrow transport before direct readout",
            "status": "candidate_narrow",
        },
        "relation_to_read": {
            "formal_form": f"Tau_phase^(relation->read) = {mean_family_budget:.4f} + relation_support - switch_cost",
            "meaning": "relation-lifted states can read out only if the relation support exceeds the switching cost",
            "status": "candidate_conditional",
        },
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_phase_level_transport_operator",
        },
        "phase_transport_operator": {
            "core_statement": "Family transport operators can now be lifted into phase-sensitive transport operators.",
            "formal_form": "Tau_phase(mode_1 -> mode_2) = family_transport_budget +/- phase_overlap +/- switch_load",
            "operators": phase_operators,
        },
        "motivation": {
            "current_phase_limitation": limitations["core_limitations"]["phase_switch_limitation"],
            "why_needed": "The remaining dynamic gap is no longer concept geometry alone, but transport under phase switching.",
        },
        "mathematical_meaning": {
            "core_statement": "The bottleneck is not a missing static readout map but a phase-conditioned transport law.",
            "why_useful": "This is the missing bridge from family operator closure to a switching-aware dynamic system.",
        },
        "verdict": {
            "core_answer": "Phase-level transport operators can now be defined as the next layer above family transport operators.",
            "next_theory_target": "merge family and phase transport into a single switching-aware readout law for the next engineering P3 block",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
