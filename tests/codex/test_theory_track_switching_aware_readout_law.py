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
    ap = argparse.ArgumentParser(description="Theory-track switching-aware readout law")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_switching_aware_readout_law_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    family_transport = load("theory_track_family_level_transport_operator_20260312.json")
    phase_transport = load("theory_track_phase_level_transport_operator_20260312.json")
    inv_limit = load("theory_track_inventory_limitations_analysis_20260312.json")

    family_rows = family_transport["family_transport_operator"]["formal_family"]
    mean_family_budget = float(sum(float(row["mean_transport_budget"]) for row in family_rows.values()) / len(family_rows))
    phase_ops = phase_transport["phase_transport_operator"]["operators"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_switching_aware_readout_law",
        },
        "switching_aware_readout_law": {
            "core_statement": "Readout succeeds only when family transport budget and phase transport condition are jointly satisfied.",
            "formal_form": (
                "Tau_readout(c, mode_1 -> mode_2) = Tau_read^(f_c) + Phi(mode_1 -> mode_2) - switch_cost(c, mode_1, mode_2)"
            ),
            "expanded_form": {
                "stabilize_to_read": f"Tau_readout = Tau_read^(f_c) + ({phase_ops['stabilize_to_read']['formal_form'].split('= ')[1]}) - switch_cost",
                "novelty_to_read": f"Tau_readout = Tau_read^(f_c) + ({phase_ops['novelty_to_read']['formal_form'].split('= ')[1]}) - switch_cost",
                "relation_to_read": f"Tau_readout = Tau_read^(f_c) + ({phase_ops['relation_to_read']['formal_form'].split('= ')[1]}) - switch_cost",
            },
        },
        "interpretation": {
            "family_term": "Tau_read^(f_c) captures family-patch transport capacity",
            "phase_term": "Phi(mode_1 -> mode_2) captures whether the current operational phase opens or narrows readout transport",
            "switch_cost_term": "switch_cost penalizes unstable or abrupt transport under mode transitions",
            "overall_mean_family_budget": mean_family_budget,
        },
        "closure_effect": {
            "what_it_adds": [
                "converts static overlap reasoning into switching-aware transport",
                "ties readout success to both atlas patch structure and phase dynamics",
                "gives engineering P3 a direct pruning criterion",
            ],
            "what_it_still_needs": [
                "explicit empirical switch_cost estimator",
                "family-to-family transition operator",
                "brain-side validation of phase-sensitive readout transport",
            ],
        },
        "open_limitations": {
            "phase_switch_limitation": inv_limit["core_limitations"]["phase_switch_limitation"],
        },
        "verdict": {
            "core_answer": "The readout bottleneck can now be rewritten as a switching-aware transport law rather than a vague compatibility issue.",
            "next_theory_target": "use this law as the sole acceptance criterion for the next P3 candidate family search",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
