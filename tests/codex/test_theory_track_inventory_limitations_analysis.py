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
    ap = argparse.ArgumentParser(description="Theory-track inventory limitations analysis")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_inventory_limitations_analysis_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inv_map = load("theory_track_inventory_seven_question_mapping_20260312.json")
    bottleneck = load("theory_track_inventory_bottleneck_resolution_analysis_20260312.json")
    inv_sys = load("theory_track_inventory_unified_system_formalization_20260312.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_inventory_limitations_analysis",
        },
        "what_inventory_solves_well": {
            "object_layer": inv_map["seven_question_mapping"]["Q1_encoding_object_layer"]["current_strength"],
            "crossmodal_layer": inv_map["seven_question_mapping"]["Q5_crossmodal_consistency"]["current_strength"],
            "readout_constraint_layer": inv_map["seven_question_mapping"]["Q6_discriminative_geometry"]["current_strength"],
            "summary": "inventory is strong at structural encoding questions and at excluding obviously wrong geometry/readout candidates",
        },
        "core_limitations": {
            "dynamics_limitation": {
                "severity": "high",
                "statement": "inventory stores and parameterizes structure, but it does not by itself fully determine long-horizon dynamics",
            },
            "update_law_limitation": {
                "severity": "high",
                "statement": "inventory constrains what updates must preserve, but not yet the complete lawful update operator under novelty pressure",
            },
            "bridge_role_limitation": {
                "severity": "medium",
                "statement": "inventory can anchor bridge-role lift, but it does not yet densely generate full relational mechanics",
            },
            "brain_projection_limitation": {
                "severity": "high",
                "statement": "inventory can define probe families, but not yet complete 3D causal brain mapping",
            },
            "phase_switch_limitation": {
                "severity": "medium",
                "statement": "inventory conditions overlaps and charts, but not yet the full transition law between operational phases",
            },
        },
        "why_limitations_exist": {
            "main_reason": "inventory is an atlas-conditioned object; some open problems require operators, trajectories, and causal execution beyond atlas structure",
            "examples": [
                "A(I) still needs explicit dynamic update operators",
                "M_feas(I) still needs trajectory-level closure under switching and readout transport",
                "brain-side projection needs empirical execution, not only abstract probe construction",
            ],
        },
        "implication_for_theory_track": {
            "core_statement": "Inventory should be treated as the central conditioning object, not as the complete theory by itself.",
            "safe_use": [
                "use inventory to reconstruct object structure",
                "use inventory to parameterize A and M_feas",
                "use inventory to exclude wrong candidate families",
            ],
            "unsafe_use": [
                "do not assume inventory alone closes update law",
                "do not assume inventory alone closes bridge-role dynamics",
                "do not assume inventory alone closes brain-side causality",
            ],
        },
        "current_main_open_point": bottleneck["current_bottlenecks"]["main_bottleneck"],
        "unified_system_anchor": inv_sys["unified_object"]["high_level_form"],
        "verdict": {
            "core_answer": "Inventory is now the best central theoretical object, but it must remain coupled to operators, viability, switching, and brain-side execution. On its own, it is not the whole solution.",
            "next_theory_target": "build explicit operator families and transition laws on top of the inventory rather than replacing them with inventory alone",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
