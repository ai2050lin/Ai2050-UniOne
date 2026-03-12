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
    ap = argparse.ArgumentParser(description="Theory-track inventory stress to readout transport coupling")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_inventory_stress_to_readout_transport_coupling_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stress = load("theory_track_inventory_stress_profiling_20260312.json")
    overlap = load("theory_track_restricted_overlap_maps_20260312.json")
    inv_map = load("theory_track_inventory_seven_question_mapping_20260312.json")

    readout_transport = {}
    for concept, row in stress["stress_rows"].items():
        family = row["family"]
        object_disc_overlap = overlap["restricted_overlap_maps"][family]["object_disc_overlap"]
        novelty_pressure = float(row["novelty_pressure"])
        retention_risk = float(row["retention_risk"])
        relation_capacity = float(row["relation_lift_capacity"])

        transport_budget = float(max(0.0, object_disc_overlap - 0.5 * novelty_pressure - retention_risk))
        transport_quality = "open" if transport_budget > 0.10 else "narrow" if transport_budget > 0.05 else "fragile"
        readout_transport[concept] = {
            "family": family,
            "object_disc_overlap": object_disc_overlap,
            "novelty_pressure": novelty_pressure,
            "retention_risk": retention_risk,
            "relation_lift_capacity": relation_capacity,
            "transport_budget": transport_budget,
            "transport_quality": transport_quality,
        }

    open_count = sum(1 for row in readout_transport.values() if row["transport_quality"] == "open")
    narrow_count = sum(1 for row in readout_transport.values() if row["transport_quality"] == "narrow")
    fragile_count = sum(1 for row in readout_transport.values() if row["transport_quality"] == "fragile")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_inventory_stress_to_readout_transport_coupling",
        },
        "transport_law": {
            "core_statement": "Readout transport should be modeled as a stress-gated budget defined on top of restricted object-disc overlaps.",
            "formal_form": "tau_read(c) = overlap_obj_disc(f_c) - lambda_n sigma_novel(c) - lambda_r sigma_ret(c)",
            "meaning": "the same object-disc overlap can be open, narrow, or fragile depending on concept-local stress",
        },
        "concept_transport_profiles": readout_transport,
        "headline_metrics": {
            "open_count": int(open_count),
            "narrow_count": int(narrow_count),
            "fragile_count": int(fragile_count),
            "Q6_status": inv_map["seven_question_mapping"]["Q6_discriminative_geometry"]["current_strength"],
        },
        "mathematical_meaning": {
            "why_needed": "Inventory alone explains object geometry, but readout closure depends on how local stress consumes transport budget.",
            "bottleneck_link": "This gives a more explicit shape to the main open point: shared object manifold to discriminative geometry compatibility.",
            "next_step": "promote concept-local transport budgets into family and phase transport operators",
        },
        "verdict": {
            "core_answer": "Stress-to-readout transport can now be treated as a concrete inventory-conditioned law rather than a vague bottleneck label.",
            "next_theory_target": "close family-level and phase-level transport operators using these concept-local budgets",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
