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
    ap = argparse.ArgumentParser(description="Theory-track ICSPB operator generation")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_icspb_operator_generation_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    axioms = load("theory_track_icspb_axiom_layer_20260312.json")
    info = load("theory_track_inventory_information_gain_summary_20260312.json")
    mapping = load("theory_track_inventory_improvement_mapping_20260312.json")
    op_change = load("stage_p3_inventory_guided_operator_form_change_20260312.json")
    stress = load("theory_track_stress_coupled_write_read_law_20260312.json")

    improvements = mapping["information_to_improvement"]
    op_candidates = op_change["operator_form_change_candidates"]

    operators = [
        {
            "name": "recurrent_dim_scaffolded_readout",
            "source_axioms": ["A1_family_stratification", "A5_stratified_viability", "A6_path_conditioned_computation"],
            "inventory_evidence": [
                info["new_information"]["recurrent_dimensions"]["meaning"],
                info["new_information"]["restricted_overlap"]["meaning"],
            ],
            "predicted_gain": op_candidates["recurrent_dim_scaffolded_readout"]["predicted_gain"],
            "priority": "highest",
        },
        {
            "name": "dual_overlap_transport_operator",
            "source_axioms": ["A4_intersected_admissibility", "A5_stratified_viability", "A6_path_conditioned_computation"],
            "inventory_evidence": [
                info["new_information"]["restricted_overlap"]["meaning"],
                "restricted overlap has already forced readout to become path-conditioned transport",
            ],
            "predicted_gain": op_candidates["dual_overlap_transport_operator"]["predicted_gain"],
            "priority": "high",
        },
        {
            "name": "family_low_rank_readout_operator",
            "source_axioms": ["A1_family_stratification", "A2_section_based_concepts", "A3_attached_fibers"],
            "inventory_evidence": [
                info["new_information"]["low_rank_family_axes"]["meaning"],
                "family low-rank axes have already entered operator-family design",
            ],
            "predicted_gain": op_candidates["family_low_rank_readout_operator"]["predicted_gain"],
            "priority": "medium",
        },
        {
            "name": "stress_guarded_write_read_operator",
            "source_axioms": ["A3_attached_fibers", "A4_intersected_admissibility", "A6_path_conditioned_computation"],
            "inventory_evidence": [
                "Q2 and Q3 have been rewritten as a stress-coupled write/read law",
                "P4 has been expanded into a stress probe bundle",
            ],
            "predicted_gain": 0.0,
            "priority": "closure_support",
        },
        {
            "name": "family_anchored_bridge_lift_operator",
            "source_axioms": ["A1_family_stratification", "A3_attached_fibers", "A6_path_conditioned_computation"],
            "inventory_evidence": [
                "bridge-role search has already been rewritten into a family-anchored role-kernel form",
                "bridge lift is now jointly constrained by restricted overlap and family anchoring",
            ],
            "predicted_gain": 0.0,
            "priority": "bridge_support",
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_ICSPB_operator_generation",
        },
        "theory_name": axioms["theory_name"],
        "generated_operator_count": len(operators),
        "operators": operators,
        "current_closure_support": {
            "stable_read_count": stress["headline_metrics"]["stable_read_count"],
            "guarded_write_count": stress["headline_metrics"]["guarded_write_count"],
            "highest_priority_operator": operators[0]["name"],
        },
        "verdict": {
            "core_answer": "ICSPB now generates concrete operator families rather than only abstract structure claims.",
            "next_theory_target": "run falsifiable prediction synthesis and route the highest-priority operator to the next P3 benchmark.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
