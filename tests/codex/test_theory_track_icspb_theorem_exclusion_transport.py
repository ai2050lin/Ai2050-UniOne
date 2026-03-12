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
    ap = argparse.ArgumentParser(description="Theory track ICSPB theorem / exclusion / transport layer")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_icspb_theorem_exclusion_transport_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    axioms = load("theory_track_icspb_axiom_layer_20260312.json")
    operators = load("theory_track_icspb_operator_generation_20260312.json")
    predictions = load("theory_track_icspb_falsifiable_predictions_20260312.json")
    inverse_recon = load("theory_track_encoding_inverse_reconstruction_20260312.json")

    theorem_candidates = [
        {
            "name": "family_section_theorem",
            "statement": "If A1 and A2 hold, stable concept identity is better modeled as family-conditioned local sections than as globally uniform points.",
            "depends_on": ["A1_family_stratification", "A2_section_based_concepts"],
        },
        {
            "name": "restricted_readout_transport_theorem",
            "statement": "If A5 and A6 hold, discriminative readout can only remain stable on restricted overlaps and admissible paths, not by direct global collapse.",
            "depends_on": ["A5_stratified_viability", "A6_path_conditioned_computation"],
        },
        {
            "name": "stress_guarded_update_theorem",
            "statement": "If A3 and A4 hold, novelty-heavy updates must contract into guarded-write regimes before stable-read collapses.",
            "depends_on": ["A3_attached_fibers", "A4_intersected_admissibility"],
        },
        {
            "name": "anchored_bridge_lift_theorem",
            "statement": "If A1, A3, and A6 hold, relation-role structure should emerge as a family-anchored lift rather than a free symbolic layer.",
            "depends_on": ["A1_family_stratification", "A3_attached_fibers", "A6_path_conditioned_computation"],
        },
    ]

    exclusions = [
        "single_global_smooth_object_chart",
        "global_isotropic_transport",
        "direct_object_to_disc_collapse",
        "free_symbolic_role_layer",
        "fully_shared_global_central_loop_as_sufficient_explanation",
    ]

    transport_law = {
        "law_name": "ICSPB_transport_law",
        "formal": "Tau_total(c, mode_1 -> mode_2) = Tau_read^(f_c) + Phi(mode_1 -> mode_2) + Psi_reason^(f_c) - switch_cost(c) - fragility(c)",
        "components": {
            "Tau_read^(f_c)": "family-conditioned readout transport budget",
            "Phi(mode_1 -> mode_2)": "phase switching contribution",
            "Psi_reason^(f_c)": "family-conditioned reasoning slice support",
            "switch_cost(c)": "mode transition cost for concept c",
            "fragility(c)": "transport fragility under stress and overlap limits",
        },
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_ICSPB_theorem_exclusion_transport",
        },
        "axiom_count": axioms["axiom_count"],
        "generated_operator_count": operators["generated_operator_count"],
        "prediction_count": predictions["prediction_count"],
        "inverse_reconstruction_confidence": inverse_recon["overall_inverse_reconstruction_confidence"],
        "theorem_candidates": theorem_candidates,
        "exclusions": exclusions,
        "transport_law": transport_law,
        "verdict": {
            "core_answer": "ICSPB has now advanced from axiom and operator levels into a first theorem / exclusion / transport-law layer.",
            "next_theory_target": "bind these theorem candidates to brain-side falsification and filtered engineering benchmarks.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
