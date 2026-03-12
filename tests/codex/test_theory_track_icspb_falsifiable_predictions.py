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
    ap = argparse.ArgumentParser(description="Theory-track ICSPB falsifiable predictions")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_icspb_falsifiable_predictions_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    axioms = load("theory_track_icspb_axiom_layer_20260312.json")
    operators = load("theory_track_icspb_operator_generation_20260312.json")
    p4 = load("stage_p4_brain_side_execution_report_20260312.json")
    probes = load("stage_p4_relation_stress_probe_execution_20260312.json")
    benchmark = load("stage_p3_recurrent_dim_scaffolded_readout_benchmark_20260312.json")

    predictions = [
        {
            "id": "P1_family_patch_probe",
            "statement": "Object probes should continue to show family-patch separation rather than global uniform mixing.",
            "expected_signal": "same-family overlap remains higher than cross-family overlap",
            "linked_axioms": ["A1_family_stratification", "A2_section_based_concepts"],
        },
        {
            "id": "P2_recurrent_dim_readout",
            "statement": "Scaffolded readout on recurrent dims should lift abstract-family readout before weaker operator forms do.",
            "expected_signal": f"predicted_score approaches {benchmark['predicted_benchmark']['predicted_score']}",
            "linked_axioms": ["A5_stratified_viability", "A6_path_conditioned_computation"],
        },
        {
            "id": "P3_relation_anchor",
            "statement": "Relation probes should support family-anchored bridges rather than a free symbolic role layer.",
            "expected_signal": f"bridge_ready_count stays at {probes['executed_probes']['relation_probe']['bridge_ready_count']}",
            "linked_axioms": ["A3_attached_fibers", "A6_path_conditioned_computation"],
        },
        {
            "id": "P4_stress_asymmetry",
            "statement": "Stress probes should continue to show guarded-write versus stable-read asymmetry.",
            "expected_signal": (
                f"open_write={probes['executed_probes']['stress_probe']['open_write_count']}, "
                f"guarded_write={probes['executed_probes']['stress_probe']['guarded_write_count']}, "
                f"stable_read={probes['executed_probes']['stress_probe']['stable_read_count']}"
            ),
            "linked_axioms": ["A3_attached_fibers", "A4_intersected_admissibility"],
        },
        {
            "id": "P5_phase_transport_budget",
            "statement": "The novelty-to-read channel should stay narrower than the stabilize-to-read channel without collapsing to zero.",
            "expected_signal": "phase-conditioned transport remains narrow-open rather than collapse",
            "linked_axioms": ["A5_stratified_viability", "A6_path_conditioned_computation"],
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_ICSPB_falsifiable_predictions",
        },
        "theory_name": axioms["theory_name"],
        "executed_probe_count": p4["headline_metrics"]["executed_probe_count"],
        "prediction_count": len(predictions),
        "predictions": predictions,
        "bridge_and_stress_execution_status": {
            "relation_probe_status": probes["executed_probes"]["relation_probe"]["status"],
            "stress_probe_status": probes["executed_probes"]["stress_probe"]["status"],
            "highest_priority_operator": operators["current_closure_support"]["highest_priority_operator"],
        },
        "verdict": {
            "core_answer": "ICSPB now yields explicit falsifiable predictions that can drive the next brain-side and P3 tests.",
            "next_theory_target": "close the loop by validating the recurrent-dim scaffolded readout and integrating probe evidence into one causal falsification report.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
