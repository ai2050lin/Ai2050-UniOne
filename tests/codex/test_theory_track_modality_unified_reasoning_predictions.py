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
    ap = argparse.ArgumentParser(description="Theory-track modality-unified reasoning predictions")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_modality_unified_reasoning_predictions_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    law = load("theory_track_modality_unified_reasoning_law_20260312.json")
    p4 = load("stage_p4_brain_side_execution_report_20260312.json")
    icspb_pred = load("theory_track_icspb_falsifiable_predictions_20260312.json")

    predictions = [
        {
            "id": "MR1_modality_substitution_preserves_reasoning_patch",
            "statement": "If the same concept is entered through different modalities, reasoning should stay inside the same family-conditioned reasoning slice before final readout.",
            "observable": "object and attribute probes should preserve family patch identity under modality substitution",
        },
        {
            "id": "MR2_reasoning_depends_on_lift_not_raw_modality",
            "statement": "Reasoning degradation should be explained more by failed modality lift into the reasoning slice than by raw modality absence alone.",
            "observable": "performance drop should correlate with lift bottlenecks, not only with single-modality masking",
        },
        {
            "id": "MR3_cross_modal_reasoning_uses_shared_slice",
            "statement": "Reasoning transfer across modalities should reuse a shared family-conditioned slice rather than a fully global central loop.",
            "observable": "family-conditioned scaffold should outperform fully shared central-loop style hypotheses",
        },
        {
            "id": "MR4_stress_binds_reasoning_access",
            "statement": "Novelty and retention stress should narrow reasoning access paths before they erase concept identity.",
            "observable": "guarded-write / stable-read asymmetry should appear before concept collapse",
        },
        {
            "id": "MR5_relation_reasoning_is_lift_based",
            "statement": "Reasoning about relations should preserve family-anchored bridge structure rather than invoke a free symbolic role layer.",
            "observable": "relation probes should continue to favor family-anchored bridge lift",
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_modality_unified_reasoning_predictions",
        },
        "law_name": law["law_name"],
        "available_probe_batches": p4["headline_metrics"]["executed_probe_count"],
        "prediction_count": len(predictions),
        "predictions": predictions,
        "linked_existing_predictions": icspb_pred["prediction_count"],
        "verdict": {
            "core_answer": "The consciousness clue now yields a concrete prediction set for modality-unified reasoning inside ICSPB.",
            "next_theory_target": "route these reasoning predictions into the next P4 causal falsification bundle.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
