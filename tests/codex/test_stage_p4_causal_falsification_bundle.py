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
    ap = argparse.ArgumentParser(description="Stage P4 causal falsification bundle")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p4_causal_falsification_bundle_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    report = load("stage_p4_brain_side_execution_report_20260312.json")
    icspb_pred = load("theory_track_icspb_falsifiable_predictions_20260312.json")
    reason_pred = load("theory_track_modality_unified_reasoning_predictions_20260312.json")
    clue = load("theory_track_conscious_modality_unification_clue_20260312.json")

    falsification_blocks = [
        {
            "block": "object_family_patch_falsification",
            "evidence_source": "object_probe",
            "target_claim": "family-patch separation is preserved in brain-side projection",
            "failure_condition": "cross-family mixing rises to same-family overlap level",
        },
        {
            "block": "attribute_direction_falsification",
            "evidence_source": "attribute_probe",
            "target_claim": "attribute gradients remain local chart directions rather than arbitrary labels",
            "failure_condition": "attribute structure cannot be read as patch-local gradients",
        },
        {
            "block": "relation_anchor_falsification",
            "evidence_source": "relation_probe",
            "target_claim": "relation reasoning stays family-anchored instead of becoming a free symbolic role layer",
            "failure_condition": "family anchoring disappears while relation performance remains high",
        },
        {
            "block": "stress_asymmetry_falsification",
            "evidence_source": "stress_probe",
            "target_claim": "guarded-write and stable-read asymmetry is a real part of the coding system",
            "failure_condition": "stress perturbation does not narrow write before concept collapse",
        },
        {
            "block": "reasoning_slice_falsification",
            "evidence_source": "cross-modal reasoning predictions",
            "target_claim": clue["core_inference"]["more_plausible_form"],
            "failure_condition": "a fully shared global loop outperforms conditioned-entry shared-slice explanations",
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP4_causal_falsification_bundle",
        },
        "probe_execution_stage": report["status"]["brain_side_execution_stage"],
        "icspb_prediction_count": icspb_pred["prediction_count"],
        "reasoning_prediction_count": reason_pred["prediction_count"],
        "falsification_blocks": falsification_blocks,
        "verdict": {
            "core_answer": "P4 now has a concrete causal falsification bundle built from executed probes plus ICSPB and reasoning-slice predictions.",
            "next_engineering_target": "run the next brain-side round as intervention tests against these five falsification blocks rather than as generic probe collection.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
