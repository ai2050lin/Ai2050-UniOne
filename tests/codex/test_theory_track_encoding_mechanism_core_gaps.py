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
    ap = argparse.ArgumentParser(description="Theory-track core gaps of the encoding mechanism")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_encoding_mechanism_core_gaps_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    progress = load("theory_track_brain_encoding_progress_assessment_20260312.json")
    mapping = load("theory_track_engineering_to_encoding_mechanism_mapping_20260312.json")
    icspb_pred = load("theory_track_icspb_falsifiable_predictions_20260312.json")
    reason_pred = load("theory_track_modality_unified_reasoning_predictions_20260312.json")

    gaps = [
        {
            "name": "object_to_readout_compatibility",
            "severity": "highest",
            "why_open": progress["current_status"]["main_open_bottleneck"],
            "affected_tracks": ["P3_readout_track"],
            "needed_next_move": "validate operator-form changes beyond local gain tuning",
        },
        {
            "name": "stress_bound_dynamic_update_closure",
            "severity": "high",
            "why_open": mapping["mapping"]["P2_update_track"]["main_gap"],
            "affected_tracks": ["P2_update_track"],
            "needed_next_move": "run stronger novelty-retention-switching coupled update benchmarks",
        },
        {
            "name": "bridge_role_dense_coupling",
            "severity": "high",
            "why_open": mapping["mapping"]["B_line_bridge_track"]["main_gap"],
            "affected_tracks": ["B_line_bridge_track"],
            "needed_next_move": "turn filtered bridge families into dense dynamic bridge benchmarks",
        },
        {
            "name": "brain_side_causal_closure",
            "severity": "highest",
            "why_open": mapping["mapping"]["P4_brain_probe_track"]["main_gap"],
            "affected_tracks": ["P4_brain_probe_track"],
            "needed_next_move": "convert executed probes into a causal falsification report and intervention tests",
        },
        {
            "name": "reasoning_slice_engineering_integration",
            "severity": "medium",
            "why_open": mapping["mapping"]["reasoning_slice_track"]["main_gap"],
            "affected_tracks": ["reasoning_slice_track", "P3_readout_track", "P4_brain_probe_track"],
            "needed_next_move": "bind reasoning-slice laws to readout, bridge, and brain-side predictions",
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_encoding_mechanism_core_gaps",
        },
        "gap_count": len(gaps),
        "gaps": gaps,
        "prediction_budget": {
            "icspb_prediction_count": icspb_pred["prediction_count"],
            "reasoning_prediction_count": reason_pred["prediction_count"],
        },
        "verdict": {
            "core_answer": "The remaining hard problems are now concentrated in a small set of encoding-mechanism gaps rather than spread across many unrelated theories.",
            "next_theory_target": "drive every next engineering step by one of these named encoding gaps.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
