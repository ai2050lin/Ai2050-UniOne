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
    ap = argparse.ArgumentParser(description="Stage P4 brain probe execution bundle")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p4_brain_probe_execution_bundle_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    probe = load("theory_track_inventory_brain_probe_coupling_20260312.json")
    progress = load("theory_track_brain_encoding_progress_assessment_20260312.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP4_brain_probe_execution_bundle",
        },
        "execution_bundle": {
            "object_probe_bundle": {
                "projection": probe["candidate_projection_rules"]["object_to_brain"],
                "goal": "test whether family basis and concept offsets map to region-parameterized cortical patterns",
            },
            "attribute_probe_bundle": {
                "projection": probe["candidate_projection_rules"]["attribute_to_brain"],
                "goal": "test whether attribute axes map to local cortical gradients",
            },
            "relation_probe_bundle": {
                "projection": probe["candidate_projection_rules"]["relation_to_brain"],
                "goal": "test whether bridge-role lifts map to cross-area coordination patterns",
            },
            "stress_probe_bundle": {
                "projection": probe["candidate_projection_rules"]["stress_to_brain"],
                "goal": "test whether novelty and retention stress map to plasticity-vs-stability signatures",
            },
        },
        "execution_order": [
            "1. object probes",
            "2. attribute probes",
            "3. relation probes",
            "4. stress probes",
        ],
        "why_this_order": {
            "core_answer": "The bundle starts from the strongest solved substrate and then climbs toward the still-open dynamic probes.",
            "open_bottleneck_reference": progress["current_status"]["main_open_bottleneck"],
        },
        "readiness": {
            "brain_mapping_status": progress["current_status"]["what_is_partially_closed"][-1],
            "brain_execution_gap": "protocol-ready but not yet causally executed",
            "bundle_status": "execution_ready",
        },
        "verdict": {
            "core_answer": "P4 can now move from abstract probe design to an ordered execution bundle.",
            "next_engineering_target": "start brain-side execution from object and attribute probes before escalating to relation and stress probes",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
