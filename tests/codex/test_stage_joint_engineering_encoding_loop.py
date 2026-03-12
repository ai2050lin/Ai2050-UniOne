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
    ap = argparse.ArgumentParser(description="Stage joint engineering-encoding loop")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_joint_engineering_encoding_loop_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    mapping = load("theory_track_engineering_to_encoding_mechanism_mapping_20260312.json")
    gaps = load("theory_track_encoding_mechanism_core_gaps_20260312.json")
    icspb_ops = load("theory_track_icspb_operator_generation_20260312.json")
    reason_law = load("theory_track_modality_unified_reasoning_law_20260312.json")

    execution_loop = [
        {
            "step": "P3_operator_benchmark",
            "encoding_gap": "object_to_readout_compatibility",
            "candidate": icspb_ops["current_closure_support"]["highest_priority_operator"],
            "why": "This is the current highest-priority operator-form change generated directly from ICSPB.",
        },
        {
            "step": "P2_stress_joint_benchmark",
            "encoding_gap": "stress_bound_dynamic_update_closure",
            "candidate": "stress_guarded_write_read_operator",
            "why": "Update-law closure must now be driven by novelty-retention-switching stress rather than free update search.",
        },
        {
            "step": "B_dense_bridge_benchmark",
            "encoding_gap": "bridge_role_dense_coupling",
            "candidate": "family_anchored_bridge_lift_operator",
            "why": "Bridge-space is already filtered; the next move is dense dynamic closure rather than more symbolic search.",
        },
        {
            "step": "P4_causal_falsification_bundle",
            "encoding_gap": "brain_side_causal_closure",
            "candidate": "object/attribute/relation/stress integrated probe report",
            "why": "Brain-side work must move from executed probes to causal falsification.",
        },
        {
            "step": "reasoning_slice_integration",
            "encoding_gap": "reasoning_slice_engineering_integration",
            "candidate": reason_law["law_name"],
            "why": "The new modality-unified reasoning law should be threaded into readout, bridge, and brain-side tests.",
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "Stage_joint_engineering_encoding_loop",
        },
        "execution_loop": execution_loop,
        "shared_principle": "Every engineering block must now declare which encoding-mechanism gap it attacks and which theoretical object it instantiates.",
        "current_tracks": list(mapping["mapping"].keys()),
        "current_gap_count": gaps["gap_count"],
        "verdict": {
            "core_answer": "Engineering and theory can now be tied together by one explicit encoding-mechanism closure loop.",
            "next_engineering_target": "run the first P3 operator benchmark under this joint loop and then update all other tracks against the same closure map.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
