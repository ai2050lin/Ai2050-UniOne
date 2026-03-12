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
    ap = argparse.ArgumentParser(description="Theory-track brain encoding mechanism current synthesis")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_brain_encoding_mechanism_current_synthesis_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inv_sys = load("theory_track_inventory_unified_system_formalization_20260312.json")
    inv_map = load("theory_track_inventory_seven_question_mapping_20260312.json")
    bottleneck = load("theory_track_inventory_bottleneck_resolution_analysis_20260312.json")
    bridge = load("theory_track_inventory_bridge_role_coupling_20260312.json")
    brain = load("theory_track_inventory_brain_probe_coupling_20260312.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_brain_encoding_mechanism_current_synthesis",
        },
        "current_puzzle": {
            "already_supported": [
                "family-patched object atlas",
                "concept identity as family basis plus local offset",
                "attribute directions as local chart axes",
                "cross-family patch boundaries",
                "restricted object-to-disc overlap",
                "inventory-conditioned admissible and viable structures",
            ],
            "partially_supported": [
                "local update law",
                "write/read separation",
                "bridge-role lift",
                "brain-side projection",
            ],
            "main_open_point": bottleneck["current_bottlenecks"]["main_bottleneck"],
        },
        "feature_to_code_process": [
            "multimodal input first lands on a family patch of the object atlas",
            "local evidence forms a concept-specific family offset inside that patch",
            "attribute-like directions refine the concept state on the same local chart",
            "admissible-update constraints decide which updates are allowed",
            "restricted overlaps decide whether state can be transported toward memory, relation, or readout charts",
            "bridge-role lift organizes object entries into relation and role structure",
            "brain-side realizations should be region-parameterized projections of this same inventory-conditioned system",
        ],
        "seven_question_status": inv_map["seven_question_mapping"],
        "unified_system": inv_sys["candidate_equations"],
        "bridge_role_status": bridge["candidate_equations"],
        "brain_probe_status": brain["candidate_projection_rules"],
        "verdict": {
            "core_answer": "The current best synthesis is that brain encoding is a controlled encoding dynamical system organized around an inventory-conditioned object atlas.",
            "next_theory_target": "close stress-to-readout transport and brain-side execution so the remaining open points move from partial to strong support",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
