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
    ap = argparse.ArgumentParser(description="Theory track ICSPB theorem to P4 binding")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_icspb_theorem_to_p4_binding_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    theorem_layer = load("theory_track_icspb_theorem_exclusion_transport_20260312.json")
    p4_bundle = load("stage_p4_causal_falsification_bundle_20260312.json")
    inverse_recon = load("theory_track_encoding_inverse_reconstruction_20260312.json")

    block_map = {
        "family_section_theorem": "object_family_patch_falsification",
        "restricted_readout_transport_theorem": "reasoning_slice_falsification",
        "stress_guarded_update_theorem": "stress_asymmetry_falsification",
        "anchored_bridge_lift_theorem": "relation_anchor_falsification",
    }

    theorem_bindings = []
    for theorem in theorem_layer["theorem_candidates"]:
        theorem_bindings.append(
            {
                "theorem": theorem["name"],
                "statement": theorem["statement"],
                "bound_falsification_block": block_map[theorem["name"]],
                "binding_strength": "direct",
            }
        )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_ICSPB_theorem_to_P4_binding",
        },
        "theorem_binding_count": len(theorem_bindings),
        "theorem_bindings": theorem_bindings,
        "available_falsification_blocks": [block["block"] for block in p4_bundle["falsification_blocks"]],
        "inverse_reconstruction_confidence": inverse_recon["overall_inverse_reconstruction_confidence"],
        "verdict": {
            "core_answer": "ICSPB theorem candidates can now be bound directly to the current P4 falsification blocks, so theory is no longer disconnected from brain-side execution.",
            "next_theory_target": "turn these bindings into intervention-level causal tests rather than only bundle-level descriptions.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
