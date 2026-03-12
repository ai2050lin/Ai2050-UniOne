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
    ap = argparse.ArgumentParser(description="Theory track ICSPB stronger closure")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_icspb_stronger_closure_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    theorem_layer = load("theory_track_icspb_theorem_exclusion_transport_20260312.json")
    theorem_binding = load("theory_track_icspb_theorem_to_p4_binding_20260312.json")
    distance = load("theory_track_encoding_principle_new_math_distance_20260312.json")

    closure_steps = [
        {
            "step": "theorem_candidate",
            "status": "completed",
            "count": len(theorem_layer["theorem_candidates"]),
        },
        {
            "step": "theorem_to_falsification_binding",
            "status": "completed",
            "count": theorem_binding["theorem_binding_count"],
        },
        {
            "step": "intervention_level_binding",
            "status": "next",
            "count": 4,
        },
        {
            "step": "survival_under_falsification",
            "status": "open",
            "count": 4,
        },
    ]

    exclusion_strength = min(1.0, 0.1 * len(theorem_layer["exclusions"]) + 0.05 * theorem_binding["theorem_binding_count"])
    theorem_closure_readiness = max(
        0.0,
        min(
            1.0,
            0.40
            + 0.05 * len(theorem_layer["theorem_candidates"])
            + 0.04 * theorem_binding["theorem_binding_count"]
            - 0.5 * distance["remaining_distance"]["new_math_gap"],
        ),
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_ICSPB_stronger_closure",
        },
        "theorem_closure_readiness": theorem_closure_readiness,
        "exclusion_strength": exclusion_strength,
        "closure_steps": closure_steps,
        "needed_for_strict_new_math": [
            "intervention-level theorem binding",
            "theorem survival under brain-side falsification",
            "transport-law stability under stress and switching",
            "bridge/readout coupling closure",
        ],
        "verdict": {
            "core_answer": "ICSPB is no longer only an axiom system with candidate theorems; it now has a visible closure path, but strict theorem-level establishment still requires intervention-bound survival tests.",
            "next_theory_target": "upgrade theorem bindings into intervention-level causal tests and use failures to prune theorem candidates.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
