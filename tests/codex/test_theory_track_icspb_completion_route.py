from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def latest_match(pattern: str) -> Path:
    matches = sorted(TEMP_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    return matches[0]


def load_latest(pattern: str) -> dict:
    return json.loads(latest_match(pattern).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory-track ICSPB completion route")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_icspb_completion_route_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    mined = load_latest("theory_track_dnn_encoding_pattern_mining_*.json")
    reverse_constraints = load_latest("theory_track_dnn_to_brain_reverse_constraints_*.json")
    frontier = load_latest("theory_track_long_chain_block_progress_assessment_*.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_ICSPB_completion_route",
        },
        "completion_route": [
            {
                "phase": "R1",
                "name": "strictify_stage_successor_survival",
                "target": "push the two provisional theorem survivals into stronger pass/fail status",
            },
            {
                "phase": "R2",
                "name": "expand_realistic_reasoning_inventory",
                "target": "grow inventory toward more natural, longer reasoning traces and strengthen successor coherence",
            },
            {
                "phase": "R3",
                "name": "bind_reverse_constraints_to_P4",
                "target": "convert DNN-derived reverse constraints into stricter brain-side falsification targets",
            },
            {
                "phase": "R4",
                "name": "close_stress_and_bridge_frontiers",
                "target": "bring stress_guarded_update and anchored_bridge_lift into the active survival frontier",
            },
        ],
        "current_frontier": frontier["closure_metrics"],
        "reverse_constraint_count": len(reverse_constraints["constraints"]),
        "stable_pattern_count": len(mined["stable_patterns"]),
        "verdict": {
            "core_answer": "ICSPB completion no longer looks like open-ended exploration; it now has a four-phase completion route driven by DNN-side invariants, theorem frontier status, and brain-side reverse constraints.",
            "next_theory_target": "execute R1 first as the most leverage-heavy phase, because it directly tightens both the encoding theory and the new mathematical framework.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
