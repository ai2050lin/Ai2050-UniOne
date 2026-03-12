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
    ap = argparse.ArgumentParser(description="Theory-track phase P1-P4 current map")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_phase_p1_p4_current_map_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    progress = load("theory_track_brain_encoding_progress_assessment_20260312.json")
    p2 = load("stage_p2_stress_coupled_update_pruned_search_20260312.json")
    p3 = load("stage_p3_operator_head_to_head_benchmark_20260312.json")
    p4 = load("stage_p4_brain_side_execution_report_20260312.json")

    phases = {
        "P1": {
            "meaning": "object manifold / bridge / role atlas consolidation",
            "current_score": progress["progress_snapshot"]["p1_score"],
            "current_state": "strong base layer",
            "main_value": "defines what is encoded and how family patches are organized",
        },
        "P2": {
            "meaning": "controlled update law + write-read separation + admissible update geometry",
            "current_score": progress["progress_snapshot"]["p2_score"],
            "current_state": "filtered dynamic law stage",
            "main_value": "controls how new evidence is written without destroying old structure",
            "kept_pillar_count": p2["bridge_to_engineering"]["kept_pillar_count"],
        },
        "P3": {
            "meaning": "shared object manifold to discriminative geometry compatibility",
            "current_score": progress["progress_snapshot"]["p3_score"],
            "current_state": "main bottleneck but now operator-sensitive",
            "main_value": "decides whether encoded objects can become stable decisions and readouts",
            "current_winner": p3["headline_metrics"]["winner"],
        },
        "P4": {
            "meaning": "brain-side mapping, probe execution, and causal falsification",
            "current_score": progress["progress_snapshot"]["p4_score"],
            "current_state": p4["status"]["brain_side_execution_stage"],
            "main_value": "tests whether the abstract encoding theory survives contact with brain-side evidence",
            "executed_probe_count": p4["headline_metrics"]["executed_probe_count"],
        },
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_phase_P1_P4_current_map",
        },
        "phases": phases,
        "verdict": {
            "core_answer": "P1-P4 now form a clean staged reconstruction of the encoding mechanism from object atlas to brain-side falsification.",
            "next_theory_target": "keep all next engineering actions explicitly attached to one of these four phases.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
