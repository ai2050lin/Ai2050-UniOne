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
    ap = argparse.ArgumentParser(description="Theory-track new math theory candidate")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_new_math_theory_candidate_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    higher = load("theory_track_inventory_higher_order_geometry_20260312.json")
    unified = load("theory_track_inventory_unified_system_formalization_20260312.json")
    progress = load("theory_track_brain_encoding_progress_assessment_20260312.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_new_math_theory_candidate",
        },
        "theory_candidate": {
            "name": "Inventory-Conditioned Stratified Path-Bundle Theory",
            "short_name": "ICSPB",
            "core_claim": "Brain encoding is governed by a stratified family-patch base manifold whose concept sections carry attribute, relation, and stress fibers, while learning and readout are controlled by admissible path bundles rather than flat vector updates alone.",
            "unified_form": "Sys*(I) = (H(I), A(I), M_feas(I), F, Q, R)",
            "bundle_reference": higher["higher_order_object"]["high_level_form"],
            "system_reference": unified["unified_object"]["high_level_form"],
        },
        "why_new_theory_is_needed": {
            "core_answer": "Current mathematics is good at flat vector spaces, static manifolds, or simple dynamical systems. The observed coding structure now appears to require base charts, attached fibers, admissible path constraints, and restricted overlaps all at once.",
            "signs": [
                "flat global chart hypotheses were excluded",
                "simple local gain tuning plateaued in P3",
                "inventory now yields operator families, stress fibers, and path-conditioned readout/bridge laws",
            ],
        },
        "what_this_theory_explains": {
            "Q1_Q5_Q6": "why object patches, crossmodal identity, and readout constraints can coexist",
            "Q2_Q3": "why write/read should be stress-coupled local path gates",
            "Q4": "why bridge-role is a conditioned lift rather than a free symbolic module",
            "Q7": "why brain mapping should preserve patch structure and fiber projections rather than only whole-brain connectivity",
        },
        "current_limitations": {
            "main_open_gap": progress["current_status"]["main_open_bottleneck"],
            "still_missing": [
                "causal brain-side execution closure",
                "validated operator-form change beyond prediction",
                "fully explicit overlap and projection operators",
            ],
        },
        "verdict": {
            "core_answer": "A plausible new mathematical theory candidate has now emerged: Inventory-Conditioned Stratified Path-Bundle Theory.",
            "next_theory_target": "use ICSPB to generate the next operator-form changes and the next falsifiable brain-side integration tests",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
