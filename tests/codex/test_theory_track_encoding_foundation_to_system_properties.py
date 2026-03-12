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
    ap = argparse.ArgumentParser(description="Theory-track encoding foundation to system properties")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_encoding_foundation_to_system_properties_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    theory = load("theory_track_new_math_theory_candidate_20260312.json")
    higher = load("theory_track_inventory_higher_order_geometry_20260312.json")
    progress = load("theory_track_brain_encoding_progress_assessment_20260312.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_encoding_foundation_to_system_properties",
        },
        "foundation_claim": {
            "core_statement": "The encoding mechanism is the generative root of large-scale nervous-system properties rather than one module among many.",
            "formal_form": "Properties_brain = Phi(H(I), A(I), M_feas(I), F, Q, R)",
            "meaning": "global nervous-system behavior is treated as a projection of the same encoding system, not as an unrelated layer above it",
        },
        "derived_system_properties": {
            "multimodal_unification": "arises because multiple modalities project into family-patched object sections over shared path bundles",
            "memory_stability": "arises because updates must remain inside admissible cones and viability strata",
            "relation_and_reasoning": "arise because bridge-role fibers are conditioned lifts from object sections",
            "selective_readout": "arises because readout is a path-conditioned query over restricted overlaps rather than a global flat head",
            "regional_specialization": "arises because the same encoding system is region-parameterized rather than because each region uses a different fundamental code",
        },
        "why_this_matters": {
            "core_answer": "If encoding is foundational, then solving the encoding mechanism is not solving one subproblem. It is solving the base object from which many brain-level properties are induced.",
            "current_support": progress["current_status"]["what_is_closed_most"],
            "main_open_gap": progress["current_status"]["main_open_bottleneck"],
        },
        "verdict": {
            "core_answer": "The theory track now treats brain-wide properties as generated consequences of the encoding mechanism rather than independent mysteries.",
            "next_theory_target": "derive more whole-system properties directly from the encoding object and use them as falsifiable predictions",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
