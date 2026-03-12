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
    ap = argparse.ArgumentParser(description="Map engineering tracks to encoding mechanism layers")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_engineering_to_encoding_mechanism_mapping_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    progress = load("theory_track_brain_encoding_progress_assessment_20260312.json")
    p2 = load("stage_p2_stress_coupled_update_pruned_search_20260312.json")
    p3 = load("stage_p3_filtered_candidate_benchmark_20260312.json")
    bridge = load("stage_b_path_conditioned_bridge_filtered_search_20260312.json")
    p4 = load("stage_p4_brain_side_execution_report_20260312.json")
    path_reason = load("theory_track_modality_unified_reasoning_law_20260312.json")

    mapping = {
        "P2_update_track": {
            "encoding_layer": "local plastic update plus write/read separation",
            "current_role": "controls which update families are compatible with guarded-write and stable-read coding",
            "current_evidence": {
                "kept_pillar_count": p2["bridge_to_engineering"]["kept_pillar_count"],
                "guarded_write_count": p2["bridge_to_engineering"]["guarded_write_count"],
                "stable_read_count": p2["bridge_to_engineering"]["stable_read_count"],
            },
            "main_gap": "novelty, retention, and switching are not yet jointly closed in one validated dynamic law",
        },
        "P3_readout_track": {
            "encoding_layer": "shared object manifold to discriminative geometry transport",
            "current_role": "tests whether object atlas states can be transported into stable readout geometry",
            "current_evidence": {
                "best_family": p3["headline_metrics"]["best_family"],
                "best_candidate_score": p3["headline_metrics"]["best_candidate_score"],
                "mean_phase_budget": p3["headline_metrics"]["mean_phase_budget"],
            },
            "main_gap": progress["current_status"]["main_open_bottleneck"],
        },
        "B_line_bridge_track": {
            "encoding_layer": "bridge-role uplift from object entries into relation structure",
            "current_role": "tests whether relation and role structure can stay family-anchored rather than become free symbolic heads",
            "current_evidence": {
                "family_count": bridge["bridge_to_engineering"]["family_count"],
                "law_reference": bridge["law_reference"],
            },
            "main_gap": "bridge-role dynamics are filtered correctly, but not yet densely coupled into system-level closure",
        },
        "P4_brain_probe_track": {
            "encoding_layer": "brain-side projection and causal falsification of the encoding system",
            "current_role": "projects object, attribute, relation, and stress structure into brain-side probe form",
            "current_evidence": {
                "executed_probe_count": p4["headline_metrics"]["executed_probe_count"],
                "brain_side_stage": p4["status"]["brain_side_execution_stage"],
            },
            "main_gap": p4["status"]["remaining_gap"],
        },
        "reasoning_slice_track": {
            "encoding_layer": "modality-unified reasoning over family-conditioned shared slices",
            "current_role": "tests whether unified conscious-style reasoning is realized through conditioned entry and shared slices rather than one global loop",
            "current_evidence": {
                "law_name": path_reason["law_name"],
                "high_level_form": path_reason["high_level_form"],
            },
            "main_gap": "reasoning slices are formalized, but not yet integrated into engineering-side operator benchmarks",
        },
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_engineering_to_encoding_mechanism_mapping",
        },
        "mapping": mapping,
        "core_answer": "Every current engineering track can now be read as an explicit attack on one layer of the brain encoding mechanism rather than as an isolated benchmark line.",
        "verdict": {
            "core_answer": "Engineering progress is now structured as encoding-mechanism reconstruction rather than disconnected search.",
            "next_theory_target": "bind these five tracks into one encoding-mechanism closure loop with shared success criteria.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
