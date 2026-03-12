from __future__ import annotations

import argparse
import json
import statistics
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
    ap = argparse.ArgumentParser(description="Theory-track DNN encoding pattern mining")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_dnn_encoding_pattern_mining_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    concept = load_latest("theory_track_large_scale_concept_inventory_analysis_*.json")
    concept_synth = load_latest("theory_track_large_scale_inventory_to_brain_math_synthesis_*.json")
    relctx = load_latest("theory_track_large_inventory_relation_context_synthesis_*.json")
    long_chain = load_latest("theory_track_large_scale_long_chain_inventory_*.json")
    frontier = load_latest("theory_track_long_chain_block_progress_assessment_*.json")

    family_ratio_series = [
        concept["headline_metrics"]["cross_to_within_ratio"],
        relctx["global_constraints"]["family_cross_to_within_ratio"],
        long_chain["headline_metrics"]["family_cross_to_within_ratio"],
    ]
    relation_ratio_series = [
        relctx["global_constraints"]["relation_cross_to_within_ratio"],
        long_chain["headline_metrics"]["relation_cross_to_within_ratio"],
    ]
    context_ratio_series = [
        relctx["global_constraints"]["context_cross_to_within_ratio"],
        long_chain["headline_metrics"]["context_cross_to_within_ratio"],
    ]
    temporal_ratio = long_chain["headline_metrics"]["temporal_cross_to_within_ratio"]
    successor_ratio = long_chain["headline_metrics"]["chain_successor_to_cross_stage_ratio"]

    recurrent_dims = concept["global_recurrent_dims"]
    stable_patterns = [
        {
            "pattern": "family_patch_stability",
            "evidence": round(statistics.mean(family_ratio_series), 6),
            "meaning": "family patch separation remains strong from concept-only to long-chain inventory",
        },
        {
            "pattern": "recurrent_scaffold_reuse",
            "evidence": len(recurrent_dims),
            "meaning": "a small reusable scaffold appears across families rather than a fully disjoint codebook",
        },
        {
            "pattern": "relation_fiber_emergence",
            "evidence": round(statistics.mean(relation_ratio_series), 6),
            "meaning": "relation structure becomes visible and strengthens as inventory moves toward reasoning traces",
        },
        {
            "pattern": "context_fiber_emergence",
            "evidence": round(statistics.mean(context_ratio_series), 6),
            "meaning": "context behaves more like a conditioning fiber than a dominant class axis",
        },
        {
            "pattern": "temporal_transition_emergence",
            "evidence": round(temporal_ratio, 6),
            "meaning": "temporal stage structure is detectable but weaker than family structure, suggesting operator-like rather than class-like behavior",
        },
        {
            "pattern": "successor_coherence_emergence",
            "evidence": round(successor_ratio, 6),
            "meaning": "reasoning chains begin to exhibit local successor coherence, but not yet strongly enough for full closure",
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_dnn_encoding_pattern_mining",
        },
        "inventory_scales": {
            "concept_only": concept["headline_metrics"]["num_concepts"],
            "concept_relation_context": relctx["inventory_scale"]["concept_relation_context_count"],
            "long_chain": long_chain["headline_metrics"]["num_concepts"],
        },
        "stable_patterns": stable_patterns,
        "frontier_state": {
            "gain_total_vs_baseline": frontier["closure_metrics"]["gain_total_vs_baseline"],
            "strict_survivals": frontier["closure_metrics"]["strict_survivals"],
            "provisional_survivals": frontier["closure_metrics"]["provisional_survivals"],
            "extended_theorem_count": frontier["closure_metrics"]["extended_theorem_count"],
        },
        "verdict": {
            "core_answer": "DNN-side inventory mining now exposes a stable set of coding invariants: family patches, reusable recurrent scaffold, relation/context fibers, temporal transitions, and emerging successor coherence.",
            "next_theory_target": "turn these invariants into hard reverse constraints on brain encoding and into stricter completion targets for ICSPB.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
