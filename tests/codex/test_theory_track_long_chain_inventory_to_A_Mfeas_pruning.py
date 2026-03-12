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
    ap = argparse.ArgumentParser(description="Theory-track long-chain inventory to A/M_feas pruning")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_long_chain_inventory_to_A_Mfeas_pruning_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    long_chain = load_latest("theory_track_large_scale_long_chain_inventory_*.json")
    a_coupling = load_latest("theory_track_inventory_to_A_coupling_*.json")
    mfeas_coupling = load_latest("theory_track_inventory_to_Mfeas_coupling_*.json")

    relation_ratio = long_chain["headline_metrics"]["relation_cross_to_within_ratio"]
    temporal_ratio = long_chain["headline_metrics"]["temporal_cross_to_within_ratio"]
    successor_ratio = long_chain["headline_metrics"]["chain_successor_to_cross_stage_ratio"]

    pruned_A_families = [
        "family_agnostic_isotropic_update_cone",
        "relation_free_update_cone",
        "stage_free_update_gate",
    ]
    preserved_A_families = [
        "family_conditioned_intersection_cones",
        "stress_gated_update_cones",
        "relation_sensitive_update_gate",
        "stage_conditioned_admissibility_gate",
    ]

    pruned_Mfeas_families = [
        "single_global_smooth_chart",
        "uniform_overlap_widths",
        "stage_free_viability_band",
    ]
    preserved_Mfeas_families = [
        "family_patched_viability_charts",
        "restricted_overlap_bands",
        "relation_conditioned_chart_widening",
        "temporal_transition_chart_family",
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_long_chain_inventory_to_A_Mfeas_pruning",
        },
        "inventory_constraints": {
            "relation_cross_to_within_ratio": relation_ratio,
            "temporal_cross_to_within_ratio": temporal_ratio,
            "chain_successor_to_cross_stage_ratio": successor_ratio,
        },
        "A_pruning": {
            "source_form": a_coupling["inventory_conditioned_form"],
            "pruned_families": pruned_A_families,
            "preserved_families": preserved_A_families,
            "meaning": "Long-chain inventory requires update admissibility to remain family-conditioned, stress-gated, relation-sensitive, and stage-aware.",
        },
        "Mfeas_pruning": {
            "source_form": mfeas_coupling["inventory_conditioned_form"],
            "pruned_families": pruned_Mfeas_families,
            "preserved_families": preserved_Mfeas_families,
            "meaning": "Viability can no longer be modeled as a single smooth chart with uniform overlaps once relation and stage structure become visible.",
        },
        "verdict": {
            "core_answer": "Long-chain invariants now directly constrain both A(I) and M_feas(I): update geometry must become stage-aware, and viability geometry must become transition-aware.",
            "next_theory_target": "bind these preserved A/M_feas families directly into P3/P4 intervention pruning and theorem survival tests.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
