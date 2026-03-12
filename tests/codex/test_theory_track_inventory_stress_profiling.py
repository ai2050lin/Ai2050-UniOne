from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import test_continuous_input_grounding_proto as proto
import test_continuous_multimodal_grounding_proto as cmg


ROOT = Path(__file__).resolve().parents[2]


def concept_groups() -> dict[str, list[str]]:
    return {family: proto.PHASE1[family] + proto.PHASE2[family] for family in proto.FAMILIES}


def concept_state(concept: str) -> np.ndarray:
    family = proto.concept_family(concept)
    vt = proto.family_basis()[family] + proto.concept_offset()[concept]
    lg = cmg.lang_family_basis()[family] + cmg.lang_concept_offset()[concept]
    return np.concatenate([vt, lg], axis=0).astype(np.float32)


def family_center(family: str) -> np.ndarray:
    states = [concept_state(concept) for concept in concept_groups()[family]]
    return np.mean(np.stack(states, axis=0), axis=0)


def arr(x: np.ndarray) -> list[float]:
    return [float(v) for v in x.tolist()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory-track inventory stress profiling")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_inventory_stress_profiling_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    groups = concept_groups()
    family_centers = {family: family_center(family) for family in proto.FAMILIES}
    all_concepts = [concept for family in proto.FAMILIES for concept in groups[family]]

    stress_rows = {}
    novelty_shock = 0.20
    retention_budget = 0.08
    relation_lift_gain = 0.35

    for concept in all_concepts:
        family = proto.concept_family(concept)
        state = concept_state(concept)
        centered = state - family_centers[family]
        within_neighbors = [other for other in groups[family] if other != concept]
        cross_neighbors = [other for other in all_concepts if proto.concept_family(other) != family]
        nearest_within = min(float(np.linalg.norm(state - concept_state(other))) for other in within_neighbors)
        nearest_cross = min(float(np.linalg.norm(state - concept_state(other))) for other in cross_neighbors)

        novelty_pressure = float(np.linalg.norm(centered) * novelty_shock)
        retention_risk = float(max(0.0, novelty_pressure - retention_budget))
        relation_lift_capacity = float(nearest_cross - nearest_within * relation_lift_gain)
        stable_under_stress = bool(retention_risk < 0.02 and relation_lift_capacity > 1.5)

        stress_rows[concept] = {
            "family": family,
            "centered_norm": float(np.linalg.norm(centered)),
            "nearest_within": nearest_within,
            "nearest_cross": nearest_cross,
            "novelty_pressure": novelty_pressure,
            "retention_risk": retention_risk,
            "relation_lift_capacity": relation_lift_capacity,
            "stable_under_stress": stable_under_stress,
            "centered_vector": arr(centered),
        }

    stable_count = sum(1 for row in stress_rows.values() if row["stable_under_stress"])
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_inventory_stress_profiling",
        },
        "headline_metrics": {
            "num_concepts": int(len(all_concepts)),
            "stable_under_stress_ratio": float(stable_count / max(1, len(all_concepts))),
            "mean_novelty_pressure": float(np.mean([row["novelty_pressure"] for row in stress_rows.values()])),
            "mean_retention_risk": float(np.mean([row["retention_risk"] for row in stress_rows.values()])),
            "mean_relation_lift_capacity": float(np.mean([row["relation_lift_capacity"] for row in stress_rows.values()])),
        },
        "stress_rows": stress_rows,
        "theory_implications": {
            "core_statement": "Inventory entries can be stress-profiled, so the atlas is no longer only static; each concept entry can now carry a novelty, retention, and relation-lift profile.",
            "why_useful": "This is the missing bridge from structural atlas to dynamic closure. It tells us which concept coordinates are fragile and which are robust.",
            "next_step": "tie these stress profiles to admissible-update operators and bridge-role lift candidates.",
        },
        "verdict": {
            "core_answer": "The encoding inventory can now be extended into a dynamic stress inventory rather than remaining a static atlas.",
            "next_theory_target": "formalize the inventory as a layered mathematical object with structural coordinates and dynamic stress fields",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
