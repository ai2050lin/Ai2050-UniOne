from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import test_continuous_input_grounding_proto as proto
import test_continuous_multimodal_grounding_proto as cmg


ROOT = Path(__file__).resolve().parents[2]


def arr(x: np.ndarray) -> list[float]:
    return [float(v) for v in x.tolist()]


def concept_groups() -> dict[str, list[str]]:
    return {family: proto.PHASE1[family] + proto.PHASE2[family] for family in proto.FAMILIES}


def concept_state(family: str, concept: str) -> np.ndarray:
    vt = proto.family_basis()[family] + proto.concept_offset()[concept]
    lg = cmg.lang_family_basis()[family] + cmg.lang_concept_offset()[concept]
    return np.concatenate([vt, lg], axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory-track concept encoding inventory")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_concept_encoding_inventory_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    groups = concept_groups()
    states: dict[str, np.ndarray] = {}
    family_center: dict[str, np.ndarray] = {}
    concept_inventory: dict[str, dict[str, object]] = {}

    for family, concepts in groups.items():
        stack = np.stack([concept_state(family, concept) for concept in concepts], axis=0)
        family_center[family] = np.mean(stack, axis=0)
        for concept, state in zip(concepts, stack):
            states[concept] = state

    all_concepts = [concept for family in proto.FAMILIES for concept in groups[family]]
    for concept in all_concepts:
        family = proto.concept_family(concept)
        state = states[concept]
        within = []
        cross = []
        for other in all_concepts:
            if other == concept:
                continue
            dist = float(np.linalg.norm(state - states[other]))
            record = {"concept": other, "family": proto.concept_family(other), "distance": dist}
            if proto.concept_family(other) == family:
                within.append(record)
            else:
                cross.append(record)
        within.sort(key=lambda item: item["distance"])
        cross.sort(key=lambda item: item["distance"])
        family_offset = state - family_center[family]
        concept_inventory[concept] = {
            "family": family,
            "state_norm": float(np.linalg.norm(state)),
            "family_offset_norm": float(np.linalg.norm(family_offset)),
            "nearest_same_family": within[:2],
            "nearest_cross_family": cross[:2],
            "within_to_cross_margin": float(cross[0]["distance"] - within[0]["distance"]),
            "family_offset_top_dims": [int(i) for i in np.argsort(np.abs(family_offset))[-4:][::-1].tolist()],
            "family_offset_vector": arr(family_offset),
        }

    avg_margin = float(np.mean([concept_inventory[concept]["within_to_cross_margin"] for concept in all_concepts]))
    avg_offset = float(np.mean([concept_inventory[concept]["family_offset_norm"] for concept in all_concepts]))

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_concept_encoding_inventory",
        },
        "headline_metrics": {
            "num_concepts": int(len(all_concepts)),
            "mean_family_offset_norm": avg_offset,
            "mean_within_to_cross_margin": avg_margin,
        },
        "concept_inventory": concept_inventory,
        "system_implications": {
            "core_statement": "Each concept can now be represented as one entry in a growing object-atlas inventory: family membership, family offset, local neighbors, and cross-family margins.",
            "why_useful": "This turns abstract encoding talk into concrete concept-level chart entries that can be accumulated into a denser atlas.",
            "next_step": "attach attributes, relations, and novelty stress to each entry",
        },
        "verdict": {
            "core_answer": "A concept inventory can now be built systematically rather than one concept at a time.",
            "next_theory_target": "add attribute axes and relation probes on top of the concept inventory",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
