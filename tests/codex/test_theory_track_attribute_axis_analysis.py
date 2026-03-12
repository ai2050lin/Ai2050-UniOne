from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import test_continuous_input_grounding_proto as proto
import test_continuous_multimodal_grounding_proto as cmg


ROOT = Path(__file__).resolve().parents[2]


CONCEPT_ATTRIBUTES = {
    "apple": ["fruit", "edible", "round", "sweet", "concrete"],
    "banana": ["fruit", "edible", "sweet", "elongated", "concrete"],
    "pear": ["fruit", "edible", "sweet", "round", "concrete"],
    "cat": ["animal", "living", "mobile", "concrete", "domestic"],
    "dog": ["animal", "living", "mobile", "concrete", "domestic"],
    "horse": ["animal", "living", "mobile", "concrete", "large"],
    "truth": ["abstract", "cognitive", "symbolic", "stable"],
    "logic": ["abstract", "cognitive", "symbolic", "structured"],
    "memory": ["abstract", "cognitive", "symbolic", "persistent"],
}


def arr(x: np.ndarray) -> list[float]:
    return [float(v) for v in x.tolist()]


def concept_state(concept: str) -> np.ndarray:
    family = proto.concept_family(concept)
    vt = proto.family_basis()[family] + proto.concept_offset()[concept]
    lg = cmg.lang_family_basis()[family] + cmg.lang_concept_offset()[concept]
    return np.concatenate([vt, lg], axis=0).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory-track attribute axis analysis")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_attribute_axis_analysis_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    concepts = list(CONCEPT_ATTRIBUTES.keys())
    states = {concept: concept_state(concept) for concept in concepts}
    families = {concept: proto.concept_family(concept) for concept in concepts}
    family_centers = {}
    for family in proto.FAMILIES:
        family_concepts = [concept for concept in concepts if families[concept] == family]
        family_centers[family] = np.mean(np.stack([states[concept] for concept in family_concepts], axis=0), axis=0)

    centered = {concept: states[concept] - family_centers[families[concept]] for concept in concepts}
    all_attributes = sorted({attr for attrs in CONCEPT_ATTRIBUTES.values() for attr in attrs})

    attribute_axes = {}
    for attr in all_attributes:
        pos = [centered[concept] for concept in concepts if attr in CONCEPT_ATTRIBUTES[concept]]
        neg = [centered[concept] for concept in concepts if attr not in CONCEPT_ATTRIBUTES[concept]]
        pos_mean = np.mean(np.stack(pos, axis=0), axis=0) if pos else np.zeros_like(next(iter(centered.values())))
        neg_mean = np.mean(np.stack(neg, axis=0), axis=0) if neg else np.zeros_like(next(iter(centered.values())))
        axis = pos_mean - neg_mean
        attribute_axes[attr] = {
            "axis_vector": arr(axis),
            "axis_norm": float(np.linalg.norm(axis)),
            "top_dims": [int(i) for i in np.argsort(np.abs(axis))[-4:][::-1].tolist()],
        }

    concept_attribute_alignment = {}
    for concept in concepts:
        aligns = {}
        for attr in CONCEPT_ATTRIBUTES[concept]:
            aligns[attr] = cosine(centered[concept], np.array(attribute_axes[attr]["axis_vector"], dtype=np.float32))
        concept_attribute_alignment[concept] = {
            "family": families[concept],
            "attributes": CONCEPT_ATTRIBUTES[concept],
            "alignment": aligns,
            "mean_alignment": float(np.mean(list(aligns.values()))),
        }

    attribute_summary = {}
    for attr in all_attributes:
        holders = [concept for concept in concepts if attr in CONCEPT_ATTRIBUTES[concept]]
        vals = [concept_attribute_alignment[concept]["alignment"][attr] for concept in holders]
        attribute_summary[attr] = {
            "holders": holders,
            "mean_alignment": float(np.mean(vals)),
            "axis_norm": attribute_axes[attr]["axis_norm"],
            "top_dims": attribute_axes[attr]["top_dims"],
        }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_attribute_axis_analysis",
        },
        "headline_metrics": {
            "num_attributes": int(len(all_attributes)),
            "mean_attribute_alignment": float(np.mean([v["mean_alignment"] for v in concept_attribute_alignment.values()])),
            "mean_axis_norm": float(np.mean([attribute_summary[attr]["axis_norm"] for attr in all_attributes])),
        },
        "attribute_axes": attribute_summary,
        "concept_attribute_alignment": concept_attribute_alignment,
        "theory_implications": {
            "core_statement": "Concept attributes can be modeled as local directions on family-centered object charts rather than as disconnected symbolic tags.",
            "why_useful": "This gives the atlas another layer: not just concept offsets, but reusable attribute-like axes that can accumulate across concepts.",
            "next_step": "connect attribute axes to relation axes and novelty stress to see which axes are stable under admissible updates.",
        },
        "verdict": {
            "core_answer": "Attribute-level analysis can now be layered on top of concept atlas entries to build a richer encoding inventory.",
            "next_theory_target": "expand from concept inventory plus attribute axes to concept-relation-attribute atlas synthesis",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
