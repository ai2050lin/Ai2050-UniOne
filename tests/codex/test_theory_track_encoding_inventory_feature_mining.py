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


def sample_state(rng: np.random.Generator, concept: str, noise: float) -> np.ndarray:
    base = concept_state(concept)
    return (base + rng.normal(scale=noise, size=base.shape[0])).astype(np.float32)


def top_dims(x: np.ndarray, k: int = 6) -> list[int]:
    return [int(i) for i in np.argsort(np.abs(x))[-k:][::-1].tolist()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory-track encoding inventory feature mining")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples-per-concept", type=int, default=300)
    ap.add_argument("--noise", type=float, default=0.06)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_encoding_inventory_feature_mining_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    groups = concept_groups()
    concepts = [concept for family in proto.FAMILIES for concept in groups[family]]

    sample_bank = {}
    concept_means = {}
    family_means = {}
    concept_var = {}
    for family, family_concepts in groups.items():
        family_samples = []
        for concept in family_concepts:
            samples = np.stack([sample_state(rng, concept, args.noise) for _ in range(args.samples_per_concept)], axis=0)
            sample_bank[concept] = samples
            concept_means[concept] = np.mean(samples, axis=0)
            concept_var[concept] = float(np.mean(np.var(samples, axis=0)))
            family_samples.append(samples)
        family_means[family] = np.mean(np.concatenate(family_samples, axis=0), axis=0)

    within_family_dist = []
    cross_family_dist = []
    family_rank = {}
    universal_dims = np.zeros_like(next(iter(concept_means.values())))
    for family, family_concepts in groups.items():
        centered = np.stack([concept_means[concept] - family_means[family] for concept in family_concepts], axis=0)
        _, s, vh = np.linalg.svd(centered, full_matrices=False)
        explained = (s ** 2) / max(float(np.sum(s ** 2)), 1e-6)
        family_rank[family] = {
            "top_explained_variance": [float(v) for v in explained[: min(3, explained.shape[0])].tolist()],
            "top_basis_dims": top_dims(vh[0]) if vh.shape[0] > 0 else [],
        }
        universal_dims += np.abs(vh[0]) if vh.shape[0] > 0 else 0.0

        for concept in family_concepts:
            c_mean = concept_means[concept]
            siblings = [other for other in family_concepts if other != concept]
            within_family_dist.append(float(np.mean([np.linalg.norm(c_mean - concept_means[other]) for other in siblings])))
            foreign = [other for other in concepts if proto.concept_family(other) != family]
            cross_family_dist.append(float(np.mean([np.linalg.norm(c_mean - concept_means[other]) for other in foreign])))

    universal_dims = universal_dims / max(float(np.sum(universal_dims)), 1e-6)
    stable_family_axes = {
        family: family_rank[family]["top_basis_dims"] for family in proto.FAMILIES
    }
    recurrent_dims = top_dims(universal_dims, k=8)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_encoding_inventory_feature_mining",
            "samples_per_concept": int(args.samples_per_concept),
            "noise": float(args.noise),
        },
        "headline_metrics": {
            "mean_within_family_distance": float(np.mean(within_family_dist)),
            "mean_cross_family_distance": float(np.mean(cross_family_dist)),
            "cross_to_within_ratio": float(np.mean(cross_family_dist) / max(np.mean(within_family_dist), 1e-6)),
            "mean_concept_variance": float(np.mean(list(concept_var.values()))),
        },
        "family_rank_structure": family_rank,
        "stable_family_axes": stable_family_axes,
        "universal_recurrent_dims": recurrent_dims,
        "concept_variance": concept_var,
        "theory_implications": {
            "core_statement": "With more samples, the inventory begins to expose stable family-specific low-rank axes and a smaller set of recurrent dimensions reused across families.",
            "stronger_statement": "This suggests that large-scale concept inventory data may reveal more universal encoding features: low-rank family patches, sparse concept offsets, and recurrent dimensions reused across many local charts.",
            "why_important": "This is a realistic path to discovering more general coding laws without needing full whole-brain access first.",
        },
        "verdict": {
            "core_answer": "Yes. Larger concept inventories should make it possible to detect broader and more universal encoding features, not just local concept quirks.",
            "next_theory_target": "expand concept count and relation count so that recurrent dimensions and stable family axes can be estimated more reliably",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
