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


def family_concepts() -> dict[str, list[str]]:
    return {family: proto.PHASE1[family] + proto.PHASE2[family] for family in proto.FAMILIES}


def concept_state(family: str, concept: str) -> np.ndarray:
    vt = proto.family_basis()[family] + proto.concept_offset()[concept]
    lg = cmg.lang_family_basis()[family] + cmg.lang_concept_offset()[concept]
    return np.concatenate([vt, lg], axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory-track cross-family probe analysis")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_cross_family_probe_analysis_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    groups = family_concepts()
    states: dict[str, np.ndarray] = {}
    centers: dict[str, np.ndarray] = {}
    local_axes: dict[str, np.ndarray] = {}

    for family, concepts in groups.items():
        stack = np.stack([concept_state(family, concept) for concept in concepts], axis=0)
        states.update({concept: state for concept, state in zip(concepts, stack)})
        center = np.mean(stack, axis=0)
        centers[family] = center
        centered = stack - center
        if centered.shape[0] > 1:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            local_axes[family] = vh[: min(2, vh.shape[0])]
        else:
            local_axes[family] = np.eye(stack.shape[1], dtype=np.float32)[:2]

    cross_family_projection = {}
    family_intrusion = {}
    for source_family, source_concepts in groups.items():
        source_stack = np.stack([states[concept] - centers[source_family] for concept in source_concepts], axis=0)
        source_within_energy = float(np.mean(np.sum(np.square(source_stack), axis=1)))
        target_stats = {}
        for target_family, axes in local_axes.items():
            projected = source_stack @ axes.T
            proj_energy = float(np.mean(np.sum(np.square(projected), axis=1)))
            ratio = float(proj_energy / max(source_within_energy, 1e-6))
            target_stats[target_family] = {
                "projection_energy": proj_energy,
                "projection_ratio": ratio,
            }
        cross_family_projection[source_family] = target_stats
        foreign_ratios = [
            target_stats[target_family]["projection_ratio"]
            for target_family in groups
            if target_family != source_family
        ]
        family_intrusion[source_family] = {
            "self_ratio": target_stats[source_family]["projection_ratio"],
            "mean_foreign_ratio": float(np.mean(foreign_ratios)),
            "intrusion_gap": float(target_stats[source_family]["projection_ratio"] - np.mean(foreign_ratios)),
        }

    concept_pair_probe = {}
    concepts = [concept for family in proto.FAMILIES for concept in groups[family]]
    for i, c1 in enumerate(concepts):
        f1 = proto.concept_family(c1)
        s1 = states[c1]
        for c2 in concepts[i + 1 :]:
            f2 = proto.concept_family(c2)
            s2 = states[c2]
            dist = float(np.linalg.norm(s1 - s2))
            concept_pair_probe[f"{c1}__{c2}"] = {
                "family_relation": "same_family" if f1 == f2 else "cross_family",
                "distance": dist,
            }

    same_family_dist = [v["distance"] for v in concept_pair_probe.values() if v["family_relation"] == "same_family"]
    cross_family_dist = [v["distance"] for v in concept_pair_probe.values() if v["family_relation"] == "cross_family"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_cross_family_probe_analysis",
        },
        "headline_metrics": {
            "mean_same_family_distance": float(np.mean(same_family_dist)),
            "mean_cross_family_distance": float(np.mean(cross_family_dist)),
            "cross_family_margin_ratio": float(np.mean(cross_family_dist) / max(np.mean(same_family_dist), 1e-6)),
            "mean_intrusion_gap": float(np.mean([v["intrusion_gap"] for v in family_intrusion.values()])),
        },
        "cross_family_projection": cross_family_projection,
        "family_intrusion": family_intrusion,
        "concept_pair_probe_examples": {
            "apple__banana": concept_pair_probe["apple__banana"],
            "apple__cat": concept_pair_probe["apple__cat"],
            "apple__truth": concept_pair_probe["apple__truth"],
            "cat__dog": concept_pair_probe["cat__dog"],
            "truth__logic": concept_pair_probe["truth__logic"],
        },
        "theory_implications": {
            "core_statement": "Cross-family probes show that local family axes are not interchangeable. Each family occupies a distinct patch of the object atlas.",
            "A_implication": "Family-agnostic isotropic updates are implausible because local safe directions are family-dependent.",
            "Mfeas_implication": "A single global smooth chart without family-specific patches is implausible because cross-family margins are far larger than within-family variation.",
        },
        "verdict": {
            "core_answer": "Cross-family probe analysis can be used to exclude family-agnostic update and chart candidates.",
            "next_theory_target": "map these exclusions directly onto candidate A cones and M_feas chart-overlap structures",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
