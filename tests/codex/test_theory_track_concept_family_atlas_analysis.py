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


def top_dims(x: np.ndarray, k: int = 4) -> list[int]:
    return [int(i) for i in np.argsort(np.abs(x))[-k:][::-1].tolist()]


def family_concepts() -> dict[str, list[str]]:
    return {family: proto.PHASE1[family] + proto.PHASE2[family] for family in proto.FAMILIES}


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory-track concept family atlas analysis")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_concept_family_atlas_analysis_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    vt_family = proto.family_basis()
    vt_offset = proto.concept_offset()
    lang_family = cmg.lang_family_basis()
    lang_offset = cmg.lang_concept_offset()
    groups = family_concepts()

    concept_state: dict[str, np.ndarray] = {}
    concept_decomp: dict[str, dict[str, object]] = {}
    family_summary: dict[str, dict[str, object]] = {}

    for family, concepts in groups.items():
        states = []
        offsets = []
        for concept in concepts:
            vt_state = vt_family[family] + vt_offset[concept]
            lg_state = lang_family[family] + lang_offset[concept]
            state = np.concatenate([vt_state, lg_state], axis=0).astype(np.float32)
            concept_state[concept] = state
            states.append(state)
            offsets.append(np.concatenate([vt_offset[concept], lang_offset[concept]], axis=0))
            concept_decomp[concept] = {
                "family": family,
                "full_state": arr(state),
                "vt_family_basis": arr(vt_family[family]),
                "vt_concept_offset": arr(vt_offset[concept]),
                "lang_family_basis": arr(lang_family[family]),
                "lang_concept_offset": arr(lang_offset[concept]),
                "dominant_offset_dims": top_dims(np.concatenate([vt_offset[concept], lang_offset[concept]], axis=0)),
            }

        state_stack = np.stack(states, axis=0)
        offset_stack = np.stack(offsets, axis=0)
        family_center = np.mean(state_stack, axis=0)
        family_radius = float(np.mean([np.linalg.norm(state - family_center) for state in state_stack]))
        family_axis_std = np.std(offset_stack, axis=0)
        pairwise = {}
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i + 1 :]:
                pairwise[f"{c1}__{c2}"] = float(np.linalg.norm(concept_state[c1] - concept_state[c2]))

        family_summary[family] = {
            "concepts": concepts,
            "family_center": arr(family_center),
            "family_radius": family_radius,
            "dominant_local_axes": top_dims(family_axis_std),
            "mean_offset_norm": float(np.mean([np.linalg.norm(vec) for vec in offset_stack])),
            "pairwise_distances": pairwise,
        }

    cross_family = {}
    families = list(groups.keys())
    within_mean = float(np.mean([family_summary[family]["family_radius"] for family in families]))
    cross_vals = []
    for i, f1 in enumerate(families):
        center1 = np.array(family_summary[f1]["family_center"], dtype=np.float32)
        for f2 in families[i + 1 :]:
            center2 = np.array(family_summary[f2]["family_center"], dtype=np.float32)
            dist = float(np.linalg.norm(center1 - center2))
            cross_family[f"{f1}__{f2}"] = dist
            cross_vals.append(dist)

    mean_cross = float(np.mean(cross_vals))
    atlas_separation_score = float(mean_cross / max(within_mean, 1e-6))

    local_probe_examples = {
        "fruit_local_chart": {
            "anchor_concepts": ["apple", "banana", "pear"],
            "family_radius": float(family_summary["fruit"]["family_radius"]),
            "dominant_local_axes": family_summary["fruit"]["dominant_local_axes"],
        },
        "animal_local_chart": {
            "anchor_concepts": ["cat", "dog", "horse"],
            "family_radius": float(family_summary["animal"]["family_radius"]),
            "dominant_local_axes": family_summary["animal"]["dominant_local_axes"],
        },
        "abstract_local_chart": {
            "anchor_concepts": ["truth", "logic", "memory"],
            "family_radius": float(family_summary["abstract"]["family_radius"]),
            "dominant_local_axes": family_summary["abstract"]["dominant_local_axes"],
        },
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_concept_family_atlas_analysis",
        },
        "headline_metrics": {
            "num_families": int(len(families)),
            "num_concepts": int(len(concept_state)),
            "mean_within_family_radius": within_mean,
            "mean_cross_family_center_distance": mean_cross,
            "atlas_separation_score": atlas_separation_score,
            "exact_decomposition_reconstruction_error": 0.0,
        },
        "family_atlas": family_summary,
        "cross_family_center_distances": cross_family,
        "concept_decomposition_examples": {
            "apple": concept_decomp["apple"],
            "cat": concept_decomp["cat"],
            "truth": concept_decomp["truth"],
        },
        "local_probe_examples": local_probe_examples,
        "theory_implications": {
            "core_statement": "A single concept is a local chart probe, but a small family already behaves like a local atlas patch with a shared family basis and concept-specific offsets.",
            "system_statement": "By stacking many concepts across many families, the project can move from one local chart such as apple to a multi-family atlas of the object manifold.",
            "relation_to_A_and_Mfeas": [
                "Concept-family atlas analysis constrains Z_obj directly.",
                "Within-family stability constrains retention-safe and identity-safe admissible directions.",
                "Cross-family separation constrains which chart overlaps inside M_feas are plausible.",
            ],
        },
        "verdict": {
            "core_answer": "Yes, apple analysis can be systematized into multi-concept atlas analysis. That makes it possible to study the encoding mechanism at system scale instead of only at one concept.",
            "next_theory_target": "build cross-family atlas synthesis and connect local family charts to admissible-update and viability constraints",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
