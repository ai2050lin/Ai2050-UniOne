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


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory-track apple concept encoding analysis")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_apple_concept_encoding_analysis_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    family_basis = proto.family_basis()
    concept_offset = proto.concept_offset()
    lang_family = cmg.lang_family_basis()
    lang_offset = cmg.lang_concept_offset()

    apple_family = proto.concept_family("apple")
    apple_visual_tactile = family_basis[apple_family] + concept_offset["apple"]
    apple_language = lang_family[apple_family] + lang_offset["apple"]

    apple_visual = apple_visual_tactile[:8]
    apple_tactile = apple_visual_tactile[8:]
    fruit_family_visual = family_basis["fruit"][:8]
    fruit_family_tactile = family_basis["fruit"][8:]

    banana_visual_tactile = family_basis["fruit"] + concept_offset["banana"]
    pear_visual_tactile = family_basis["fruit"] + concept_offset["pear"]

    apple_specific_offset = concept_offset["apple"]
    within_family_delta_banana = apple_visual_tactile - banana_visual_tactile
    within_family_delta_pear = apple_visual_tactile - pear_visual_tactile

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_apple_concept_encoding_analysis",
        },
        "core_question": {
            "question": "Can a single object concept such as apple help reconstruct the internal encoding mechanism?",
            "short_answer": "Yes, but only as a local chart into the full mechanism, not as the whole mechanism by itself.",
        },
        "apple_decomposition": {
            "family": "fruit",
            "family_basis_visual": arr(fruit_family_visual),
            "family_basis_tactile": arr(fruit_family_tactile),
            "apple_specific_offset_visual_tactile": arr(apple_specific_offset),
            "apple_language_family_basis": arr(lang_family["fruit"]),
            "apple_language_specific_offset": arr(lang_offset["apple"]),
            "apple_visual_state": arr(apple_visual),
            "apple_tactile_state": arr(apple_tactile),
            "apple_language_state": arr(apple_language),
        },
        "what_apple_reveals": {
            "object_kernel": "apple shows that an object concept can be decomposed into family basis plus concept-specific offset",
            "crossmodal_binding": "apple shows how visual, tactile, and language-like states can point to the same object identity",
            "family_structure": "apple/banana/pear reveal a family manifold for fruit with local concept offsets",
            "attribute_structure": "apple-like attributes such as roundness, redness, sweetness, and edibility are not independent labels; they are likely local axes on the fruit object chart",
            "relation_structure": "apple can enter relation slots such as object-of-eating, object-in-basket, or compare-with-pear, exposing bridge and role structure",
        },
        "limits_of_apple_only_analysis": {
            "why_not_enough": [
                "A single concept only reveals one local object chart.",
                "It cannot by itself recover admissible-update constraints across many concepts.",
                "It cannot by itself recover discriminative geometry or phase switching for the whole system.",
            ],
            "best_use": "apple is best treated as a microscope into one local region of the encoding manifold, then compared against nearby family members and cross-family contrasts.",
        },
        "how_to_use_apple_for_theory_track": {
            "step_1": "Decompose apple into family basis + concept offset + modality-specific views.",
            "step_2": "Compare apple with banana and pear to infer local family manifold coordinates.",
            "step_3": "Attach attributes such as color, shape, taste, and affordance to see whether they behave like local axes or separate symbolic tags.",
            "step_4": "Place apple into relational contexts to identify bridge-role operators.",
            "step_5": "Track how apple survives novel learning pressure to expose admissible-update and retention constraints.",
        },
        "mapping_to_seven_questions": {
            "Q1_encoding_object_layer": "apple reveals object kernel plus family basis",
            "Q2_local_update_law": "how apple changes under new evidence reveals local update law",
            "Q3_write_read_separation": "stable apple identity under new writes reveals write/read separation",
            "Q4_bridge_role_kernel": "apple in relations reveals bridge and role structure",
            "Q5_crossmodal_consistency": "apple across vision, touch, language reveals shared object manifold",
            "Q6_discriminative_geometry": "apple versus pear/banana reveals how object geometry becomes decision geometry",
            "Q7_brain_mapping_3d": "apple-related cortical patterns can later be used as a brain-side probe once the abstract mechanism is clearer",
        },
        "theory_answer": {
            "core_statement": "Concept-centered analysis is useful because each concrete concept exposes a local chart of the encoding manifold. Apple is a valid entry point, but only if analyzed together with attributes, family neighbors, cross-modal views, and relational roles.",
            "stronger_statement": "If the unsolved points truly come from the same encoding mechanism, then analyzing apple and its neighborhood can help recover local structure of Z_obj, but the full theory still requires A and M_feas across many concepts and regimes.",
        },
        "verdict": {
            "core_answer": "Yes, apple-centered analysis can help reconstruct the encoding mechanism, but only as a local manifold probe. It is a powerful microscope, not the whole atlas.",
            "next_theory_target": "build concept-centered local chart analysis for apple/banana/pear and then generalize to cross-family probes",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
