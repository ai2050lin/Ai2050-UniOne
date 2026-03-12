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
    ap = argparse.ArgumentParser(description="Theory-track conscious modality unification clue")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_conscious_modality_unification_clue_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    shared_mod = load("parameterized_shared_modality_law_20260310.json")
    central = load("shared_central_loop_modality_hypothesis_20260310.json")
    axioms = load("theory_track_icspb_axiom_layer_20260312.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_conscious_modality_unification_clue",
        },
        "empirical_clue": {
            "human_level_observation": "Conscious reasoning can operate over visual, tactile, language, and other modality inputs in one analyzable stream.",
            "parameterized_shared_gap": shared_mod["parameterized_shared_law"]["mean_held_out_gap"],
            "fully_shared_gap": shared_mod["fully_shared_law"]["mean_held_out_gap"],
            "central_loop_gap": central["shared_central_loop_law"]["mean_held_out_gap"],
            "oracle_gap": shared_mod["modality_separate_oracle"]["mean_held_out_gap"],
        },
        "core_inference": {
            "supported": "there is some shared low-dimensional reasoning-compatible structure across modalities",
            "not_supported": "one fully shared global central loop is sufficient",
            "more_plausible_form": "modality-conditioned entry into a family-conditioned shared reasoning slice",
        },
        "relation_to_icspb": {
            "linked_axioms": [
                "A1_family_stratification",
                "A2_section_based_concepts",
                "A3_attached_fibers",
                "A6_path_conditioned_computation",
            ],
            "interpretation": "Conscious unification is better read as a path-conditioned reasoning slice inside ICSPB rather than a single universal processing core.",
        },
        "verdict": {
            "core_answer": "The consciousness clue supports a shared reasoning-compatible substrate, but only after modality-conditioned and family-conditioned lifting.",
            "next_theory_target": "formalize a modality-unified reasoning law inside ICSPB.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
