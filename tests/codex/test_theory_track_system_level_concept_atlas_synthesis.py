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
    ap = argparse.ArgumentParser(description="Theory-track system-level concept atlas synthesis")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_system_level_concept_atlas_synthesis_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    apple = load("theory_track_apple_concept_encoding_analysis_20260312.json")
    atlas = load("theory_track_concept_family_atlas_analysis_20260312.json")
    explicit = load("theory_track_explicit_A_Mfeas_formalization_20260312.json")

    fruit_radius = float(atlas["family_atlas"]["fruit"]["family_radius"])
    animal_radius = float(atlas["family_atlas"]["animal"]["family_radius"])
    abstract_radius = float(atlas["family_atlas"]["abstract"]["family_radius"])
    atlas_separation = float(atlas["headline_metrics"]["atlas_separation_score"])

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_system_level_concept_atlas_synthesis",
        },
        "from_local_probe_to_global_atlas": {
            "local_probe": "apple exposes one local chart of the fruit manifold",
            "family_patch": "apple/banana/pear expose a fruit atlas patch with shared basis plus local offsets",
            "multi_family_atlas": "fruit, animal, and abstract concepts expose multiple atlas patches and cross-family boundaries",
            "system_result": "stacking many concept probes makes it possible to reconstruct an object-manifold atlas rather than isolated concept facts",
        },
        "how_concept_analysis_reconstructs_the_seven_questions": {
            "Q1_encoding_object_layer": "concept-family atlas directly reconstructs object kernels, family bases, and concept offsets",
            "Q2_local_update_law": "tracking how concepts shift under new evidence reveals which local update directions are safe",
            "Q3_write_read_separation": "stable concept identity under new writes reveals whether write, read, and persistence states are separated",
            "Q4_bridge_role_kernel": "placing many concepts into relational contexts exposes bridge and role coordinates that sit above local object charts",
            "Q5_crossmodal_consistency": "each concept across vision, touch, and language constrains whether a shared object manifold exists",
            "Q6_discriminative_geometry": "family neighborhoods and cross-family margins reveal how object charts are read into decision geometry",
            "Q7_brain_mapping_3d": "once atlas patches are stable, they become probes for cortical region mapping, topology prediction, and falsification",
        },
        "mapping_to_explicit_formalism": {
            "Z_obj": "reconstructed from multi-concept family atlas patches",
            "A": explicit["explicit_A"]["high_level_form"],
            "M_feas": explicit["explicit_M_feas"]["high_level_form"],
            "main_constraint": "concept probes constrain which local charts and overlaps are real, and therefore shrink the candidate space of A and M_feas",
        },
        "system_level_reconstruction_hypothesis": {
            "formation_process": [
                "Many concrete concepts produce many local atlas patches.",
                "Family neighbors reveal shared bases and concept offsets.",
                "Cross-modal views tie these patches onto a shared object manifold.",
                "Novel-learning stress reveals admissible update constraints A.",
                "Readout and switching behavior reveal which chart overlaps belong to M_feas.",
            ],
            "runtime_mechanism": [
                "Inputs first land on a local object chart.",
                "Local plasticity proposes an update direction.",
                "Only updates inside A are allowed.",
                "The trajectory must remain inside M_feas while moving between object, memory, relation, readout, and phase charts.",
            ],
        },
        "why_this_is_the_most_feasible_route": {
            "engineering_reason": "concept-centered probes are easier to scale than direct whole-brain reverse engineering",
            "mathematical_reason": "local charts are observable earlier than the full global viability law, so they provide a realistic path to reconstruct the larger theory",
            "project_reason": "this route connects deep-network evidence and brain evidence through the same intermediate object-manifold atlas",
        },
        "atlas_metrics": {
            "fruit_radius": fruit_radius,
            "animal_radius": animal_radius,
            "abstract_radius": abstract_radius,
            "atlas_separation_score": atlas_separation,
        },
        "verdict": {
            "core_answer": "Yes. After apple analysis, the next scalable step is systematic multi-concept atlas analysis. That can grow from local concept probes into a system-level reconstruction route for the full encoding mechanism.",
            "stronger_answer": "This does not immediately solve the theory, but it is the most feasible route because it turns the hidden global mechanism into a sequence of observable local chart, overlap, and admissibility constraints.",
            "next_theory_target": "expand from three family patches to a denser concept atlas and use it to falsify candidate A and M_feas structures",
        },
        "prior_support": {
            "apple_local_probe_answer": apple["verdict"]["core_answer"],
            "atlas_patch_support": atlas["verdict"]["core_answer"],
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
