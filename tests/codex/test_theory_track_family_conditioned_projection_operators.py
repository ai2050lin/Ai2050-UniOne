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
    ap = argparse.ArgumentParser(description="Theory-track family-conditioned projection operators")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_family_conditioned_projection_operators_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    atlas = load("theory_track_concept_family_atlas_analysis_20260312.json")
    cross = load("theory_track_cross_family_probe_analysis_20260312.json")

    families = list(atlas["family_atlas"].keys())
    operator_family = {}
    for family in families:
        family_patch = atlas["family_atlas"][family]
        local_axes = family_patch["dominant_local_axes"]
        intrusion = cross["family_intrusion"][family]
        operator_family[family] = {
            "P_obj_family": {
                "support_dims": local_axes,
                "meaning": "family-local object chart projector",
            },
            "P_mem_family": {
                "support_dims": local_axes[:2],
                "meaning": "retention-protective memory projector anchored to stable local family axes",
            },
            "P_id_family": {
                "support_dims": local_axes[:3],
                "meaning": "same-object identity projector restricted to family-conditioned object directions",
            },
            "P_disc_family": {
                "support_dims": local_axes[-2:],
                "meaning": "readout-facing projector; narrower than P_obj to avoid direct collapse",
            },
            "family_radius": family_patch["family_radius"],
            "intrusion_gap": intrusion["intrusion_gap"],
        }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_family_conditioned_projection_operators",
        },
        "core_operators": operator_family,
        "operator_laws": {
            "family_conditioning": "Projection operators should be indexed by family patch rather than globally shared.",
            "nested_structure": "P_disc_family should be narrower than P_obj_family, while P_mem_family should prioritize stable family axes.",
            "anti_isotropy": "Large intrusion gaps imply globally isotropic projectors are implausible.",
        },
        "formal_candidates": {
            "K_ret_refined": "||P_mem_family Delta|| <= tau_mem_family and retention(z + Delta) >= tau_ret_floor",
            "K_id_refined": "||P_id_family Delta|| <= alpha_id ||P_obj_family Delta|| + epsilon_id",
            "K_read_refined": "||P_disc_family Delta|| <= beta_read ||P_obj_family Delta|| + gamma_read",
        },
        "verdict": {
            "core_answer": "Projection operators can now be refined from global placeholders into family-conditioned operator families.",
            "next_theory_target": "attach these operators to restricted overlap maps and then use P3 failures to falsify bad operator-overlap pairings",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
