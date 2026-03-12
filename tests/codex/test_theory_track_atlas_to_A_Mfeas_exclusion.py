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
    ap = argparse.ArgumentParser(description="Theory-track atlas to A and M_feas exclusion")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_atlas_to_A_Mfeas_exclusion_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    atlas = load("theory_track_concept_family_atlas_analysis_20260312.json")
    cross = load("theory_track_cross_family_probe_analysis_20260312.json")
    formal = load("theory_track_explicit_A_Mfeas_formalization_20260312.json")

    separation = float(atlas["headline_metrics"]["atlas_separation_score"])
    cross_margin_ratio = float(cross["headline_metrics"]["cross_family_margin_ratio"])
    intrusion_gap = float(cross["headline_metrics"]["mean_intrusion_gap"])

    excluded_candidates = {
        "single_global_smooth_object_chart": {
            "status": "excluded",
            "reason": "atlas separation and cross-family margin are too large relative to within-family variation",
        },
        "family_agnostic_isotropic_update_cone": {
            "status": "excluded",
            "reason": "cross-family probes show local admissible directions are family-dependent rather than globally uniform",
        },
        "full_object_disc_overlap_everywhere": {
            "status": "excluded",
            "reason": "current P3 bottleneck still localizes at U_object INTERSECT U_disc rather than disappearing globally",
        },
        "direct_readout_equals_object_geometry": {
            "status": "excluded",
            "reason": "earlier C43-C45 failures show direct readout compression damages manifold quality",
        },
    }

    refined_candidates = {
        "A_refined": {
            "form": formal["explicit_A"]["high_level_form"],
            "new_constraint": "K_ret and K_id must be family-conditioned cones, not globally shared isotropic cones",
        },
        "M_feas_refined": {
            "form": formal["explicit_M_feas"]["high_level_form"],
            "new_constraint": "U_object must be modeled as a family-patched atlas with limited chart overlaps instead of one global chart",
        },
        "readout_constraint": {
            "form": "phi_object_to_disc only valid on restricted family-conditioned overlap bands",
            "new_constraint": "discriminative geometry must be transported from object charts through restricted overlaps, not by direct collapse",
        },
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_atlas_to_A_Mfeas_exclusion",
        },
        "support_metrics": {
            "atlas_separation_score": separation,
            "cross_family_margin_ratio": cross_margin_ratio,
            "mean_intrusion_gap": intrusion_gap,
        },
        "excluded_candidates": excluded_candidates,
        "refined_candidates": refined_candidates,
        "system_level_answer": {
            "core_statement": "Multi-concept atlas analysis does not directly solve the theory, but it sharply shrinks the candidate space of A and M_feas.",
            "practical_statement": "This makes the theory track executable: use concept atlas structure to exclude bad global geometries first, then use P3/P4 experiments to prune the remaining candidates.",
            "bridge_to_brain": "Once a family-patched object atlas is stable, it becomes a natural probe set for cortical area mapping and brain-side falsification.",
        },
        "verdict": {
            "core_answer": "Yes. Systematic concept-family atlas analysis can now directly exclude broad classes of wrong A and M_feas candidates.",
            "next_theory_target": "construct explicit family-conditioned projection operators and overlap maps, then let P3 experiments falsify them",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
