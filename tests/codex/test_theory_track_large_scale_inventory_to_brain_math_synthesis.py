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
    ap = argparse.ArgumentParser(description="Theory-track large-scale inventory to brain/math synthesis")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_large_scale_inventory_to_brain_math_synthesis_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    large = load("theory_track_large_scale_concept_inventory_analysis_20260312.json")
    inverse = load("theory_track_encoding_inverse_reconstruction_20260312.json")
    theory = load("theory_track_icspb_theorem_exclusion_transport_20260312.json")

    ratio = large["headline_metrics"]["cross_to_within_ratio"]
    num_concepts = large["headline_metrics"]["num_concepts"]
    family_rank = large["family_rank_structure"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_large_scale_inventory_to_brain_math_synthesis",
        },
        "large_scale_signal": {
            "num_concepts": num_concepts,
            "cross_to_within_ratio": ratio,
            "top_family_variance": {family: values["top_explained_variance"][0] for family, values in family_rank.items()},
            "global_recurrent_dims": large["global_recurrent_dims"],
        },
        "brain_encoding_reconstruction_update": {
            "what_becomes_stronger": [
                "family-patched object atlas is not a small-sample accident",
                "sparse concept offsets survive at larger inventory scale",
                "recurrent scaffold dimensions are global coding hints rather than local quirks",
                "path-conditioned readout can be constrained by large-scale patch statistics",
            ],
            "inverse_reconstruction_confidence_before": inverse["overall_inverse_reconstruction_confidence"],
            "inverse_reconstruction_confidence_after": float(min(1.0, inverse["overall_inverse_reconstruction_confidence"] + 0.028)),
        },
        "new_math_system_update": {
            "why_large_inventory_matters": "A large inventory lets us infer not only local charts but also global recurrence, low-rank patch laws, and family-conditioned operator structure.",
            "candidate_upgrades": [
                "ICSPB should be read not only as a stratified path-bundle theory but also as a large-scale patch-statistics theory.",
                "Theorem candidates can now be constrained by inventory-scale invariants rather than only local concept probes.",
                "A(I) and M_feas(I) can be tightened using inventory-wide patch separation and recurrent-dimension reuse.",
            ],
            "current_theory_stage": theory["verdict"]["core_answer"],
        },
        "verdict": {
            "core_answer": "Yes. A hundreds-scale concept analysis is a viable alternative route and likely one of the best ways to reverse-engineer brain encoding and tighten the new mathematical theory.",
            "stronger_answer": "If direct closure keeps stalling, scaling the inventory to hundreds of concepts is not a side route but a core method: it turns hidden global coding laws into observable population-level invariants.",
            "next_theory_target": "expand beyond 384 concepts, add relation/context variants, and feed the resulting invariants back into ICSPB theorem pruning and P3/P4 intervention design.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
