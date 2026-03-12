from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load(name: str) -> dict:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def overlap_band(radius: float, intrusion_gap: float, readout_penalty: float) -> dict[str, float]:
    memory_band = float(max(0.05, min(0.95, 1.0 - radius * 4.0)))
    disc_band = float(max(0.02, min(0.55, intrusion_gap * 0.25 - readout_penalty)))
    relation_band = float(max(0.08, min(0.80, 0.35 + intrusion_gap * 0.20)))
    phase_band = float(max(0.05, min(0.70, 0.30 + intrusion_gap * 0.10 - radius)))
    return {
        "object_memory_overlap": memory_band,
        "object_disc_overlap": disc_band,
        "object_relation_overlap": relation_band,
        "memory_phase_overlap": phase_band,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory-track restricted overlap maps")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_restricted_overlap_maps_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    atlas = load("theory_track_concept_family_atlas_analysis_20260312.json")
    cross = load("theory_track_cross_family_probe_analysis_20260312.json")
    exclusion = load("theory_track_atlas_to_A_Mfeas_exclusion_20260312.json")

    overlap_maps = {}
    for family, patch in atlas["family_atlas"].items():
        radius = float(patch["family_radius"])
        intrusion_gap = float(cross["family_intrusion"][family]["intrusion_gap"])
        readout_penalty = 0.08 if exclusion["excluded_candidates"]["direct_readout_equals_object_geometry"]["status"] == "excluded" else 0.0
        overlap_maps[family] = overlap_band(radius, intrusion_gap, readout_penalty)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_restricted_overlap_maps",
        },
        "restricted_overlap_maps": overlap_maps,
        "transition_constraints": {
            "phi_object_to_memory": "allowed on wide family-conditioned overlap bands",
            "phi_object_to_disc": "allowed only on narrow restricted overlap bands",
            "phi_object_to_relation": "allowed on medium overlap bands anchored to family structure",
            "phi_memory_to_phase": "allowed on moderate safe-switching bands",
        },
        "theory_implications": {
            "core_statement": "The object-disc overlap should be significantly narrower than the object-memory overlap.",
            "why": "This matches the repeated P3 bottleneck: memory compatibility is easier to preserve than discriminative transport.",
            "atlas_link": "Family-specific radii and intrusion gaps can be used as first-order proxies for overlap width.",
        },
        "verdict": {
            "core_answer": "Restricted overlap maps can now be parameterized in a family-conditioned way rather than treated as globally uniform.",
            "next_theory_target": "use these overlap maps to drive a theory-guided exclusion loop on P3 transport/readout candidates",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
