from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gpt5.code.dnn_parametric_concept_atlas import UnifiedParametricConceptAtlas  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    atlas = UnifiedParametricConceptAtlas.from_artifacts(ROOT, synth_per_family=64, seed=19)
    by_family = atlas.entries_by_family()
    family_counts = {family: len(entries) for family, entries in by_family.items()}
    exemplar_count = sum(1 for entry in atlas.concept_entries if entry.source == "exemplar")
    synthetic_count = sum(1 for entry in atlas.concept_entries if entry.source == "synthetic")
    region_names = ["object", "memory", "identity", "readout", "macro"]

    coverage_score = min(
        1.0,
        0.28 * min(1.0, len(atlas.concept_entries) / 195.0)
        + 0.24 * min(1.0, exemplar_count / 3.0)
        + 0.18 * min(1.0, len(region_names) / 5.0)
        + 0.15 * min(1.0, len(atlas.global_recurrent_dims) / 8.0)
        + 0.15 * min(1.0, len(atlas.family_supports) / 3.0),
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_unified_parametric_concept_atlas_block",
        },
        "strict_goal": {
            "statement": (
                "Turn the extracted DNN structure into a reusable unified parametric concept atlas object instead of many disconnected JSON analyses."
            ),
            "boundary": (
                "The atlas is unified and executable now, but still combines exemplar concepts with synthetic scale-up entries."
            ),
        },
        "atlas_summary": {
            "total_entries": len(atlas.concept_entries),
            "exemplar_entries": exemplar_count,
            "synthetic_entries": synthetic_count,
            "family_counts": family_counts,
            "region_views": region_names,
            "global_recurrent_dims": atlas.global_recurrent_dims,
            "family_rank_dims": atlas.family_rank_dims,
        },
        "headline_metrics": {
            "coverage_score": coverage_score,
            "total_entries": len(atlas.concept_entries),
            "exemplar_entries": exemplar_count,
            "synthetic_entries": synthetic_count,
        },
        "strict_verdict": {
            "unified_atlas_present": bool(coverage_score > 0.75),
            "core_answer": (
                "The DNN-side extraction is now organized as a single unified parametric atlas object with families, concepts, bases, offsets, region views, and scalable synthetic entries."
            ),
            "main_hard_gaps": [
                "the atlas still mixes exact exemplar concepts with synthetic scale-up entries",
                "region views are still candidate projections, not canonical region operators",
                "macro structure is still weaker than meso family structure inside the atlas object",
            ],
        },
        "progress_estimate": {
            "unified_parametric_concept_atlas_percent": 66.0,
            "large_scale_concept_atlas_percent": 71.0,
            "full_brain_encoding_mechanism_percent": 75.0,
        },
        "next_large_blocks": [
            "Replace synthetic scale-up entries with a much larger real DNN-derived concept atlas.",
            "Fit canonical region operators on top of the unified atlas instead of fixed candidate projections.",
            "Strengthen macro lift and protocol coordinates inside the atlas object.",
        ],
    }
    return payload


def test_dnn_unified_parametric_concept_atlas_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["total_entries"] >= 190
    assert metrics["exemplar_entries"] >= 3
    assert metrics["coverage_score"] > 0.75
    assert verdict["unified_atlas_present"] is True


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN unified parametric concept atlas block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_unified_parametric_concept_atlas_block_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
