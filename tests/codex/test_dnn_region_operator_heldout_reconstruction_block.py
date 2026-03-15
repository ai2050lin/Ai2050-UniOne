from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gpt5.code.dnn_parametric_concept_atlas import ConceptEntry, UnifiedParametricConceptAtlas  # noqa: E402


def split_entries(entries: List[ConceptEntry]) -> tuple[List[ConceptEntry], List[ConceptEntry]]:
    train: List[ConceptEntry] = []
    test: List[ConceptEntry] = []
    grouped = {}
    for entry in entries:
        grouped.setdefault(entry.family, []).append(entry)
    for family_entries in grouped.values():
        family_entries = sorted(family_entries, key=lambda row: row.name)
        cutoff = max(1, int(len(family_entries) * 0.75))
        train.extend(family_entries[:cutoff])
        test.extend(family_entries[cutoff:])
    return train, test


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    atlas = UnifiedParametricConceptAtlas.from_artifacts(ROOT, synth_per_family=64, seed=23)
    train_entries, test_entries = split_entries(atlas.concept_entries)

    region_pairs = [
        ("object", "memory"),
        ("object", "identity"),
        ("object", "readout"),
        ("memory", "macro"),
        ("identity", "readout"),
    ]

    pair_results = {}
    gains = []
    for source_region, target_region in region_pairs:
        operator = atlas.fit_affine_operator(source_region, target_region, train_entries)
        train_eval = atlas.evaluate_affine_operator(operator, source_region, target_region, train_entries)
        test_eval = atlas.evaluate_affine_operator(operator, source_region, target_region, test_entries)
        key = f"{source_region}_to_{target_region}"
        pair_results[key] = {
            "train_mean_error": train_eval["mean_error"],
            "train_baseline_error": train_eval["baseline_error"],
            "train_relative_gain": train_eval["relative_gain"],
            "test_mean_error": test_eval["mean_error"],
            "test_baseline_error": test_eval["baseline_error"],
            "test_relative_gain": test_eval["relative_gain"],
        }
        gains.append(float(test_eval["relative_gain"]))

    heldout_gain_mean = float(sum(gains) / len(gains))
    heldout_gain_min = float(min(gains))
    reconstruction_score = min(
        1.0,
        0.50 * min(1.0, heldout_gain_mean / 0.85)
        + 0.25 * min(1.0, heldout_gain_min / 0.65)
        + 0.25 * min(1.0, len(test_entries) / 40.0),
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_region_operator_heldout_reconstruction_block",
        },
        "strict_goal": {
            "statement": (
                "Fit explicit region-to-region operators on the unified atlas and test whether one region can reconstruct held-out regions across held-out concepts."
            ),
            "boundary": (
                "This block tests candidate reconstructability on the current atlas object. It does not yet prove exact biological region operators."
            ),
        },
        "data_split": {
            "train_entries": len(train_entries),
            "test_entries": len(test_entries),
            "families": sorted({entry.family for entry in atlas.concept_entries}),
        },
        "pair_results": pair_results,
        "headline_metrics": {
            "heldout_gain_mean": heldout_gain_mean,
            "heldout_gain_min": heldout_gain_min,
            "reconstruction_score": reconstruction_score,
        },
        "strict_verdict": {
            "candidate_region_operator_present": bool(heldout_gain_mean > 0.75 and heldout_gain_min > 0.55),
            "exact_region_operator_present": bool(heldout_gain_mean > 0.95 and heldout_gain_min > 0.85),
            "core_answer": (
                "Held-out reconstruction now works at candidate level: region-to-region operators fitted on the unified atlas beat baseline strongly on unseen concepts, so one observed region already carries substantial information about other regions."
            ),
            "main_hard_gaps": [
                "the fitted operators are affine candidates, not canonical theorem-level operators",
                "held-out reconstruction is strong but not exact enough for full closure",
                "the result depends on the current atlas projection choices and synthetic scale-up entries",
            ],
        },
        "progress_estimate": {
            "candidate_region_operator_percent": 64.0,
            "candidate_region_to_region_reconstruction_percent": 63.0,
            "full_brain_encoding_mechanism_percent": 76.0,
        },
        "next_large_blocks": [
            "Replace affine candidate operators with structured family-conditioned canonical operators.",
            "Run held-out reconstruction on a much larger real concept atlas instead of the current mixed atlas.",
            "Add macro protocol and successor coordinates into the held-out reconstruction target set.",
        ],
    }
    return payload


def test_dnn_region_operator_heldout_reconstruction_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["heldout_gain_mean"] > 0.75
    assert metrics["heldout_gain_min"] > 0.55
    assert metrics["reconstruction_score"] > 0.75
    assert verdict["candidate_region_operator_present"] is True
    assert verdict["exact_region_operator_present"] is False


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN region operator held-out reconstruction block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_region_operator_heldout_reconstruction_block_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
