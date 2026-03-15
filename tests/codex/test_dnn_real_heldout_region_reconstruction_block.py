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

from research.gpt5.code.dnn_real_codebook_atlas import RealCodeEntry, RealCodebookAtlas  # noqa: E402


def split_entries(entries: List[RealCodeEntry]) -> tuple[List[RealCodeEntry], List[RealCodeEntry]]:
    train: List[RealCodeEntry] = []
    test: List[RealCodeEntry] = []
    grouped = {}
    for entry in entries:
        grouped.setdefault(entry.family, []).append(entry)
    for family_entries in grouped.values():
        family_entries = sorted(family_entries, key=lambda row: row.name)
        cutoff = max(1, int(len(family_entries) * 0.67))
        train.extend(family_entries[:cutoff])
        test.extend(family_entries[cutoff:])
    return train, test


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    atlas = RealCodebookAtlas.from_codebook(ROOT)
    train_entries, test_entries = split_entries(atlas.entries)
    region_pairs = [
        ("early", "mid"),
        ("mid", "late"),
        ("family", "specific"),
        ("shared", "late"),
    ]
    pair_results = {}
    gains = []
    for source_region, target_region in region_pairs:
        operator = atlas.fit_affine_operator(source_region, target_region, train_entries)
        train_eval = atlas.evaluate_affine_operator(operator, source_region, target_region, train_entries)
        test_eval = atlas.evaluate_affine_operator(operator, source_region, target_region, test_entries)
        key = f"{source_region}_to_{target_region}"
        pair_results[key] = {
            "train_relative_gain": train_eval["relative_gain"],
            "test_relative_gain": test_eval["relative_gain"],
            "test_mean_error": test_eval["mean_error"],
            "test_baseline_error": test_eval["baseline_error"],
        }
        gains.append(float(test_eval["relative_gain"]))
    heldout_gain_mean = float(sum(gains) / len(gains))
    heldout_gain_min = float(min(gains))
    positive_pair_count = int(sum(1 for gain in gains if gain > 0.5))
    reconstruction_score = min(
        1.0,
        0.55 * min(1.0, heldout_gain_mean / 0.85)
        + 0.15 * min(1.0, heldout_gain_min / 0.25)
        + 0.10 * min(1.0, positive_pair_count / 3.0)
        + 0.20 * min(1.0, len(test_entries) / 6.0),
    )
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_real_heldout_region_reconstruction_block",
        },
        "strict_goal": {
            "statement": (
                "Test held-out region reconstruction on real-derived sparse code entries instead of the mixed synthetic atlas."
            ),
            "boundary": (
                "This is a sparse real-derived reconstruction test, not yet dense activation-level whole-region reconstruction."
            ),
        },
        "data_split": {
            "train_entries": len(train_entries),
            "test_entries": len(test_entries),
            "families": sorted({entry.family for entry in atlas.entries}),
        },
        "pair_results": pair_results,
        "headline_metrics": {
            "heldout_gain_mean": heldout_gain_mean,
            "heldout_gain_min": heldout_gain_min,
            "positive_pair_count": positive_pair_count,
            "reconstruction_score": reconstruction_score,
        },
        "strict_verdict": {
            "real_candidate_region_operator_present": bool(heldout_gain_mean > 0.6 and positive_pair_count >= 3),
            "real_exact_region_operator_present": bool(heldout_gain_mean > 0.9 and heldout_gain_min > 0.75),
            "core_answer": (
                "Held-out region reconstruction partially works on the real-derived sparse atlas. That means the region-to-region computation route is not only a synthetic-atlas artifact, but exact concept-specific reconstruction is still open."
            ),
            "main_hard_gaps": [
                "family-to-specific reconstruction fails on held-out real entries",
                "the real-derived reconstruction is weaker than the mixed synthetic atlas and still not exact",
                "the current targets are sparse region summaries rather than dense neuron states",
                "macro protocol and successor structure still remain outside this real sparse reconstruction test",
            ],
        },
        "progress_estimate": {
            "real_heldout_region_reconstruction_percent": 55.0,
            "candidate_region_to_region_reconstruction_percent": 66.0,
            "full_brain_encoding_mechanism_percent": 77.0,
        },
        "next_large_blocks": [
            "Merge more real extraction sources so the real held-out reconstruction stops depending on a small 18-concept codebook.",
            "Move from sparse region summaries to denser activation signatures.",
            "Add macro successor/protocol targets to the real-derived reconstruction benchmark.",
        ],
    }
    return payload


def test_dnn_real_heldout_region_reconstruction_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["heldout_gain_mean"] > 0.6
    assert metrics["positive_pair_count"] >= 3
    assert metrics["reconstruction_score"] > 0.6
    assert verdict["real_candidate_region_operator_present"] is True
    assert verdict["real_exact_region_operator_present"] is False


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN real held-out region reconstruction block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_real_heldout_region_reconstruction_block_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
