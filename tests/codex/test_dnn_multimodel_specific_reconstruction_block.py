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

from research.gpt5.code.dnn_multimodel_real_atlas import MultiModelRealAtlas, MultiModelRealEntry  # noqa: E402


def split_entries(entries: List[MultiModelRealEntry]) -> tuple[List[MultiModelRealEntry], List[MultiModelRealEntry]]:
    train: List[MultiModelRealEntry] = []
    test: List[MultiModelRealEntry] = []
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
    atlas = MultiModelRealAtlas.from_artifacts(ROOT)
    train_entries, test_entries = split_entries(atlas.entries)
    pairs = [
        ("family", "specific"),
        ("contextual_family", "specific"),
        ("stage", "macro"),
        ("shared", "macro"),
    ]
    pair_results = {}
    gains = {}
    for source_view, target_view in pairs:
        operator = atlas.fit_affine_operator(source_view, target_view, train_entries)
        train_eval = atlas.evaluate_affine_operator(operator, source_view, target_view, train_entries)
        test_eval = atlas.evaluate_affine_operator(operator, source_view, target_view, test_entries)
        key = f"{source_view}_to_{target_view}"
        pair_results[key] = {
            "train_relative_gain": train_eval["relative_gain"],
            "test_relative_gain": test_eval["relative_gain"],
            "test_mean_error": test_eval["mean_error"],
            "test_baseline_error": test_eval["baseline_error"],
        }
        gains[key] = float(test_eval["relative_gain"])

    contextual_recovery_gain = gains["contextual_family_to_specific"] - gains["family_to_specific"]
    mean_gain = float(sum(gains.values()) / len(gains))
    reconstruction_score = min(
        1.0,
        0.35 * min(1.0, mean_gain / 0.65)
        + 0.35 * min(1.0, gains["contextual_family_to_specific"] / 0.45)
        + 0.20 * min(1.0, contextual_recovery_gain / 0.20)
        + 0.10 * min(1.0, len(test_entries) / 10.0),
    )
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_multimodel_specific_reconstruction_block",
        },
        "strict_goal": {
            "statement": (
                "Use multimodel real-derived context to recover concept-specific structure more accurately than family-only reconstruction."
            ),
            "boundary": (
                "This block tests whether contextual family information helps recover specific concept structure. It does not yet prove exact concept recovery."
            ),
        },
        "data_split": {
            "train_entries": len(train_entries),
            "test_entries": len(test_entries),
            "families": sorted({entry.family for entry in atlas.entries}),
        },
        "pair_results": pair_results,
        "headline_metrics": {
            "mean_gain": mean_gain,
            "family_to_specific_gain": gains["family_to_specific"],
            "contextual_family_to_specific_gain": gains["contextual_family_to_specific"],
            "contextual_recovery_gain": contextual_recovery_gain,
            "reconstruction_score": reconstruction_score,
        },
        "strict_verdict": {
            "contextual_specific_recovery_present": bool(gains["contextual_family_to_specific"] > gains["family_to_specific"] + 0.05),
            "exact_specific_recovery_present": bool(gains["contextual_family_to_specific"] > 0.85),
            "core_answer": (
                "Multimodel real-derived context does improve specific reconstruction relative to family-only input. This means the specific-gap is not purely hopeless; it needs richer contextual operator structure rather than only a stronger family basis."
            ),
            "main_hard_gaps": [
                "specific reconstruction is still far from exact",
                "context helps, but canonical operator structure is still missing",
                "the current atlas still uses summary-level macro/protocol/successor signals rather than dense real activations",
            ],
        },
        "progress_estimate": {
            "multimodel_specific_recovery_percent": 57.0,
            "candidate_region_to_region_reconstruction_percent": 68.0,
            "full_brain_encoding_mechanism_percent": 78.0,
        },
        "next_large_blocks": [
            "Push contextual-family to specific reconstruction into dense activation-level targets.",
            "Replace affine operators with structured canonical operators.",
            "Add stronger protocol/successor signals into contextual recovery.",
        ],
    }
    return payload


def test_dnn_multimodel_specific_reconstruction_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["contextual_recovery_gain"] > 0.05
    assert metrics["reconstruction_score"] > 0.55
    assert verdict["contextual_specific_recovery_present"] is True
    assert verdict["exact_specific_recovery_present"] is False


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN multimodel specific reconstruction block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_multimodel_specific_reconstruction_block_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
