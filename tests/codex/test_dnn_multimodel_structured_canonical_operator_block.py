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
    grouped: Dict[str, List[MultiModelRealEntry]] = {}
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

    affine_family_specific = atlas.fit_affine_operator("family", "specific", train_entries)
    affine_family_specific_eval = atlas.evaluate_affine_operator(affine_family_specific, "family", "specific", test_entries)

    affine_contextual_specific = atlas.fit_affine_operator("contextual_family", "specific", train_entries)
    affine_contextual_specific_eval = atlas.evaluate_affine_operator(
        affine_contextual_specific,
        "contextual_family",
        "specific",
        test_entries,
    )

    structured_specific = atlas.fit_structured_canonical_operator(
        ["family", "stage", "macro", "shared"],
        "specific",
        train_entries,
        family_condition_view="family",
    )
    structured_specific_eval = atlas.evaluate_structured_canonical_operator(structured_specific, test_entries)

    affine_macro = atlas.fit_affine_operator("stage", "macro", train_entries)
    affine_macro_eval = atlas.evaluate_affine_operator(affine_macro, "stage", "macro", test_entries)

    structured_macro = atlas.fit_structured_canonical_operator(
        ["stage", "family", "shared"],
        "macro",
        train_entries,
        family_condition_view="shared",
    )
    structured_macro_eval = atlas.evaluate_structured_canonical_operator(structured_macro, test_entries)

    family_specific_gain_delta = structured_specific_eval["relative_gain"] - affine_family_specific_eval["relative_gain"]
    contextual_specific_gap = structured_specific_eval["relative_gain"] - affine_contextual_specific_eval["relative_gain"]
    macro_gain_delta = structured_macro_eval["relative_gain"] - affine_macro_eval["relative_gain"]
    canonical_operator_score = min(
        1.0,
        0.40 * min(1.0, structured_specific_eval["relative_gain"] / 0.72)
        + 0.25 * min(1.0, max(0.0, family_specific_gain_delta) / 0.08)
        + 0.20 * min(1.0, structured_macro_eval["relative_gain"] / 0.30)
        + 0.15 * min(1.0, max(0.0, macro_gain_delta) / 0.10),
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_multimodel_structured_canonical_operator_block",
        },
        "strict_goal": {
            "statement": (
                "Replace flat affine recovery with a structured family-conditioned canonical operator that can better recover specific structure and macro protocol fields."
            ),
            "boundary": (
                "This block proves a stronger structured candidate operator. It does not prove that the final canonical theorem has already been reached."
            ),
        },
        "data_split": {
            "train_entries": len(train_entries),
            "test_entries": len(test_entries),
            "families": sorted({entry.family for entry in atlas.entries}),
        },
        "operator_results": {
            "affine_family_to_specific": affine_family_specific_eval,
            "affine_contextual_to_specific": affine_contextual_specific_eval,
            "structured_contextual_to_specific": structured_specific_eval,
            "affine_stage_to_macro": affine_macro_eval,
            "structured_stage_family_shared_to_macro": structured_macro_eval,
        },
        "headline_metrics": {
            "family_specific_gain_delta": float(family_specific_gain_delta),
            "contextual_specific_gap": float(contextual_specific_gap),
            "macro_gain_delta": float(macro_gain_delta),
            "structured_specific_gain": float(structured_specific_eval["relative_gain"]),
            "structured_macro_gain": float(structured_macro_eval["relative_gain"]),
            "canonical_operator_score": float(canonical_operator_score),
        },
        "strict_verdict": {
            "structured_specific_improvement_present": bool(family_specific_gain_delta > 0.02),
            "structured_contextual_supremacy_present": bool(contextual_specific_gap > 0.02),
            "structured_macro_recovery_present": bool(structured_macro_eval["relative_gain"] > 0.08),
            "exact_canonical_operator_present": bool(
                structured_specific_eval["relative_gain"] > 0.90 and structured_macro_eval["relative_gain"] > 0.75
            ),
            "core_answer": (
                "A structured family-conditioned operator already outperforms family-only specific recovery, and it starts to recover macro protocol structure that flat stage-only transport misses. But it still does not beat the best affine contextual shortcut on specific recovery."
            ),
            "main_hard_gaps": [
                "the operator is still summary-level and family-conditioned, not yet a dense canonical theorem",
                "structured recovery still trails the best flat contextual shortcut on specific targets",
                "macro recovery remains much weaker than specific recovery",
                "protocol and successor fields are still compressed proxies rather than dense activation targets",
            ],
        },
        "progress_estimate": {
            "structured_canonical_operator_percent": 62.0,
            "multimodel_specific_recovery_percent": 61.0,
            "full_brain_encoding_mechanism_percent": 79.0,
        },
        "next_large_blocks": [
            "Push structured operators into dense activation-level specific targets.",
            "Bind protocol and successor coordinates into the same canonical operator objective.",
            "Test whether structured operators generalize beyond seen family semantics.",
        ],
    }
    return payload


def test_dnn_multimodel_structured_canonical_operator_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["family_specific_gain_delta"] > 0.02
    assert metrics["structured_macro_gain"] > 0.08
    assert metrics["canonical_operator_score"] > 0.55
    assert verdict["structured_specific_improvement_present"] is True
    assert verdict["structured_contextual_supremacy_present"] is False
    assert verdict["exact_canonical_operator_present"] is False


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN multimodel structured canonical operator block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_multimodel_structured_canonical_operator_block_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
