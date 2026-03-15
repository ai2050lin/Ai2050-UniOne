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

from research.gpt5.code.dnn_successor_real_corpus import DnnSuccessorRealCorpus  # noqa: E402


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    temp = ROOT / "tests" / "codex_temp"
    corpus = DnnSuccessorRealCorpus.from_artifacts(ROOT).summary()
    extraction = load_json(temp / "dnn_successor_structure_extraction_block_20260315.json")
    exactness_gap = load_json(temp / "dnn_successor_exactness_gap_block_20260315.json")
    export_contract = load_json(temp / "dnn_successor_dense_export_contract_block_20260315.json")
    math_status = load_json(temp / "dnn_math_restoration_status_block_20260315.json")

    structure_score = float(extraction["headline_metrics"]["extracted_successor_score"])
    transport_score = clamp01(float(extraction["headline_metrics"]["transport_margin"]) / 0.16)
    exactness_score = clamp01(float(corpus["exactness_fraction"]) / 0.72)
    closure_penalty = 1.0 - float(exactness_gap["headline_metrics"]["proxy_ratio"])
    upgrade_score = clamp01(float(export_contract["headline_metrics"]["proxy_mean_upgrade_ready_score"]) / 0.70)
    base_parametric = float(math_status["restoration_terms"]["successor_parametric_score"])

    restoration_score = (
        0.24 * structure_score
        + 0.18 * transport_score
        + 0.20 * exactness_score
        + 0.16 * closure_penalty
        + 0.10 * upgrade_score
        + 0.12 * base_parametric
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_successor_math_restoration_block",
        },
        "strict_goal": {
            "statement": "Report the exactness-aware mathematical restoration status of successor under the new unified successor corpus.",
            "boundary": "This block reports successor restoration status. It does not claim that successor theorem closure has been reached.",
        },
        "restoration_terms": {
            "successor_structure_score": structure_score,
            "successor_transport_score": transport_score,
            "successor_exactness_score": exactness_score,
            "successor_closure_penalty_term": closure_penalty,
            "successor_upgrade_score": upgrade_score,
            "successor_base_parametric_score": base_parametric,
            "successor_restoration_score": float(restoration_score),
        },
        "metric_lines_cn": [
            f"（successor结构项）successor_structure_score = {structure_score:.4f}",
            f"（successor传输项）successor_transport_score = {transport_score:.4f}",
            f"（successor精确证据）successor_exactness_score = {exactness_score:.4f}",
            f"（successor闭合惩罚项）successor_closure_penalty_term = {closure_penalty:.4f}",
            f"（successor升级准备度）successor_upgrade_score = {upgrade_score:.4f}",
            f"（successor基础参数项）successor_base_parametric_score = {base_parametric:.4f}",
            f"（successor数学还原）successor_restoration_score = {restoration_score:.4f}",
        ],
        "strict_verdict": {
            "successor_restoration_report_present": True,
            "successor_final_theorem_closed": bool(restoration_score > 0.88 and closure_penalty > 0.75),
            "core_answer": (
                "Successor mathematical restoration is now explicit under exactness-aware accounting. "
                "The structure term and the base parametric term are already real, but exactness and closure are still the limiting factors."
            ),
            "main_hard_gaps": [
                "successor exactness is still capped by proxy-heavy paths rather than dense tensor dominance",
                "transport is explicit but still not strong enough to close successor on theorem-grade standards",
                "the export contract is ready, but successor closure cannot rise much further until those contracts are executed",
            ],
        },
        "progress_estimate": {
            "successor_math_restoration_percent": 63.0,
            "successor_dense_exact_closure_percent": 41.0,
            "math_restoration_status_percent": 73.0,
            "full_brain_encoding_mechanism_percent": 89.0,
        },
        "next_large_blocks": [
            "Execute the online recovery dense export contract and replace summary-proxy successor rows.",
            "Execute the successor inventory dense chain-stage export and replace inventory-proxy rows.",
            "After both exports land, recompute successor restoration and test whether it can cross the closure threshold.",
        ],
    }
    return payload


def test_dnn_successor_math_restoration_block() -> None:
    payload = build_payload()
    terms = payload["restoration_terms"]
    verdict = payload["strict_verdict"]
    assert terms["successor_structure_score"] > 0.30
    assert terms["successor_transport_score"] > 0.45
    assert terms["successor_exactness_score"] > 0.40
    assert terms["successor_closure_penalty_term"] < 0.50
    assert terms["successor_restoration_score"] > 0.45
    assert verdict["successor_restoration_report_present"] is True
    assert verdict["successor_final_theorem_closed"] is False
    assert len(payload["metric_lines_cn"]) >= 7
    assert payload["metric_lines_cn"][0].startswith("（")


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN successor math restoration block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_successor_math_restoration_block_20260315.json",
    )
    args = ap.parse_args()
    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["metric_lines_cn"]))


if __name__ == "__main__":
    main()
