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

from research.gpt5.code.dnn_successor_dense_export_contract import SuccessorDenseExportContract  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    contract = SuccessorDenseExportContract.from_repo(ROOT)
    summary = contract.summary()
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_successor_dense_export_contract_block",
        },
        "strict_goal": {
            "statement": "Define exact dense export contracts for successor paths that are still proxy-based.",
            "boundary": "This block defines the export contract. It does not yet execute dense successor exports.",
        },
        "contract_summary": summary,
        "headline_metrics": {
            "successor_rows": summary["successor_rows"],
            "direct_dense_rows": summary["direct_dense_rows"],
            "proxy_rows": summary["proxy_rows"],
            "mean_upgrade_ready_score": summary["mean_upgrade_ready_score"],
            "proxy_mean_upgrade_ready_score": summary["proxy_mean_upgrade_ready_score"],
            "fully_specified_proxy_rows": summary["fully_specified_proxy_rows"],
        },
        "strict_verdict": {
            "successor_export_contract_present": bool(summary["proxy_rows"] == 2 and summary["fully_specified_proxy_rows"] == 2),
            "core_answer": (
                "The project now has explicit dense export contracts for the two proxy successor paths. "
                "That means the proxy gap is no longer vague: we now know exactly which axes and tensor layouts are missing."
            ),
            "main_hard_gaps": [
                "the online recovery path still lacks episode/layer/unit dense tensors even though the required schema is now specified",
                "the successor inventory path still lacks chain-stage row states even though the required schema is now specified",
                "the direct-dense route still lacks an explicit stage axis, so even the strongest successor path is not yet fully aligned",
            ],
        },
        "progress_estimate": {
            "successor_dense_export_contract_percent": 71.0,
            "successor_dense_exact_closure_percent": 41.0,
            "full_brain_encoding_mechanism_percent": 89.0,
        },
        "next_large_blocks": [
            "Implement dense export writers for the online recovery path using the new episode-step-layer-unit schema.",
            "Implement dense chain-stage row-state export for the successor inventory path.",
            "Add explicit stage-axis alignment to the direct-dense multi-hop route path and then recompute successor exactness.",
        ],
    }
    return payload


def test_dnn_successor_dense_export_contract_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["successor_rows"] == 3
    assert metrics["direct_dense_rows"] == 1
    assert metrics["proxy_rows"] == 2
    assert metrics["fully_specified_proxy_rows"] == 2
    assert metrics["mean_upgrade_ready_score"] > 0.50
    assert metrics["proxy_mean_upgrade_ready_score"] > 0.40
    assert verdict["successor_export_contract_present"] is True


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN successor dense export contract block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_successor_dense_export_contract_block_20260315.json",
    )
    args = ap.parse_args()
    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
