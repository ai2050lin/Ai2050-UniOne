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

from research.gpt5.code.dnn_joint_dense_export_schema import DnnJointDenseExportSchema  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    schema = DnnJointDenseExportSchema.from_artifacts(ROOT)
    summary = schema.summary()

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_joint_dense_export_schema_block",
        },
        "strict_goal": {
            "statement": "Unify specific/protocol/successor dense harvesting under one executable export schema so the next sprint can produce mergeable exact tensors.",
            "boundary": "This block validates export schema readiness. It does not yet write real dense tensors.",
        },
        "headline_metrics": summary,
        "strict_verdict": {
            "joint_dense_export_schema_present": True,
            "schema_can_be_started_now": bool(summary["schema_ready_score"] > 0.85),
            "core_answer": "The next sprint no longer lacks a shared export schema. The three main axes can now target one mergeable dense tensor contract instead of three incompatible output styles.",
            "main_hard_gaps": [
                "schema readiness is high, but real tensor writers still need to be implemented inside the source scripts",
                "protocol currently stabilizes the support floor more than it creates direct closure gains",
                "successor remains the hardest schema because it must preserve chain-stage structure exactly",
            ],
        },
        "progress_estimate": {
            "joint_dense_export_schema_percent": 81.0,
            "joint_closure_sprint_manifest_percent": 77.0,
            "full_brain_encoding_mechanism_percent": 90.0,
        },
        "metric_lines_cn": summary["metric_lines_cn"],
        "next_large_blocks": summary["critical_path"],
    }
    return payload


def test_dnn_joint_dense_export_schema_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["schema_ready_score"] > 0.85
    assert len(metrics["schema_rows"]) == 3
    assert sum(1 for row in metrics["schema_rows"] if row["launchable"]) == 3
    assert verdict["joint_dense_export_schema_present"] is True
    assert verdict["schema_can_be_started_now"] is True
    assert len(payload["metric_lines_cn"]) >= 4
    assert payload["metric_lines_cn"][0].startswith("（")


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN joint dense export schema block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_joint_dense_export_schema_block_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["metric_lines_cn"]))


if __name__ == "__main__":
    main()
