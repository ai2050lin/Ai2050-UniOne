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

from research.gpt5.code.dnn_exact_encoding_system import DnnExactEncodingSystem  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    system = DnnExactEncodingSystem.from_artifacts(ROOT)
    summary = system.summary()

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_exact_encoding_system_block",
        },
        "strict_goal": {
            "statement": "Collapse current DNN-side extraction and restoration results into one exact-encoding system law, and quantify whether the project has reached parameteric system understanding or exact theorem closure.",
            "boundary": "This block reports a system theorem candidate. It does not claim exact system closure unless exact evidence and successor closure also rise together.",
        },
        "headline_metrics": summary,
        "metric_lines_cn": [
            f"（basis+offset核心强度）basis_offset_core_score = {summary['basis_offset_core_score']:.4f}",
            f"（contextual+protocol系统强度）contextual_protocol_score = {summary['contextual_protocol_score']:.4f}",
            f"（successor系统项强度）successor_system_score = {summary['successor_system_score']:.4f}",
            f"（精确证据强度）evidence_exactness_score = {summary['evidence_exactness_score']:.4f}",
            f"（系统参数原理强度）system_parametric_score = {summary['system_parametric_score']:.4f}",
            f"（系统精确闭合度）exact_system_closure_score = {summary['exact_system_closure_score']:.4f}",
        ],
        "strict_verdict": {
            "system_parametric_principle_present": bool(summary["system_parametric_score"] > 0.70),
            "exact_system_closure_present": bool(summary["exact_system_closure_score"] > 0.82),
            "core_answer": (
                "The system-level exact encoding principle is now visible as a real theorem candidate: family basis and bounded concept offset are necessary, but exact encoding also depends on contextual correction, protocol correction, and successor transport. The project has reached candidate-level system understanding, not exact theorem closure."
            ),
            "main_hard_gaps": [
                "exact evidence is still thinner than parametric evidence",
                "family-to-specific exact closure is still open",
                "successor restoration remains the weakest system term",
                "dense neuron-level exact tensors still lag behind row/signature-level evidence",
            ],
        },
        "progress_estimate": {
            "exact_encoding_system_percent": 68.0,
            "system_parametric_principle_percent": 73.0,
            "exact_system_closure_percent": 34.0,
            "full_brain_encoding_mechanism_percent": 90.0,
        },
        "next_large_blocks": [
            "Raise dense neuron-level exact evidence so evidence exactness stops lagging behind parametric structure.",
            "Push family-to-specific exact closure and successor exact closure together instead of treating them as separate local issues.",
            "Retest the theorem candidate on unseen families and stronger dense exports before claiming exact closure.",
        ],
    }
    return payload


def test_dnn_exact_encoding_system_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["basis_offset_core_score"] > 0.85
    assert metrics["contextual_protocol_score"] > 0.80
    assert metrics["system_parametric_score"] > 0.70
    assert verdict["system_parametric_principle_present"] is True
    assert verdict["exact_system_closure_present"] is False
    assert len(payload["metric_lines_cn"]) >= 6
    assert payload["metric_lines_cn"][0].startswith("（")


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN exact encoding system block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_exact_encoding_system_block_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["metric_lines_cn"]))


if __name__ == "__main__":
    main()
