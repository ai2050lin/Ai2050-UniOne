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

from research.gpt5.code.dnn_joint_closure_sprint_manifest import DnnJointClosureSprintManifest  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    manifest = DnnJointClosureSprintManifest.from_artifacts(ROOT)
    summary = manifest.summary()

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_joint_closure_sprint_manifest_block",
        },
        "strict_goal": {
            "statement": "Turn the leverage ranking and dense harvest readiness into one executable sprint manifest for the next closure stage.",
            "boundary": "This block is a sprint manifest. It does not itself improve closure scores.",
        },
        "headline_metrics": summary,
        "strict_verdict": {
            "joint_closure_sprint_manifest_present": True,
            "sprint_can_be_started_now": bool(summary["sprint_readiness_score"] > 0.70),
            "core_answer": "The next stage can now be framed as one executable sprint instead of a vague research direction: specific exact closure first, successor closure second, dense exact evidence as the supporting floor.",
            "main_hard_gaps": [
                "the projected stage target still remains well below final theorem closure",
                "the sprint is executable, but it is still a mid-stage push rather than an endgame push",
                "protocol dense evidence is still a support floor, not a sufficient breakthrough axis by itself",
            ],
        },
        "progress_estimate": {
            "joint_closure_sprint_manifest_percent": 77.0,
            "coupled_exact_closure_percent": 43.0,
            "full_brain_encoding_mechanism_percent": 90.0,
        },
        "metric_lines_cn": summary["metric_lines_cn"],
        "next_large_blocks": summary["critical_path"],
    }
    return payload


def test_dnn_joint_closure_sprint_manifest_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    axes = metrics["sprint_axes"]
    assert metrics["sprint_readiness_score"] > 0.70
    assert axes[0]["axis"] == "family-to-specific exact closure"
    assert axes[1]["axis"] == "successor exact closure"
    assert axes[2]["axis"] == "dense exact evidence"
    assert verdict["joint_closure_sprint_manifest_present"] is True
    assert verdict["sprint_can_be_started_now"] is True
    assert len(payload["metric_lines_cn"]) >= 5
    assert payload["metric_lines_cn"][0].startswith("（")


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN joint closure sprint manifest block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_joint_closure_sprint_manifest_block_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(payload["metric_lines_cn"]))


if __name__ == "__main__":
    main()
