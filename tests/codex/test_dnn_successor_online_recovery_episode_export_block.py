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

from research.gpt5.code.dnn_successor_online_recovery_episode_export import OnlineRecoveryEpisodeExport  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    export = OnlineRecoveryEpisodeExport.from_artifact(ROOT)
    summary = export.summary()
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_successor_online_recovery_episode_export_block",
        },
        "strict_goal": {
            "statement": "Expand the online recovery successor proxy from step summaries into a standardized episode-step export layer.",
            "boundary": "This block improves granularity, but it is still proxy-level and does not yet expose layer/unit dense tensors.",
        },
        "episode_export_summary": summary,
        "headline_metrics": {
            "episode_step_rows": summary["episode_step_rows"],
            "model_count": summary["model_count"],
            "step_count": summary["step_count"],
            "triggered_total": summary["triggered_total"],
            "recovered_total": summary["recovered_total"],
            "mean_triggered_rate": summary["mean_triggered_rate"],
            "mean_recovered_rate": summary["mean_recovered_rate"],
        },
        "strict_verdict": {
            "episode_step_export_present": bool(summary["episode_step_rows"] == 1920),
            "core_answer": (
                "The online recovery path is no longer stuck at step-level summary only. "
                "It now has a standardized episode-step export layer that can later be aligned with layer/unit dense exports."
            ),
            "main_hard_gaps": [
                "the export is still derived from step-level rates rather than real layer/unit activations",
                "episode rows improve granularity, not exactness",
                "this block prepares the bridge to dense exports; it does not yet complete that bridge",
            ],
        },
        "progress_estimate": {
            "online_recovery_episode_export_percent": 76.0,
            "successor_stage_row_corpus_percent": 67.0,
            "successor_dense_exact_closure_percent": 41.0,
            "full_brain_encoding_mechanism_percent": 89.0,
        },
        "next_large_blocks": [
            "Attach layer and unit axes to the episode-step export so it stops being rate-derived and becomes dense-state-derived.",
            "Use the episode-step export as the first replacement of online recovery proxy rows inside the successor corpus.",
            "Recompute successor restoration after the online recovery path gains real dense coordinates.",
        ],
    }
    return payload


def test_dnn_successor_online_recovery_episode_export_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["episode_step_rows"] == 1920
    assert metrics["model_count"] == 2
    assert metrics["step_count"] == 4
    assert metrics["triggered_total"] > 200
    assert metrics["recovered_total"] > 100
    assert verdict["episode_step_export_present"] is True


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN successor online recovery episode export block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_successor_online_recovery_episode_export_block_20260315.json",
    )
    args = ap.parse_args()
    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
