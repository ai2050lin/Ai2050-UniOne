from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def latest_match(pattern: str) -> Path:
    matches = sorted(TEMP_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="10轮自动科研闭环 v3 评估")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_10round_excavation_loop_v3_assessment_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    loop = json.loads(latest_match("theory_track_10round_excavation_loop_v3_*.json").read_text(encoding="utf-8"))

    gaps = loop["round_31_recommendation"]["core_gap_order"]
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_10Round_Excavation_Loop_V3_Assessment",
        },
        "headline_metrics": {
            "strict_theorem_count_after_30rounds_total": int(loop["ending_point"]["strict_count"]),
            "active_theorem_count_after_30rounds_total": int(loop["ending_point"]["active_count"]),
            "queued_theorem_count_after_30rounds_total": int(loop["ending_point"]["queued_count"]),
            "pruned_hypothesis_count": int(loop["ending_point"]["pruned_hypothesis_count"]),
            "rollback_event_count": int(loop["ending_point"]["rollback_event_count"]),
            "encoding_inverse_reconstruction_readiness": float(
                loop["derived_readiness"]["encoding_inverse_reconstruction_readiness"]
            ),
            "new_math_closure_readiness": float(loop["derived_readiness"]["new_math_closure_readiness"]),
        },
        "key_findings": {
            "block_usage_counts": loop["ending_point"]["block_usage_counts"],
            "most_open_gaps": gaps[:5],
            "round_31_recommendation": loop["round_31_recommendation"]["recommended_block"],
        },
        "verdict": {
            "core_answer": (
                "v3 已经具备自动科研闭环体的执行核心：不仅会调度 block，还会强制理论收缩、更新脑侧因果闭合、"
                "自动淘汰弱假设并在必要时触发恢复。"
            ),
            "remaining_hard_gaps": [
                "successor_coherence 依然是最大缺口",
                "protocol_calling 仍需继续深化",
                "仍缺真实模型内部在线自然 trace 抓取层",
            ],
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
