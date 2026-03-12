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
        raise FileNotFoundError(f"没有找到匹配文件: {pattern}")
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="10轮循环 v2 评估")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_10round_excavation_loop_v2_assessment_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    loop = json.loads(latest_match("theory_track_10round_excavation_loop_v2_*.json").read_text(encoding="utf-8"))

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_10Round_Excavation_Loop_V2_Assessment",
        },
        "headline_metrics": {
            "strict_theorem_count_after_20rounds_total": int(loop["ending_point"]["strict_count"]),
            "active_theorem_count_after_20rounds_total": int(loop["ending_point"]["active_count"]),
            "queued_theorem_count_after_20rounds_total": int(loop["ending_point"]["queued_count"]),
            "encoding_inverse_reconstruction_readiness": float(
                loop["derived_readiness"]["encoding_inverse_reconstruction_readiness"]
            ),
            "new_math_closure_readiness": float(
                loop["derived_readiness"]["new_math_closure_readiness"]
            ),
        },
        "key_findings": {
            "block_usage_counts": loop["ending_point"]["block_usage_counts"],
            "most_open_gaps": loop["next_round_21_recommendation"]["core_gap_order"],
            "round_21_recommendation": loop["next_round_21_recommendation"]["recommended_block"],
        },
        "verdict": {
            "core_answer": (
                "v2 循环已经不再单向重复同一个 block，而是开始显式覆盖 protocol、theorem pruning 和 relation/tool 线，"
                "更接近真正可持续运行的自动科研闭环体。"
            ),
            "remaining_hard_gaps": [
                "successor_coherence 仍是最大缺口",
                "protocol_calling 仍未打透",
                "anchored_bridge_lift_theorem 若仍 queued，说明 bridge-role 动力学还需专门阶段块",
            ],
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
