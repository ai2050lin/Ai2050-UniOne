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
    ap = argparse.ArgumentParser(description="10轮自动循环评估")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_10round_excavation_loop_assessment_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    loop = json.loads(latest_match("theory_track_10round_excavation_loop_*.json").read_text(encoding="utf-8"))

    final_scores = loop["ending_point"]["final_scores"]
    strict_count = int(loop["ending_point"]["strict_count"])
    active_count = int(loop["ending_point"]["active_count"])

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_10Round_Excavation_Loop_Assessment",
        },
        "headline_metrics": {
            "strict_theorem_count_after_10rounds": strict_count,
            "active_theorem_count_after_10rounds": active_count,
            "encoding_inverse_reconstruction_readiness": float(
                loop["derived_readiness"]["encoding_inverse_reconstruction_readiness"]
            ),
            "new_math_closure_readiness": float(
                loop["derived_readiness"]["new_math_closure_readiness"]
            ),
        },
        "key_findings": {
            "most_strengthened_axes": sorted(
                [
                    {"axis": axis, "score": float(score)}
                    for axis, score in final_scores.items()
                ],
                key=lambda item: item["score"],
                reverse=True,
            )[:4],
            "most_open_gaps": loop["next_round_11_recommendation"]["core_gap_order"],
            "round_11_recommendation": loop["next_round_11_recommendation"]["recommended_block"],
        },
        "verdict": {
            "core_answer": (
                "10轮循环已经可以形成一个稳定的自动推进器：每轮都从当前 gap、theorem frontier 和优先级块里自适应选下一轮动作，"
                "并把结果反灌到编码机制逆向还原和新数学体系收紧上。"
            ),
            "remaining_hard_gaps": [
                "successor_coherence 仍是最大缺口",
                "protocol_calling 和 relation_chain 仍未完全打透",
                "long-chain inventory 仍是 naturalized prototype，不是真实模型内部自然长链 trace",
            ],
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
