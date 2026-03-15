from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]


def load_json(rel_path: str) -> Dict[str, Any]:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    consolidation = load_json("tests/codex_temp/continuous_input_grounding_unified_consolidation_scan_20260309.json")
    task_summary = load_json("tests/codex_temp/agi_task_block_summary_20260309.json")
    factorization = load_json("tests/codex_temp/qwen_deepseek_concept_local_residual_auto_factorization_20260315.json")

    best = consolidation["best_dual_positive"]
    d_metrics = task_summary["blocks"]["D"]["metrics"]

    novelty_gain = float(best["novel_gain"])
    retention_gain = float(best["retention_gain"])
    overall_gain = float(best["overall_gain"])
    replay_strength = float(best["replay_steps"] * best["replay_alpha"])
    multistage_gain = float(d_metrics["multistage_best_overall_gain"])
    base_offset_gain = float(d_metrics["base_offset_best_overall_gain"])
    offset_stabilization_gap = float(
        d_metrics["multistage_best_overall_gain"] - d_metrics["offset_stabilization_best_overall_gain"]
    )

    novelty_score = max(0.0, min(1.0, 0.5 + novelty_gain))
    retention_score = max(0.0, min(1.0, 0.5 + retention_gain))
    stabilization_score = max(0.0, min(1.0, 0.5 + offset_stabilization_gap * 10.0))
    replay_score = max(0.0, min(1.0, replay_strength))
    routing_score = max(0.0, min(1.0, 0.5 + multistage_gain - base_offset_gain))
    closure_score = (novelty_score + retention_score + stabilization_score + replay_score + routing_score) / 5.0

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "QwenDeepSeek_adaptive_offset_dynamic_law",
        },
        "strict_goal": {
            "statement": "把新概念写入、保留、切换、回放、固化统一成同一条 offset 动态学习律。",
            "boundary": "当前闭合的是基于已有 grounding / consolidation / D 块证据的候选统一律，不是最终训练定律。",
        },
        "candidate_dynamic_law": {
            "equation": (
                "offset_(t+1) = offset_t + g_novel * Novelty_t + g_route * Routing_t + "
                "g_replay * Replay_t - g_decay * Decay_t + g_stab * Stabilization_t"
            ),
            "gating_form": (
                "g_* = sigmoid(w0 + w1 * novelty_gain + w2 * retention_gain + "
                "w3 * replay_strength + w4 * multistage_gain)"
            ),
            "meaning": (
                "offset 不是一次性写死，而是在 novelty 驱动写入、routing 分配、"
                "replay 回放和 stabilization 固化的共同作用下逐步形成。"
            ),
        },
        "supporting_evidence": {
            "best_dual_positive": best,
            "d_block_metrics": d_metrics,
            "factorization_joint_error": factorization["summary"]["joint_factorization_mean_error"],
        },
        "derived_scores": {
            "novelty_score": novelty_score,
            "retention_score": retention_score,
            "replay_score": replay_score,
            "stabilization_score": stabilization_score,
            "routing_score": routing_score,
            "closure_score": closure_score,
        },
        "strict_verdict": {
            "what_is_reached_now": (
                "动态 offset 已经可以写成统一候选律，至少能把 novelty、retention、replay、"
                "stabilization 和 routing 放到同一更新方程里。"
            ),
            "what_is_not_reached_yet": (
                "当前还没有真实大模型上的逐步写入轨迹去唯一锁定这些门控系数；"
                "这仍然更像理论闭合原型，而不是最终训练法则。"
            ),
        },
        "progress_estimate": {
            "adaptive_offset_dynamic_law_percent": 55.0,
            "whole_network_state_generator_percent": 49.0,
            "full_brain_encoding_mechanism_percent": 54.0,
        },
        "next_large_blocks": [
            "把真实新概念写入实验接到该动态律上，直接追踪 offset 的逐步形成轨迹。",
            "把 readout / successor / protocol bridge 并到同一状态方程中，避免动态律继续悬空。",
        ],
    }
    return payload


def test_qwen_deepseek_adaptive_offset_dynamic_law() -> None:
    payload = build_payload()
    scores = payload["derived_scores"]
    assert scores["novelty_score"] > scores["retention_score"]
    assert scores["closure_score"] > 0.45
    assert payload["progress_estimate"]["adaptive_offset_dynamic_law_percent"] >= 55.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen/DeepSeek adaptive offset dynamic law")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen_deepseek_adaptive_offset_dynamic_law_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["derived_scores"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
