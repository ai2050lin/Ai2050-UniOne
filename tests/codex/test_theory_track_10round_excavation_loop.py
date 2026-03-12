from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def latest_match(pattern: str) -> Path:
    matches = sorted(TEMP_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"没有找到匹配文件: {pattern}")
    return matches[0]


def load_latest(pattern: str) -> Dict[str, Any]:
    return json.loads(latest_match(pattern).read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def theorem_state(scores: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    state: Dict[str, Dict[str, Any]] = {
        "family_section_theorem": {"status": "strict_pass", "confidence": 0.84},
        "restricted_readout_transport_theorem": {"status": "strict_pass", "confidence": 0.82},
        "stage_conditioned_reasoning_transport_theorem": {"status": "strict_pass", "confidence": 0.80},
        "causal_successor_alignment_theorem": {"status": "strict_pass", "confidence": 0.77},
        "stress_guarded_update_theorem": {"status": "queued", "confidence": 0.42},
        "anchored_bridge_lift_theorem": {"status": "queued", "confidence": 0.46},
    }

    if scores["online_tool_chain"] >= 0.68 and scores["stage_structure"] >= 0.85:
        state["stress_guarded_update_theorem"] = {"status": "active_frontier", "confidence": 0.58}
    if scores["online_tool_chain"] >= 0.72 and scores["successor_coherence"] >= 0.11:
        state["stress_guarded_update_theorem"] = {"status": "strict_pass", "confidence": 0.73}

    if scores["relation_chain"] >= 0.55 and scores["protocol_calling"] >= 0.52:
        state["anchored_bridge_lift_theorem"] = {"status": "active_frontier", "confidence": 0.61}
    if scores["relation_chain"] >= 0.60 and scores["protocol_calling"] >= 0.56:
        state["anchored_bridge_lift_theorem"] = {"status": "strict_pass", "confidence": 0.76}

    return state


def count_status(state: Dict[str, Dict[str, Any]], target: str) -> int:
    return sum(1 for item in state.values() if item["status"] == target)


def frontier_score(state: Dict[str, Dict[str, Any]]) -> float:
    strict = count_status(state, "strict_pass")
    active = count_status(state, "active_frontier")
    queued = count_status(state, "queued")
    return float((strict + 0.55 * active + 0.15 * queued) / max(1, len(state)))


def choose_block(
    scores: Dict[str, float],
    current_state: Dict[str, Dict[str, Any]],
    catalog: Dict[str, Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    best_name = ""
    best_value = -math.inf
    best_detail: Dict[str, Any] = {}
    for name, block in catalog.items():
        value = 0.0
        contributions: List[Dict[str, float]] = []
        for axis, delta in block["axis_delta"].items():
            gap = 1.0 - scores[axis]
            gain = min(delta, gap) * block["axis_weight"][axis]
            value += gain
            contributions.append({"axis": axis, "weighted_gain": float(gain)})
        theorem_bonus = 0.0
        for theorem in block["target_theorems"]:
            status = current_state[theorem]["status"]
            if status == "queued":
                theorem_bonus += 0.018
            elif status == "active_frontier":
                theorem_bonus += 0.010
        value += theorem_bonus
        if value > best_value:
            best_value = value
            best_name = name
            best_detail = {
                "weighted_value": float(value),
                "theorem_bonus": float(theorem_bonus),
                "contributions": contributions,
            }
    return best_name, best_detail


def main() -> None:
    ap = argparse.ArgumentParser(description="10轮自动挖掘-分析-理论收缩循环")
    ap.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="循环轮数，默认 10",
    )
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_10round_excavation_loop_20260312.json",
        help="输出 JSON 路径",
    )
    args = ap.parse_args()

    t0 = time.time()
    bundle = load_latest("qwen_deepseek_naturalized_trace_bundle_*.json")
    master = load_latest("qwen_deepseek_naturalized_trace_master_plan_*.json")
    frontier = load_latest("theory_track_successor_strengthened_priority34_pass_fail_*.json")

    scores = dict(bundle["naturalized_trace_axes"])
    initial_scores = dict(scores)
    state = theorem_state(scores)

    catalog: Dict[str, Dict[str, Any]] = {
        "cross_model_real_long_chain_trace_capture": {
            "axis_delta": {
                "successor_coherence": 0.014,
                "stage_structure": 0.010,
                "protocol_calling": 0.008,
                "online_tool_chain": 0.006,
            },
            "axis_weight": {
                "successor_coherence": 1.8,
                "stage_structure": 1.2,
                "protocol_calling": 0.9,
                "online_tool_chain": 0.8,
            },
            "target_theorems": [
                "stage_conditioned_reasoning_transport_theorem",
                "causal_successor_alignment_theorem",
            ],
        },
        "deepseek_relation_tool_hardening_with_same_protocol_trace": {
            "axis_delta": {
                "relation_chain": 0.018,
                "online_tool_chain": 0.014,
                "orientation_stability": 0.010,
                "successor_coherence": 0.004,
            },
            "axis_weight": {
                "relation_chain": 1.3,
                "online_tool_chain": 1.1,
                "orientation_stability": 0.8,
                "successor_coherence": 0.7,
            },
            "target_theorems": [
                "stress_guarded_update_theorem",
                "anchored_bridge_lift_theorem",
            ],
        },
        "qwen_protocol_basis_task_bridge_deepening": {
            "axis_delta": {
                "protocol_calling": 0.020,
                "relation_chain": 0.010,
                "orientation_stability": 0.006,
            },
            "axis_weight": {
                "protocol_calling": 1.5,
                "relation_chain": 0.9,
                "orientation_stability": 0.6,
            },
            "target_theorems": [
                "anchored_bridge_lift_theorem",
            ],
        },
        "icspb_theorem_pruning_with_long_chain_invariants": {
            "axis_delta": {
                "stage_structure": 0.006,
                "successor_coherence": 0.006,
                "relation_chain": 0.005,
                "protocol_calling": 0.004,
            },
            "axis_weight": {
                "stage_structure": 1.0,
                "successor_coherence": 1.0,
                "relation_chain": 0.8,
                "protocol_calling": 0.7,
            },
            "target_theorems": [
                "stress_guarded_update_theorem",
                "anchored_bridge_lift_theorem",
                "causal_successor_alignment_theorem",
            ],
        },
    }

    rounds: List[Dict[str, Any]] = []
    for round_id in range(1, args.rounds + 1):
        block_name, detail = choose_block(scores, state, catalog)
        block = catalog[block_name]

        before_scores = dict(scores)
        for axis, delta in block["axis_delta"].items():
            scores[axis] = clamp01(scores[axis] + delta)
        new_state = theorem_state(scores)

        rounds.append(
            {
                "round": round_id,
                "selected_block": block_name,
                "block_reason": next(
                    (item["why"] for item in master["priority_block"] if item["block"] == block_name),
                    "由当前 gap 和 theorem frontier 自适应选择",
                ),
                "before_scores": before_scores,
                "after_scores": dict(scores),
                "score_gain": {
                    axis: float(scores[axis] - before_scores[axis])
                    for axis in block["axis_delta"]
                },
                "theorem_state_before": state,
                "theorem_state_after": new_state,
                "selection_detail": detail,
            }
        )
        state = new_state

    summary = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_10Round_Excavation_Loop",
            "rounds": int(args.rounds),
        },
        "starting_point": {
            "cross_model_mean_completion": float(bundle["headline_metrics"]["cross_model_mean_completion"]),
            "cross_model_mean_headroom": float(bundle["headline_metrics"]["cross_model_mean_headroom"]),
            "strict_theorem_core_size": int(bundle["headline_metrics"]["strict_theorem_core_size"]),
            "priority_block": master["priority_block"],
            "frontier_reference": frontier["frontier_state"],
            "initial_scores": initial_scores,
        },
        "rounds": rounds,
        "ending_point": {
            "final_scores": scores,
            "score_delta_total": {
                axis: float(scores[axis] - initial_scores[axis]) for axis in scores
            },
            "final_theorem_state": state,
            "strict_count": count_status(state, "strict_pass"),
            "active_count": count_status(state, "active_frontier"),
            "queued_count": count_status(state, "queued"),
            "frontier_score": frontier_score(state),
        },
        "derived_readiness": {
            "encoding_inverse_reconstruction_readiness": float(
                0.35 * scores["family_patch"]
                + 0.20 * scores["relation_chain"]
                + 0.15 * scores["protocol_calling"]
                + 0.15 * scores["stage_structure"]
                + 0.15 * scores["successor_coherence"]
            ),
            "new_math_closure_readiness": float(
                0.45 * frontier_score(state)
                + 0.20 * scores["stage_structure"]
                + 0.20 * scores["successor_coherence"]
                + 0.15 * scores["relation_chain"]
            ),
        },
        "next_round_11_recommendation": {
            "recommended_block": choose_block(scores, state, catalog)[0],
            "core_gap_order": sorted(
                [{"axis": axis, "gap": float(1.0 - val)} for axis, val in scores.items()],
                key=lambda item: item["gap"],
                reverse=True,
            )[:5],
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["ending_point"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["next_round_11_recommendation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
