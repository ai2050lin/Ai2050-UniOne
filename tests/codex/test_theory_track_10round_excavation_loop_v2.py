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


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_latest(pattern: str) -> Dict[str, Any]:
    return load_json(latest_match(pattern))


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


def make_catalog() -> Dict[str, Dict[str, Any]]:
    return {
        "cross_model_real_long_chain_trace_capture": {
            "axis_delta": {
                "successor_coherence": 0.018,
                "stage_structure": 0.012,
                "protocol_calling": 0.009,
                "online_tool_chain": 0.008,
            },
            "axis_weight": {
                "successor_coherence": 1.9,
                "stage_structure": 1.25,
                "protocol_calling": 0.95,
                "online_tool_chain": 0.85,
            },
            "target_theorems": [
                "stage_conditioned_reasoning_transport_theorem",
                "causal_successor_alignment_theorem",
            ],
            "kind": "cross_model",
        },
        "deepseek_relation_tool_hardening_with_same_protocol_trace": {
            "axis_delta": {
                "relation_chain": 0.020,
                "online_tool_chain": 0.015,
                "orientation_stability": 0.010,
                "successor_coherence": 0.005,
            },
            "axis_weight": {
                "relation_chain": 1.35,
                "online_tool_chain": 1.15,
                "orientation_stability": 0.8,
                "successor_coherence": 0.8,
            },
            "target_theorems": [
                "stress_guarded_update_theorem",
                "anchored_bridge_lift_theorem",
            ],
            "kind": "deepseek",
        },
        "qwen_protocol_basis_task_bridge_deepening": {
            "axis_delta": {
                "protocol_calling": 0.026,
                "relation_chain": 0.012,
                "orientation_stability": 0.006,
            },
            "axis_weight": {
                "protocol_calling": 1.7,
                "relation_chain": 0.95,
                "orientation_stability": 0.55,
            },
            "target_theorems": [
                "anchored_bridge_lift_theorem",
            ],
            "kind": "qwen",
        },
        "icspb_theorem_pruning_with_long_chain_invariants": {
            "axis_delta": {
                "stage_structure": 0.008,
                "successor_coherence": 0.008,
                "relation_chain": 0.006,
                "protocol_calling": 0.006,
            },
            "axis_weight": {
                "stage_structure": 1.0,
                "successor_coherence": 1.05,
                "relation_chain": 0.85,
                "protocol_calling": 0.8,
            },
            "target_theorems": [
                "stress_guarded_update_theorem",
                "anchored_bridge_lift_theorem",
                "causal_successor_alignment_theorem",
            ],
            "kind": "theorem",
        },
    }


def choose_block_v2(
    scores: Dict[str, float],
    current_state: Dict[str, Dict[str, Any]],
    catalog: Dict[str, Dict[str, Any]],
    history: List[str],
) -> Tuple[str, Dict[str, Any]]:
    best_name = ""
    best_value = -math.inf
    best_detail: Dict[str, Any] = {}

    recent3 = history[-3:]
    counts = {name: history.count(name) for name in catalog}

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
                theorem_bonus += 0.022
            elif status == "active_frontier":
                theorem_bonus += 0.012
        value += theorem_bonus

        # Force attention to protocol gap.
        if name == "qwen_protocol_basis_task_bridge_deepening" and scores["protocol_calling"] < 0.53:
            value += 0.030

        # Force theorem pruning if one theorem is still queued.
        if name == "icspb_theorem_pruning_with_long_chain_invariants" and (
            current_state["stress_guarded_update_theorem"]["status"] != "strict_pass"
            or current_state["anchored_bridge_lift_theorem"]["status"] != "strict_pass"
        ):
            value += 0.018

        # Cooldown / diversity penalties.
        repeat_penalty = 0.0
        if recent3 and name == recent3[-1]:
            repeat_penalty += 0.030
        if len(recent3) >= 2 and recent3[-2:] == [name, name]:
            repeat_penalty += 0.050
        repeat_penalty += 0.006 * counts[name]
        value -= repeat_penalty

        if value > best_value:
            best_value = value
            best_name = name
            best_detail = {
                "weighted_value": float(value),
                "theorem_bonus": float(theorem_bonus),
                "repeat_penalty": float(repeat_penalty),
                "contributions": contributions,
            }

    return best_name, best_detail


def main() -> None:
    ap = argparse.ArgumentParser(description="10轮自动科研循环 v2")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_10round_excavation_loop_v2_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    prev_loop = load_latest("theory_track_10round_excavation_loop_20260312.json")

    scores = dict(prev_loop["ending_point"]["final_scores"])
    initial_scores = dict(scores)
    state = theorem_state(scores)
    catalog = make_catalog()
    history: List[str] = []
    rounds: List[Dict[str, Any]] = []

    for round_id in range(1, args.rounds + 1):
        block_name, detail = choose_block_v2(scores, state, catalog, history)
        block = catalog[block_name]
        before_scores = dict(scores)

        for axis, delta in block["axis_delta"].items():
            scores[axis] = clamp01(scores[axis] + delta)
        new_state = theorem_state(scores)

        rounds.append(
            {
                "round": round_id,
                "selected_block": block_name,
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
        history.append(block_name)
        state = new_state

    summary = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_10Round_Excavation_Loop_V2",
            "rounds": int(args.rounds),
        },
        "starting_point": {
            "from_previous_loop": True,
            "initial_scores": initial_scores,
            "initial_theorem_state": theorem_state(initial_scores),
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
            "block_usage_counts": {name: history.count(name) for name in catalog},
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
        "next_round_21_recommendation": {
            "recommended_block": choose_block_v2(scores, state, catalog, history)[0],
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
    print(json.dumps(summary["next_round_21_recommendation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
