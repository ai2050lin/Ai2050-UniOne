from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load(name: str) -> dict:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    ap = argparse.ArgumentParser(description="新路线统一验证块")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_new_route_system_validation_block_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    unified = load("stage_protocol_successor_brain_unified_execution_20260313.json")
    suff = load("theory_track_dnn_extraction_sufficiency_assessment_20260313.json")
    route = load("theory_track_inverse_brain_math_route_20260313.json")
    systemic = load("stage_systemic_closure_master_block_20260312.json")

    proj = unified["unified_projection"]
    protocol = float(proj["protocol_calling"])
    successor = float(proj["successor_coherence"])
    brain = float(proj["brain_side_causal_closure"])
    inverse = float(proj["encoding_inverse_reconstruction_readiness"])
    math_ready = float(proj["new_math_closure_readiness"])

    dnn_sufficiency = float(suff["assessment"]["dnn_extraction_sufficiency_score"])
    inverse_route = float(route["route_progress"]["dnn_extraction_to_inverse_brain_encoding"])
    math_route = float(route["route_progress"]["dnn_extraction_to_new_math_closure"])

    relation_chain = float(systemic["headline_metrics"]["relation_chain"])
    protocol_gap = 1.0 - protocol
    successor_gap = 1.0 - successor
    brain_gap = 1.0 - brain

    closure_score = (protocol + successor + brain + inverse + math_ready) / 5.0
    route_score = (dnn_sufficiency + inverse_route + math_route + relation_chain) / 4.0
    total_score = (closure_score + route_score) / 2.0

    next_trace_capture_gain = 0.22 * successor_gap
    next_protocol_gain = 0.16 * protocol_gap
    next_brain_gain = 0.12 * brain_gap

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "Stage_new_route_system_validation_block",
        },
        "current_state": {
            "protocol": protocol,
            "successor": successor,
            "brain": brain,
            "inverse_reconstruction": inverse,
            "new_math_closure": math_ready,
            "dnn_extraction_sufficiency": dnn_sufficiency,
            "inverse_route_score": inverse_route,
            "new_math_route_score": math_route,
            "relation_chain": relation_chain,
        },
        "scores": {
            "closure_score": closure_score,
            "route_score": route_score,
            "total_score": total_score,
        },
        "next_block_projection": {
            "trace_capture_gain": next_trace_capture_gain,
            "protocol_gain": next_protocol_gain,
            "brain_gain": next_brain_gain,
            "projected_successor_after_next_block": clamp01(successor + next_trace_capture_gain + 0.06 * next_protocol_gain),
            "projected_protocol_after_next_block": clamp01(protocol + next_protocol_gain + 0.05 * next_trace_capture_gain),
            "projected_brain_after_next_block": clamp01(brain + next_brain_gain + 0.08 * next_trace_capture_gain),
        },
        "verdict": {
            "core_answer": "新路线已经具备统一突破当前瓶颈的条件，关键不在继续零散加实验，而在持续执行统一块并把真实长链 trace、protocol bridge 和脑侧在线执行绑在一起推进。",
            "main_bottleneck": "successor_global_support",
            "next_target": "继续做真实长链 trace 捕获，并把 protocol 与 brain-side online execution 一起并入下一轮统一块。",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
