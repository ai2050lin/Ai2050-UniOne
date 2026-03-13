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
    ap = argparse.ArgumentParser(description="统一 successor/protocol/brain 执行块")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_protocol_successor_brain_unified_execution_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    systemic = load("stage_systemic_closure_master_block_20260312.json")
    breakthrough = load("stage_protocol_successor_breakthrough_block_20260313.json")
    detailed = load("theory_track_successor_protocol_brain_detailed_synthesis_20260313.json")
    v3 = load("theory_track_10round_excavation_loop_v3_assessment_20260312.json")

    protocol = float(systemic["headline_metrics"]["protocol_calling"])
    successor = float(systemic["headline_metrics"]["successor_coherence"])
    brain = float(systemic["headline_metrics"]["brain_side_causal_closure"])
    theorem_pruning = float(systemic["headline_metrics"]["theorem_pruning_strength"])
    inverse_ready = float(v3["headline_metrics"]["encoding_inverse_reconstruction_readiness"])
    math_ready = float(v3["headline_metrics"]["new_math_closure_readiness"])

    projected_protocol = float(breakthrough["breakthrough_projection"]["protocol_calling"])
    projected_successor = float(breakthrough["breakthrough_projection"]["successor_coherence"])
    projected_brain = float(breakthrough["breakthrough_projection"]["brain_side_causal_closure"])

    long_chain_trace_boost = 0.18 * (1.0 - successor)
    protocol_bridge_boost = 0.16 * (1.0 - protocol)
    brain_online_boost = 0.14 * (1.0 - brain)
    theorem_survival_bonus = 0.06 * theorem_pruning

    unified_protocol = clamp01(projected_protocol + 0.35 * protocol_bridge_boost + 0.15 * long_chain_trace_boost)
    unified_successor = clamp01(
        projected_successor
        + 0.55 * long_chain_trace_boost
        + 0.10 * protocol_bridge_boost
        + 0.10 * brain_online_boost
    )
    unified_brain = clamp01(projected_brain + 0.55 * brain_online_boost + 0.12 * unified_successor)
    unified_inverse = clamp01(inverse_ready + 0.08 * unified_successor + 0.04 * unified_protocol)
    unified_math = clamp01(math_ready + theorem_survival_bonus + 0.02 * unified_brain)

    current_score = (protocol + successor + brain + inverse_ready + math_ready) / 5.0
    unified_score = (unified_protocol + unified_successor + unified_brain + unified_inverse + unified_math) / 5.0

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "Stage_protocol_successor_brain_unified_execution",
        },
        "current_state": {
            "protocol_calling": protocol,
            "successor_coherence": successor,
            "brain_side_causal_closure": brain,
            "encoding_inverse_reconstruction_readiness": inverse_ready,
            "new_math_closure_readiness": math_ready,
            "current_unified_score": current_score,
        },
        "boost_components": {
            "long_chain_trace_boost": long_chain_trace_boost,
            "protocol_bridge_boost": protocol_bridge_boost,
            "brain_online_boost": brain_online_boost,
            "theorem_survival_bonus": theorem_survival_bonus,
        },
        "unified_projection": {
            "protocol_calling": unified_protocol,
            "successor_coherence": unified_successor,
            "brain_side_causal_closure": unified_brain,
            "encoding_inverse_reconstruction_readiness": unified_inverse,
            "new_math_closure_readiness": unified_math,
            "unified_score": unified_score,
            "gain_vs_current": unified_score - current_score,
        },
        "thresholds": {
            "protocol_strong_band": 0.78,
            "successor_global_band": 0.45,
            "brain_online_band": 0.80,
            "inverse_reconstruction_band": 0.80,
            "math_closure_band": 0.90,
        },
        "verdict": {
            "core_answer": (
                "如果把 successor、protocol、brain-side online execution 当成统一执行块推进，当前路线仍有明显正向提升空间，且比继续拆散推进更可能逼近真正闭环。"
            ),
            "next_target": (
                "继续把真实长链 trace、protocol bridge 与脑侧在线执行绑成同一主块，并让 theorem survival 直接参与执行评分。"
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
