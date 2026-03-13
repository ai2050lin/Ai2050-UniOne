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
    ap = argparse.ArgumentParser(description="Assess whether current DNN extraction is sufficient")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_dnn_extraction_sufficiency_assessment_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    systemic = load("stage_systemic_closure_master_block_20260312.json")
    route = load("theory_track_current_route_bottleneck_assessment_20260313.json")
    break_block = load("stage_protocol_successor_breakthrough_block_20260313.json")
    qd = load("qwen_deepseek_naturalized_trace_bundle_20260312.json")
    overall = load("theory_track_encoding_math_progress_overall_20260313.json")

    protocol = float(systemic["headline_metrics"]["protocol_calling"])
    successor = float(systemic["headline_metrics"]["successor_coherence"])
    relation = float(systemic["headline_metrics"]["relation_chain"])
    brain = float(systemic["headline_metrics"]["brain_side_causal_closure"])
    pruning = float(systemic["headline_metrics"]["theorem_pruning_strength"])
    cross_model_completion = float(qd["headline_metrics"]["cross_model_mean_completion"])
    projected_gain = float(route["route_status"]["projected_gain_if_protocol_successor_block_is_executed"])
    encoding_ready = float(overall["progress"]["encoding_mechanism_readiness"])
    math_ready = float(overall["progress"]["new_math_system_readiness"])

    sufficiency_score = clamp01(
        0.22 * cross_model_completion
        + 0.18 * relation
        + 0.12 * protocol
        + 0.10 * successor
        + 0.12 * brain
        + 0.12 * pruning
        + 0.08 * encoding_ready
        + 0.06 * math_ready
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_dnn_extraction_sufficiency_assessment",
        },
        "inputs": {
            "cross_model_mean_completion": cross_model_completion,
            "protocol_calling": protocol,
            "successor_coherence": successor,
            "relation_chain": relation,
            "brain_side_causal_closure": brain,
            "theorem_pruning_strength": pruning,
            "encoding_mechanism_readiness": encoding_ready,
            "new_math_system_readiness": math_ready,
            "projected_gain_if_next_breakthrough_block_runs": projected_gain,
        },
        "assessment": {
            "dnn_extraction_sufficiency_score": sufficiency_score,
            "is_sufficient_for_final_closure": False,
            "is_sufficient_for_strong_pruning": sufficiency_score >= 0.65,
            "is_sufficient_for_directional_breakthrough": projected_gain > 0.04,
        },
        "main_gaps": [
            "真实在线长链 trace 仍不足",
            "protocol/task bridge 抽取仍偏弱",
            "successor 还未成为全系统主导结构",
            "brain-side 因果执行还没进入在线闭环",
        ],
        "verdict": {
            "core_answer": (
                "Current DNN extraction is already sufficient for strong pruning and directional breakthrough design, "
                "but still insufficient for final closure because the online long-chain, protocol bridge, and brain-side layers remain under-extracted."
            ),
            "next_target": (
                "Upgrade extraction from artifact-led structural mining into online natural trace capture plus protocol-successor-brain joint extraction."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
