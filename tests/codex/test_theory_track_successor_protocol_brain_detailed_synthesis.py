from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load(name: str) -> dict:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def gap(x: float) -> float:
    return float(1.0 - x)


def main() -> None:
    ap = argparse.ArgumentParser(description="Detailed synthesis of successor + protocol + brain-side online execution")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_successor_protocol_brain_detailed_synthesis_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    systemic = load("stage_systemic_closure_master_block_20260312.json")
    succ_diag = load("theory_track_successor_coherence_closure_diagnosis_20260312.json")
    qd = load("qwen_deepseek_naturalized_trace_bundle_20260312.json")
    route = load("theory_track_current_route_bottleneck_assessment_20260313.json")
    block = load("stage_protocol_successor_breakthrough_block_20260313.json")
    frontier = load("theory_track_protocol_successor_breakthrough_frontier_20260313.json")

    protocol = float(systemic["headline_metrics"]["protocol_calling"])
    successor = float(systemic["headline_metrics"]["successor_coherence"])
    brain = float(systemic["headline_metrics"]["brain_side_causal_closure"])
    theorem_pruning = float(systemic["headline_metrics"]["theorem_pruning_strength"])
    p3_gain = float(route["route_status"]["projected_gain_if_protocol_successor_block_is_executed"])
    online_trace_gap = float(qd["missing_axes"][0]["gap"])

    projected = block["breakthrough_projection"]
    thresholds = block["thresholds"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_successor_protocol_brain_detailed_synthesis",
        },
        "successor_block": {
            "current_score": successor,
            "current_gap": gap(successor),
            "local_theorem_support": succ_diag["closure_status"]["effective_successor_bundle"],
            "global_score": succ_diag["closure_status"]["observed_global_successor_score"],
            "closure_threshold_hint": succ_diag["closure_status"]["closure_band_threshold_hint"],
            "online_trace_gap": online_trace_gap,
            "projected_after_breakthrough_block": float(projected["successor_coherence"]),
            "projected_pass_global_support": bool(frontier["projected_status"]["successor_global_support_pass"]),
        },
        "protocol_block": {
            "current_score": protocol,
            "current_gap": gap(protocol),
            "meaning": "concept/readout/successor 是否能稳定穿过 protocol/task bridge",
            "projected_after_breakthrough_block": float(projected["protocol_calling"]),
            "projected_pass_strong_bridge": bool(frontier["projected_status"]["protocol_bridge_strong_pass"]),
        },
        "brain_block": {
            "current_score": brain,
            "current_gap": gap(brain),
            "meaning": "brain-side causal closure 是否已经从 probe/status 进入在线执行",
            "projected_after_breakthrough_block": float(projected["brain_side_causal_closure"]),
            "projected_pass_online_execution": bool(frontier["projected_status"]["brain_online_execution_pass"]),
        },
        "joint_status": {
            "theorem_pruning_strength": theorem_pruning,
            "projected_joint_gain": p3_gain,
            "breakthrough_block_gain": float(projected["gain_vs_current"]),
            "route_fundamentally_blocked": bool(route["route_status"]["current_route_is_fundamentally_blocked"]),
            "route_currently_insufficient": bool(route["route_status"]["current_route_is_currently_insufficient"]),
        },
        "verdict": {
            "core_answer": (
                "successor 是推理链前后继结构的编码层，protocol 是让概念/读出/后继真正贯通到任务与工具层的桥层，"
                "brain-side online execution 是把这套结构从模型内闭环推进到脑侧因果闭环的验证层。三者必须一起推进。"
            ),
            "next_target": (
                "先把 protocol 和 successor 提到强支撑带，再把 brain-side causal closure 从在线准备态推进到真实在线执行。"
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
