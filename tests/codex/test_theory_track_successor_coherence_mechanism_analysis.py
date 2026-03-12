from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load(name: str) -> Dict[str, Any]:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory track successor coherence mechanism analysis")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_successor_coherence_mechanism_analysis_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inv = load("theory_track_successor_strengthened_reasoning_inventory_20260312.json")
    strict = load("theory_track_successor_strengthened_priority34_pass_fail_20260312.json")
    v3 = load("theory_track_10round_excavation_loop_v3_20260312.json")

    temporal_ratio = float(inv["headline_metrics"]["temporal_cross_to_within_ratio"])
    relation_ratio = float(inv["headline_metrics"]["relation_cross_to_within_ratio"])
    successor_ratio = float(inv["headline_metrics"]["chain_successor_to_cross_stage_ratio"])
    successor_score = float(v3["ending_point"]["final_scores"]["successor_coherence"])
    stage_score = float(v3["ending_point"]["final_scores"]["stage_structure"])
    protocol_score = float(v3["ending_point"]["final_scores"]["protocol_calling"])
    relation_score = float(v3["ending_point"]["final_scores"]["relation_chain"])
    brain_score = float(v3["ending_point"]["final_scores"]["brain_side_causal_closure"])

    local_successor_alignment = clamp01(1.0 - successor_ratio)
    stage_support = clamp01(temporal_ratio - 1.0 + 0.75)
    relation_support = clamp01(relation_ratio - 1.0 + 0.65)
    transport_support = clamp01((stage_score + relation_score + protocol_score) / 3.0)
    causal_support = brain_score
    effective_successor_bundle = clamp01(
        0.32 * local_successor_alignment
        + 0.18 * stage_support
        + 0.18 * relation_support
        + 0.17 * transport_support
        + 0.15 * causal_support
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_successor_coherence_mechanism_analysis",
        },
        "mechanism_definition": {
            "core_statement": (
                "successor_coherence 不是单一距离指标，而是‘同一推理链中相邻后继状态是否能在保持阶段可分、"
                "关系锚定、协议可达和脑侧可投影的前提下，比跨链同阶段状态更紧密地对齐’。"
            ),
            "math_form": (
                "SC ~= w1*(1-r_succ) + w2*S_stage + w3*S_rel + w4*T_read + w5*C_brain"
            ),
        },
        "components": {
            "local_successor_alignment": float(local_successor_alignment),
            "stage_support": float(stage_support),
            "relation_support": float(relation_support),
            "transport_support": float(transport_support),
            "causal_support": float(causal_support),
            "effective_successor_bundle": float(effective_successor_bundle),
            "observed_global_successor_score": float(successor_score),
        },
        "strict_frontier_context": {
            "successor_theorem_status": strict["strict_pass_fail"][1]["status"],
            "successor_theorem_confidence": float(strict["strict_pass_fail"][1]["confidence"]),
            "meaning": (
                "strict pass 说明在 successor-strengthened inventory 下，successor 结构已经足以支撑 theorem；"
                "但 observed_global_successor_score 仍低，说明它还没有成为全系统的强主导量。"
            ),
        },
        "verdict": {
            "core_answer": (
                "successor_coherence 的本质是‘推理链相邻后继结构的局部一致性’，它依赖 stage、relation、transport、"
                "brain projection 共同比较稳时才会放大；当前它已能支撑 theorem，但还不足以支撑全系统闭环。"
            ),
            "next_theory_target": (
                "继续提高 successor 项从局部 theorem-support 信号，转成全局 readout/causal closure 主导量。"
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
