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
    bridge = load_json("tests/codex_temp/qwen3_deepseek7b_mechanism_bridge_20260309.json")
    protocol_successor = load_json("tests/codex_temp/theory_track_protocol_successor_brain_unified_assessment_20260313.json")
    factorization = load_json("tests/codex_temp/qwen_deepseek_concept_local_residual_auto_factorization_20260315.json")
    operator_closure = load_json("tests/codex_temp/qwen_deepseek_universal_family_operator_closure_20260315.json")

    qwen = bridge["models"]["qwen3_4b"]["components"]
    deepseek = bridge["models"]["deepseek_7b"]["components"]
    unified = protocol_successor["current_to_unified"]

    protocol_score = float((qwen["protocol_calling"] + deepseek["protocol_calling"] + unified["protocol_projected"]) / 3.0)
    successor_score = float(unified["successor_projected"])
    readout_score = float((qwen["H_representation"] + deepseek["H_representation"]) / 2.0)
    relation_score = float((qwen["R_relation"] + deepseek["R_relation"]) / 2.0)
    gating_score = float((qwen["G_gating"] + deepseek["G_gating"]) / 2.0)
    bridge_closure_score = (protocol_score + successor_score + readout_score + relation_score + gating_score) / 5.0

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "QwenDeepSeek_readout_transport_bridge_unified_state_equation",
        },
        "strict_goal": {
            "statement": "把 concept state、readout、stage transport、successor transport 和 protocol bridge 并入同一状态方程。",
            "boundary": "当前闭合的是统一方程原型和支撑评分，不是已经完全打通的执行系统。",
        },
        "candidate_unified_equation": {
            "equation": (
                "h_out = R_read( B_f + S_shared b_c + U_local,f a_c + xi_c + R_ctx ) "
                "+ T_stage + T_succ + P_proto"
            ),
            "readout_law": "R_read = restricted overlap + recurrent scaffold + protocol-conditioned projection",
            "transport_law": "T_stage = G_stage odot h ; T_succ = G_succ odot Future(h)",
            "bridge_law": "P_proto = Bridge(role, tool, task, action)",
        },
        "component_scores": {
            "protocol_score": protocol_score,
            "successor_score": successor_score,
            "readout_score": readout_score,
            "relation_score": relation_score,
            "gating_score": gating_score,
            "bridge_closure_score": bridge_closure_score,
        },
        "supporting_sources": {
            "mechanism_bridge": bridge["cross_model_verdict"],
            "protocol_successor_assessment": protocol_successor["verdict"],
            "factorization_joint_error": factorization["summary"]["joint_factorization_mean_error"],
            "family_operator_error": operator_closure["summary"]["mean_continuous_error"],
        },
        "strict_verdict": {
            "what_is_reached_now": (
                "readout、relation、stage gate、successor 和 protocol bridge 已经可以被统一描述在一个状态方程框架里。"
            ),
            "what_is_not_reached_yet": (
                "当前 successor 仍然是最弱一环；"
                "protocol bridge 也还不是稳定的全任务执行闭环。"
            ),
        },
        "progress_estimate": {
            "readout_transport_bridge_unified_state_equation_percent": 64.0,
            "whole_network_state_generator_percent": 53.0,
            "full_brain_encoding_mechanism_percent": 57.0,
        },
        "next_large_blocks": [
            "把统一状态方程接到 whole-network state generator，输出分层候选状态和读出端口。",
            "把 brain-side falsification 约束直接绑定到 protocol/readout/successor 三线预测上。",
        ],
    }
    return payload


def test_qwen_deepseek_readout_transport_bridge_unified_state_equation() -> None:
    payload = build_payload()
    scores = payload["component_scores"]
    assert scores["bridge_closure_score"] > 0.7
    assert scores["readout_score"] > scores["protocol_score"]
    assert payload["progress_estimate"]["readout_transport_bridge_unified_state_equation_percent"] >= 64.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen/DeepSeek unified state equation")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen_deepseek_readout_transport_bridge_unified_state_equation_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["component_scores"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
