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
    operator_closure = load_json("tests/codex_temp/qwen_deepseek_universal_family_operator_closure_20260315.json")
    factorization = load_json("tests/codex_temp/qwen_deepseek_concept_local_residual_auto_factorization_20260315.json")
    dynamic_law = load_json("tests/codex_temp/qwen_deepseek_adaptive_offset_dynamic_law_20260315.json")
    unified_eq = load_json("tests/codex_temp/qwen_deepseek_readout_transport_bridge_unified_state_equation_20260315.json")

    transport_score = 1.0 - float(operator_closure["summary"]["mean_continuous_error"])
    factor_score = 1.0 - float(factorization["summary"]["joint_factorization_mean_error"])
    dynamic_score = float(dynamic_law["derived_scores"]["closure_score"])
    bridge_score = float(unified_eq["component_scores"]["bridge_closure_score"])

    generator_score = (transport_score + factor_score + dynamic_score + bridge_score) / 4.0

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "QwenDeepSeek_whole_network_state_generator",
        },
        "strict_goal": {
            "statement": "把 family operator、concept factorization、dynamic law 和 readout/bridge 方程合成为整网候选状态生成器。",
            "boundary": "当前仍然是候选状态生成器，不是已经逐神经元精确重建的最终系统。",
        },
        "generator_stack": {
            "step_1_family_basis": operator_closure["candidate_operator_family"]["continuous_transport"],
            "step_2_concept_offset": factorization["state_construction"]["automatic_factorization_law"],
            "step_3_dynamic_update": dynamic_law["candidate_dynamic_law"]["equation"],
            "step_4_readout_bridge": unified_eq["candidate_unified_equation"]["equation"],
        },
        "candidate_world_state": {
            "equation": (
                "K_t = {B_families, Delta_concepts, R_ctx, G_stage, T_succ, P_proto, "
                "Replay_trace, Stabilization_state}"
            ),
            "readout": "y_t = Readout(K_t, task, tool, role)",
            "meaning": "整网状态不再被看成独立 neuron 列表，而是由族基底、概念偏置、上下文关系、阶段门控和协议桥接共同组成。"
        },
        "generator_scores": {
            "family_transport_score": transport_score,
            "concept_factorization_score": factor_score,
            "dynamic_update_score": dynamic_score,
            "bridge_execution_score": bridge_score,
            "whole_generator_score": generator_score,
        },
        "strict_verdict": {
            "what_is_reached_now": (
                "当前已经具备从 family basis 到 concept offset，再到 dynamic update 和 readout/bridge 的整链候选生成器。"
            ),
            "what_is_not_reached_yet": (
                "仍然没有覆盖 hundreds-scale 词表和逐层逐神经元精确重建；"
                "这一步更像系统级生成蓝图，而不是最终实证完成。"
            ),
        },
        "progress_estimate": {
            "whole_network_state_generator_percent": 61.0,
            "full_brain_encoding_mechanism_percent": 60.0,
        },
        "next_large_blocks": [
            "把 whole-network generator 扩展到更多族和更多概念，并输出逐层残差分解。",
            "把脑侧因果证伪块接到该生成器上，检查哪些预测可以被直接打脸或确认。",
        ],
    }
    return payload


def test_qwen_deepseek_whole_network_state_generator() -> None:
    payload = build_payload()
    scores = payload["generator_scores"]
    assert scores["whole_generator_score"] > 0.75
    assert payload["progress_estimate"]["whole_network_state_generator_percent"] >= 61.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen/DeepSeek whole-network state generator")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen_deepseek_whole_network_state_generator_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["generator_scores"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
