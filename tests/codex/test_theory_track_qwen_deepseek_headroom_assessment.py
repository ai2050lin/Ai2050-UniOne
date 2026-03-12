#!/usr/bin/env python
"""
汇总 Qwen3-4B / DeepSeek-7B 现有工件，评估当前挖掘覆盖面、剩余提升空间与主瓶颈。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def rank_gaps(gaps: Dict[str, float]) -> List[Dict[str, Any]]:
    return [
        {"gap": name, "score": float(score)}
        for name, score in sorted(gaps.items(), key=lambda item: item[1], reverse=True)
    ]


def main() -> None:
    t0 = time.time()

    mechanism = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_mechanism_bridge_20260309.json")
    structure = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_real_model_structure_atlas_20260310.json")
    recovery = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_real_model_recovery_proxy_atlas_20260310.json")
    online_chain = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_online_recovery_chain_20260310.json")
    hard_tool = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_hard_online_tool_interface_20260310.json")
    stage_heads = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_online_learnable_stage_heads_20260310.json")
    joint_proxy = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_joint_proxy_causal_intervention_20260311.json")

    output: Dict[str, Any] = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "core_constraint": "qwen_deepseek_mining_headroom_assessment",
            "runtime_sec": 0.0,
        },
        "models": {},
    }

    mapping = {
        "qwen3_4b": "Qwen3-4B",
        "deepseek_7b": "DeepSeek-7B",
    }
    for model_key, label in mapping.items():
        mech = mechanism["models"][model_key]
        struct = structure["models"][model_key]["global_summary"]
        rec = recovery["models"][model_key]["global_summary"]
        chain = online_chain["models"][model_key]["systems"]["online_recovery_aware"]
        tool_joint = hard_tool["models"][model_key]["relation_tool_joint_head_online_tool_interface"]
        stage = stage_heads["models"][model_key]["online_learnable_stage_heads"]
        proxy = joint_proxy["models"][model_key]["joint_shared_relation_recovery"]["drops"]

        coverage_terms = {
            "mechanism_bridge": float(mech["mechanism_bridge_score"]),
            "protocol_calling": float(mech["components"]["protocol_calling"]),
            "real_structure_bridge": float(struct["mechanism_bridge_score"]),
            "recovery_proxy": float(rec["recovery_proxy_score"]),
            "online_recovery_success": float(chain["success_rate"]),
            "hard_interface_success": float(tool_joint["success_rate"]),
            "learnable_stage_success": float(stage["success_rate"]),
            "orientation_stability": clamp01(1.0 - float(struct["orientation_gap_abs"])),
            "joint_proxy_robustness": clamp01(1.0 - float(proxy["joint_drop"])),
        }
        mining_completion_score = mean(list(coverage_terms.values()))
        remaining_headroom = float(1.0 - mining_completion_score)

        gaps = {
            "protocol_calling_gap": float(1.0 - coverage_terms["protocol_calling"]),
            "recovery_proxy_gap": float(1.0 - coverage_terms["recovery_proxy"]),
            "online_recovery_gap": float(1.0 - coverage_terms["online_recovery_success"]),
            "hard_interface_gap": float(1.0 - coverage_terms["hard_interface_success"]),
            "stage_head_gap": float(1.0 - coverage_terms["learnable_stage_success"]),
            "orientation_gap_penalty": float(1.0 - coverage_terms["orientation_stability"]),
            "joint_proxy_gap": float(1.0 - coverage_terms["joint_proxy_robustness"]),
        }

        output["models"][model_key] = {
            "label": label,
            "coverage_terms": coverage_terms,
            "mining_completion_score": mining_completion_score,
            "remaining_headroom": remaining_headroom,
            "gap_ranking": rank_gaps(gaps),
            "strict_readout": {
                "orientation_gap_abs": float(struct["orientation_gap_abs"]),
                "real_bridge_score": float(struct["mechanism_bridge_score"]),
            },
            "online_chain": {
                "success_rate": float(chain["success_rate"]),
                "rollback_trigger_rate": float(chain["rollback_trigger_rate"]),
                "rollback_recovery_rate": float(chain["rollback_recovery_rate"]),
            },
            "tool_interface": {
                "success_rate": float(tool_joint["success_rate"]),
                "tool_failure_rate": float(tool_joint["tool_failure_rate"]),
            },
            "stage_heads": {
                "success_rate": float(stage["success_rate"]),
                "rollback_trigger_rate": float(stage["rollback_trigger_rate"]),
            },
        }

    qwen = output["models"]["qwen3_4b"]
    deepseek = output["models"]["deepseek_7b"]
    payload = {
        **output,
        "headline_metrics": {
            "qwen_completion_score": float(qwen["mining_completion_score"]),
            "qwen_remaining_headroom": float(qwen["remaining_headroom"]),
            "deepseek_completion_score": float(deepseek["mining_completion_score"]),
            "deepseek_remaining_headroom": float(deepseek["remaining_headroom"]),
            "completion_gap_deepseek_minus_qwen": float(
                deepseek["mining_completion_score"] - qwen["mining_completion_score"]
            ),
        },
        "project_readout": {
            "summary": (
                "当前 Qwen / DeepSeek 挖掘已经不只是结构侧或 proxy 侧单点结果，而是开始形成结构、"
                "恢复、在线链、工具接口、阶段头、联合干预的多维覆盖度。"
            ),
            "next_question": (
                "如果要继续提升，不该再平均加实验，而应围绕每个模型的主缺口做大任务块："
                "Qwen 偏 protocol / basis / task-bridge，DeepSeek 偏 online relation/tool 链、"
                "orientation gap 和 recovery stability。"
            ),
        },
    }
    payload["meta"]["runtime_sec"] = float(time.time() - t0)

    out_path = ROOT / "tests" / "codex_temp" / "qwen_deepseek_mining_headroom_assessment_20260312.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    for key in ["qwen3_4b", "deepseek_7b"]:
        print(key)
        print(json.dumps(payload["models"][key]["gap_ranking"][:4], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
