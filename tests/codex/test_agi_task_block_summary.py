#!/usr/bin/env python
"""
Aggregate the current A/B/C/D task blocks into one frontend-friendly payload.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate AGI task blocks A/B/C/D")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/agi_task_block_summary_20260309.json")
    args = ap.parse_args()

    block_a = load_json(ROOT / "tests" / "codex_temp" / "real_multistep_memory_learnable_state_machine_20260309.json")
    block_a_long = load_json(ROOT / "tests" / "codex_temp" / "real_multistep_memory_learnable_state_machine_long_validation_20260309.json")
    block_b = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_relation_behavior_bridge_20260309.json")
    block_bc = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_structure_task_real_bridge_20260309.json")
    block_c = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_concept_encoding_decomposition_20260309.json")
    block_d_residual = load_json(ROOT / "tests" / "codex_temp" / "continuous_input_grounding_consolidation_law_scan_20260309.json")
    block_d_bayes = load_json(ROOT / "tests" / "codex_temp" / "continuous_input_grounding_bayesian_consolidation_scan_20260309.json")
    block_d_learned = load_json(ROOT / "tests" / "codex_temp" / "continuous_input_grounding_learned_controller_scan_20260309.json")
    block_d_two_phase = load_json(ROOT / "tests" / "codex_temp" / "continuous_input_grounding_two_phase_consolidation_scan_20260309.json")
    block_d_three_phase = load_json(ROOT / "tests" / "codex_temp" / "continuous_input_grounding_three_phase_consolidation_scan_20260309.json")
    block_d_base_offset = load_json(ROOT / "tests" / "codex_temp" / "continuous_input_grounding_base_offset_consolidation_scan_20260309.json")
    block_d_offset_stab = load_json(ROOT / "tests" / "codex_temp" / "continuous_input_grounding_offset_stabilization_scan_20260309.json")
    block_d_multistage = load_json(ROOT / "tests" / "codex_temp" / "continuous_input_grounding_multistage_stabilization_scan_20260309.json")
    block_d_multimodal = load_json(ROOT / "tests" / "codex_temp" / "continuous_multimodal_grounding_proto_20260309.json")

    b_qwen = block_b["models"]["qwen3_4b"]["global_summary"]
    b_deepseek = block_b["models"]["deepseek_7b"]["global_summary"]
    bc_qwen = block_bc["models"]["qwen3_4b"]["global_summary"]
    bc_deepseek = block_bc["models"]["deepseek_7b"]["global_summary"]
    c_qwen = block_c["models"]["qwen3_4b"]["global_summary"]
    c_deepseek = block_c["models"]["deepseek_7b"]["global_summary"]

    a_first = block_a["best_machine"]
    a_long_gain_mean = float(block_a_long["gains"]["mean_gain_vs_single_anchor"])
    a_long_gain_min = float(block_a_long["gains"]["min_gain_vs_single_anchor"])

    d_best_dual = block_d_residual["best_dual_positive"] or {"novel_gain": 0.0, "retention_gain": 0.0, "overall_gain": 0.0}
    d_best_bayes = block_d_bayes["best_dual_positive"] or {"novel_gain": 0.0, "retention_gain": 0.0, "overall_gain": 0.0}
    d_best_learned = block_d_learned["best_overall"] or {"novel_gain": 0.0, "retention_gain": 0.0, "overall_gain": 0.0}
    d_best_two = block_d_two_phase["best_overall"] or {"novel_gain": 0.0, "retention_gain": 0.0, "overall_gain": 0.0}
    d_best_three = block_d_three_phase["best_overall"] or {"novel_gain": 0.0, "retention_gain": 0.0, "overall_gain": 0.0}
    d_best_base_offset = block_d_base_offset["top_overall"][0]
    d_best_offset_stab = block_d_offset_stab["top_overall"][0]
    d_best_multistage = block_d_multistage["top_overall"][0]
    d_multi_gain = block_d_multimodal["gains_vs_direct"]

    summary = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "current_stage": "M5",
        },
        "blocks": {
            "A": {
                "title": "超长程状态与信用分配",
                "status": "partial",
                "headline_score": float(a_first["max_length_score"]),
                "sub_score": a_long_gain_mean,
                "statement": "A 已经出现第一版突破，但把状态机拉到更长 horizon 后，平均增益重新转负，所以现在更准确的状态仍是“部分完成”。",
                "metrics": {
                    "first_win_at_L32": float(block_a["gains"]["best_machine_vs_single_anchor_at_max_length"]),
                    "best_machine_max_length_score": float(a_first["max_length_score"]),
                    "long_validation_mean_gain": a_long_gain_mean,
                    "long_validation_min_gain": a_long_gain_min,
                },
            },
            "B": {
                "title": "关系分型到任务收益",
                "status": "completed",
                "headline_score": float((b_qwen["mean_behavior_gain"] + b_deepseek["mean_behavior_gain"]) / 2.0),
                "sub_score": float((bc_qwen["mean_behavior_gain"] + bc_deepseek["mean_behavior_gain"]) / 2.0),
                "statement": "B 已从结构解释推进到任务层：relation typing 现在不仅能解释 synthetic bridge，也开始稳定提升 concept-conditioned 结构任务。",
                "metrics": {
                    "relation_bridge_mean_gain": float((b_qwen["mean_behavior_gain"] + b_deepseek["mean_behavior_gain"]) / 2.0),
                    "relation_bridge_mean_rank_corr": float((b_qwen["bridge_gain_rank_correlation"] + b_deepseek["bridge_gain_rank_correlation"]) / 2.0),
                    "structure_task_mean_gain": float((bc_qwen["mean_behavior_gain"] + bc_deepseek["mean_behavior_gain"]) / 2.0),
                    "structure_task_mean_rank_corr": float((bc_qwen["concept_gain_rank_correlation"] + bc_deepseek["concept_gain_rank_correlation"]) / 2.0),
                },
            },
            "C": {
                "title": "概念编码分解与任务桥接",
                "status": "completed",
                "headline_score": float((c_qwen["mean_margin_vs_best_wrong"] + c_deepseek["mean_margin_vs_best_wrong"]) / 2.0),
                "sub_score": float((bc_qwen["concept_gain_rank_correlation"] + bc_deepseek["concept_gain_rank_correlation"]) / 2.0),
                "statement": "C 不再停留在 `B_f + Delta_c` 的静态分解。当前编码分解质量已经开始对概念条件任务增益产生预测力。",
                "metrics": {
                    "qwen_margin": float(c_qwen["mean_margin_vs_best_wrong"]),
                    "deepseek_margin": float(c_deepseek["mean_margin_vs_best_wrong"]),
                    "qwen_royalty_gap": float(c_qwen["royalty_axis_gap"]),
                    "deepseek_royalty_gap": float(c_deepseek["royalty_axis_gap"]),
                    "qwen_task_corr": float(bc_qwen["concept_gain_rank_correlation"]),
                    "deepseek_task_corr": float(bc_deepseek["concept_gain_rank_correlation"]),
                },
            },
            "D": {
                "title": "连续输入接地与整合律",
                "status": "partial",
                "headline_score": float(d_best_learned["overall_gain"]),
                "sub_score": float(d_multi_gain["grounding_score_gain"]),
                "statement": "D 现在已经明确是固定点动力学问题。residual-gate 能打开 dual-positive，Bayesian 能压低 overall 屏障；learned controller 与最新的 multistage stabilization 更靠近 retention-first 一侧，而 two-phase、three-phase、base+offset 统一律以及 offset-stabilization 门都仍然落在 novel-first 一侧。",
                "metrics": {
                    "residual_dual_positive_count": float(block_d_residual["dual_positive_count"]),
                    "residual_best_overall_gain": float(d_best_dual["overall_gain"]),
                    "bayes_best_overall_gain": float(d_best_bayes["overall_gain"]),
                    "learned_best_overall_gain": float(d_best_learned["overall_gain"]),
                    "two_phase_overall_gain": float(d_best_two["overall_gain"]),
                    "three_phase_overall_gain": float(d_best_three["overall_gain"]),
                    "base_offset_best_overall_gain": float(d_best_base_offset["overall_gain"]),
                    "base_offset_best_novel_gain": float(d_best_base_offset["novel_gain"]),
                    "base_offset_best_retention_gain": float(d_best_base_offset["retention_gain"]),
                    "offset_stabilization_best_overall_gain": float(d_best_offset_stab["overall_gain"]),
                    "multistage_best_overall_gain": float(d_best_multistage["overall_gain"]),
                    "multistage_best_novel_gain": float(d_best_multistage["novel_gain"]),
                    "multistage_best_retention_gain": float(d_best_multistage["retention_gain"]),
                    "multimodal_grounding_gain": float(d_multi_gain["grounding_score_gain"]),
                    "multimodal_consistency_gain": float(d_multi_gain["crossmodal_consistency_gain"]),
                },
            },
        },
        "next_plan": [
            "A：把状态机升级成更稳的长 horizon 版本，而不是只在 L=32 一侧获胜。",
            "B：继续把 relation typing 推进到更真实的多步任务和失效模式分析。",
            "C：把概念编码分解扩到更大概念域，并和协议场调用直接闭环。",
            "D：停止细扫当前固定点族，转向能穿过 retention-first / novel-first 鞍点区的显式多阶段整合律。",
        ],
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
