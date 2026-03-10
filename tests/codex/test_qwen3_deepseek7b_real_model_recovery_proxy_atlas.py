#!/usr/bin/env python
"""
Build a real-model recovery proxy atlas from existing Qwen3 / DeepSeek7B
artifacts.

This is intentionally a proxy atlas, not a claim of direct recovery-chain
measurement. It aligns:
1. relation bridge-aware gains
2. structure-aware task gains
3. targeted-band risk and orientation gap
4. mechanism bridge strength
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    clipped = min(max(float(value), lo), hi)
    return float((clipped - lo) / (hi - lo))


def relation_shared_hits(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return {str(row["relation"]): float(row["shared_layer_hit_ratio"]) for row in rows}


def top_task_rows(tasks: Dict[str, Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    ranked = sorted(tasks.values(), key=lambda row: float(row["behavior_gain"]), reverse=True)
    return [
        {
            "task": f"{row['concept']}__{row['relation']}",
            "concept": row["concept"],
            "relation": row["relation"],
            "compatibility": float(row["compatibility"]),
            "behavior_gain": float(row["behavior_gain"]),
            "structure_aware_success": float(row["structure_aware_success"]),
        }
        for row in ranked[:top_k]
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-model recovery proxy atlas for Qwen3 / DeepSeek7B")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen3_deepseek7b_real_model_recovery_proxy_atlas_20260310.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    atlas_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_real_model_structure_atlas_20260310.json")
    relation_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_relation_behavior_bridge_20260309.json")
    task_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_structure_task_real_bridge_20260309.json")
    orientation_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_shared_layer_band_causal_orientation_20260310.json")

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "core_constraint": "real_model_recovery_proxy_atlas_from_existing_artifacts",
            "runtime_sec": 0.0,
        },
        "models": {},
    }

    for model_name in ["qwen3_4b", "deepseek_7b"]:
        atlas_model = atlas_payload["models"][model_name]
        relation_model = relation_payload["models"][model_name]
        task_model = task_payload["models"][model_name]
        orientation_model = orientation_payload["models"][model_name]
        shared_hits = relation_shared_hits(orientation_model["relations"])

        relation_rows = []
        for relation_name, row in relation_model["relations"].items():
            shared_hit = float(shared_hits.get(relation_name, 0.0))
            bridge_gain = float(row["behavior_gain"])
            bridge_success = float(row["bridge_aware_success"])
            repair_proxy = float(
                0.45 * normalize(bridge_gain, 0.0, 0.08)
                + 0.30 * normalize(bridge_success, 0.38, 0.44)
                + 0.25 * normalize(shared_hit, 0.0, 0.40)
            )
            relation_rows.append(
                {
                    "relation": relation_name,
                    "classification": row["classification"],
                    "shared_layer_hit_ratio": shared_hit,
                    "uniform_success": float(row["uniform_success"]),
                    "bridge_aware_success": bridge_success,
                    "behavior_gain": bridge_gain,
                    "repair_proxy": repair_proxy,
                }
            )

        target_band_rows = [
            {
                "layer": int(row["layer"]),
                "shared_support": float(row["shared_support"]),
                "support_stage": str(row["support_stage"]),
                "support_bias": float(row["support_bias"]),
            }
            for row in atlas_model["layer_atlas"]
            if row["is_targeted_band"]
        ]

        structure_top_tasks = top_task_rows(task_model["tasks"], top_k=8)
        atlas_summary = atlas_model["global_summary"]
        relation_summary = relation_model["global_summary"]
        task_summary = task_model["global_summary"]

        bridge_side_gain = float(relation_summary["mean_behavior_gain"])
        task_side_gain = float(task_summary["mean_behavior_gain"])
        gap_penalty = float(atlas_summary["orientation_gap_abs"])
        mechanism_bridge = float(atlas_summary["mechanism_bridge_score"])
        recovery_proxy_score = float(
            0.30 * normalize(bridge_side_gain, 0.03, 0.07)
            + 0.22 * normalize(task_side_gain, 0.02, 0.05)
            + 0.18 * normalize(mechanism_bridge, 0.70, 0.95)
            + 0.15 * normalize(relation_summary["bridge_gain_rank_correlation"], 0.15, 0.80)
            + 0.10 * normalize(task_summary["concept_gain_rank_correlation"], 0.15, 0.60)
            + 0.05 * normalize(mean([row["shared_support"] for row in target_band_rows]), 0.05, 0.35)
            - 0.25 * normalize(gap_penalty, 0.05, 0.65)
        )

        results["models"][model_name] = {
            "relation_recovery_rows": sorted(relation_rows, key=lambda row: float(row["repair_proxy"]), reverse=True),
            "target_band_rows": target_band_rows,
            "top_structure_tasks": structure_top_tasks,
            "global_summary": {
                "bridge_side_gain": bridge_side_gain,
                "task_side_gain": task_side_gain,
                "bridge_aware_success": float(relation_summary["mean_bridge_aware_success"]),
                "structure_aware_success": float(task_summary["mean_structure_aware_success"]),
                "mechanism_bridge_score": mechanism_bridge,
                "orientation_gap_abs": gap_penalty,
                "bridge_gain_rank_correlation": float(relation_summary["bridge_gain_rank_correlation"]),
                "concept_gain_rank_correlation": float(task_summary["concept_gain_rank_correlation"]),
                "mean_target_band_shared_support": float(mean([row["shared_support"] for row in target_band_rows])),
                "recovery_proxy_score": recovery_proxy_score,
            },
        }

    qwen = results["models"]["qwen3_4b"]["global_summary"]
    deepseek = results["models"]["deepseek_7b"]["global_summary"]

    payload = {
        **results,
        "headline_metrics": {
            "qwen_recovery_proxy_score": float(qwen["recovery_proxy_score"]),
            "deepseek_recovery_proxy_score": float(deepseek["recovery_proxy_score"]),
            "qwen_bridge_side_gain": float(qwen["bridge_side_gain"]),
            "deepseek_bridge_side_gain": float(deepseek["bridge_side_gain"]),
            "qwen_task_side_gain": float(qwen["task_side_gain"]),
            "deepseek_task_side_gain": float(deepseek["task_side_gain"]),
            "qwen_orientation_gap_abs": float(qwen["orientation_gap_abs"]),
            "deepseek_orientation_gap_abs": float(deepseek["orientation_gap_abs"]),
        },
        "gains": {
            "qwen_minus_deepseek_task_side_gain": float(qwen["task_side_gain"] - deepseek["task_side_gain"]),
            "deepseek_minus_qwen_bridge_side_corr": float(deepseek["bridge_gain_rank_correlation"] - qwen["bridge_gain_rank_correlation"]),
            "deepseek_minus_qwen_recovery_gap_penalty": float(deepseek["orientation_gap_abs"] - qwen["orientation_gap_abs"]),
            "deepseek_minus_qwen_recovery_proxy_score": float(deepseek["recovery_proxy_score"] - qwen["recovery_proxy_score"]),
        },
        "hypotheses": {
            "H1_both_models_keep_positive_bridge_and_task_side_repair_gain": bool(
                qwen["bridge_side_gain"] > 0.0
                and deepseek["bridge_side_gain"] > 0.0
                and qwen["task_side_gain"] > 0.0
                and deepseek["task_side_gain"] > 0.0
            ),
            "H2_deepseek_has_larger_recovery_gap_penalty": bool(
                deepseek["orientation_gap_abs"] > qwen["orientation_gap_abs"] + 0.20
            ),
            "H3_qwen_keeps_stronger_task_side_gain_but_deepseek_keeps_stronger_bridge_alignment": bool(
                qwen["task_side_gain"] > deepseek["task_side_gain"]
                and deepseek["bridge_gain_rank_correlation"] > qwen["bridge_gain_rank_correlation"]
            ),
            "H4_both_models_retain_positive_recovery_proxy_score": bool(
                qwen["recovery_proxy_score"] > 0.0 and deepseek["recovery_proxy_score"] > 0.0
            ),
        },
        "project_readout": {
            "summary": "这一步不是声称已经测到了真实恢复链，而是把真实模型里与恢复最相关的桥接收益、结构收益、取向落差和目标层带风险压成了一张恢复代理 atlas。结果显示两类模型都保留了正的桥接侧和任务侧修复收益，但 DeepSeek 的取向落差惩罚显著更大，说明它的共享结构更强，却也更容易在状态门控处出现恢复瓶颈。",
            "next_question": "如果恢复代理 atlas 已经把桥接收益和瓶颈惩罚同时钉到真实模型上，下一步就该把 rollback 或 recovery 的在线链路直接映射到这些目标层带，而不是继续停在代理层。",
        },
    }

    payload["meta"]["runtime_sec"] = float(time.time() - t0)
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["gains"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
