#!/usr/bin/env python
"""
Score whether stage 5 has become cross-model enough to count as unified rather
than a Qwen-only or upgrade-heavy objective.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    clipped = min(max(float(value), lo), hi)
    return float((clipped - lo) / (hi - lo))


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5D cross-model unified calibration closure")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage5d_cross_model_unified_calibration_closure_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    task3 = load_json(ROOT / "tests" / "codex_temp" / "task_block_3_real_model_task_bridge_closure_20260311.json")
    stage5 = load_json(ROOT / "tests" / "codex_temp" / "stage5_fused_unified_law_objective_20260311.json")
    mech = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_mechanism_bridge_20260309.json")
    atlas = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_real_model_structure_atlas_20260310.json")
    shared = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_shared_support_head_bridge_20260310.json")

    qwen_mech = mech["models"]["qwen3_4b"]
    deepseek_mech = mech["models"]["deepseek_7b"]
    qwen_atlas = atlas["models"]["qwen3_4b"]["global_summary"]
    deepseek_atlas = atlas["models"]["deepseek_7b"]["global_summary"]
    qwen_shared = shared["models"]["qwen3_4b"]["global_summary"]
    deepseek_shared = shared["models"]["deepseek_7b"]["global_summary"]
    task3_metrics = task3["headline_metrics"]
    stage5_best = stage5["best_config"]

    qwen_score_parts = {
        "mechanism_bridge": float(qwen_mech["mechanism_bridge_score"]),
        "task_bridge": float(task3_metrics["qwen_task_bridge_score"]),
        "shared_basis": float(qwen_mech["components"]["shared_basis"]),
        "soft_layer_overlap": float(qwen_shared["concept_relation_soft_layer_overlap_ratio"]),
        "behavior_gain": normalize(float(qwen_atlas["mean_behavior_gain"]), 0.02, 0.05),
    }
    qwen_score = mean(qwen_score_parts.values())

    deepseek_score_parts = {
        "mechanism_bridge": float(deepseek_mech["mechanism_bridge_score"]),
        "task_bridge": float(task3_metrics["deepseek_task_bridge_score"]),
        "shared_basis": float(deepseek_mech["components"]["shared_basis"]),
        "soft_layer_overlap": float(deepseek_shared["concept_relation_soft_layer_overlap_ratio"]),
        "behavior_gain": normalize(float(deepseek_atlas["mean_behavior_gain"]), 0.02, 0.05),
    }
    deepseek_score = mean(deepseek_score_parts.values())

    cross_model_upgrade = {
        "task3_upgrade_score": float(task3_metrics["cross_model_upgrade_score"]),
        "stage5_online_score": float(stage5_best["online_score"]),
        "stage5_brain_score": float(stage5_best["brain_score"]),
    }
    cross_model_upgrade_score = mean(cross_model_upgrade.values())

    model_balance = {
        "qwen_deepseek_gap_guard": float(max(0.0, 1.0 - abs(qwen_score - deepseek_score) / 0.50)),
        "bridge_gap_guard": float(
            max(
                0.0,
                1.0
                - abs(
                    float(task3_metrics["deepseek_task_bridge_score"]) - float(task3_metrics["qwen_task_bridge_score"])
                )
                / 0.25,
            )
        ),
    }
    model_balance_score = mean(model_balance.values())

    consistency = {
        "consistent_component_ratio": float(
            float(mech["cross_model_verdict"]["summary"]["n_consistent"]) / 6.0
        ),
        "overall_verdict_guard": 1.0 if mech["cross_model_verdict"]["overall_verdict"] == "mostly_consistent" else 0.0,
        "qwen_shared_basis": normalize(float(qwen_mech["components"]["shared_basis"]), 0.60, 0.95),
        "deepseek_shared_basis": normalize(float(deepseek_mech["components"]["shared_basis"]), 0.60, 0.95),
    }
    consistency_score = mean(consistency.values())

    overall_score = mean(
        [qwen_score, deepseek_score, cross_model_upgrade_score, model_balance_score, consistency_score]
    )

    hypotheses = {
        "H1_qwen_side_is_nontrivially_calibrated": bool(qwen_score >= 0.50),
        "H2_deepseek_side_is_nontrivially_calibrated": bool(deepseek_score >= 0.62),
        "H3_cross_model_upgrade_signal_is_strong": bool(cross_model_upgrade_score >= 0.66),
        "H4_cross_model_balance_is_no_longer_collapsed": bool(model_balance_score >= 0.40),
        "H5_stage5d_cross_model_calibration_is_moderately_closed": bool(overall_score >= 0.61),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage5d_cross_model_unified_calibration_closure",
        },
        "pillars": {
            "qwen": {"components": qwen_score_parts, "score": float(qwen_score)},
            "deepseek": {"components": deepseek_score_parts, "score": float(deepseek_score)},
            "cross_model_upgrade": {
                "components": cross_model_upgrade,
                "score": float(cross_model_upgrade_score),
            },
            "model_balance": {"components": model_balance, "score": float(model_balance_score)},
            "consistency": {"components": consistency, "score": float(consistency_score)},
        },
        "headline_metrics": {
            "qwen_calibration_score": float(qwen_score),
            "deepseek_calibration_score": float(deepseek_score),
            "cross_model_upgrade_score": float(cross_model_upgrade_score),
            "model_balance_score": float(model_balance_score),
            "consistency_score": float(consistency_score),
            "overall_stage5d_score": float(overall_score),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "Stage 5D is positive only if the fused objective is no longer just a stronger-route proxy and "
                "instead remains visible on both Qwen and DeepSeek with a tolerable balance gap."
            ),
            "next_question": (
                "If this stage holds, stage 5 can be treated as closed and the next frontier becomes a larger "
                "stage-6 target: causal compression of the unified law into a smaller trainable core."
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
