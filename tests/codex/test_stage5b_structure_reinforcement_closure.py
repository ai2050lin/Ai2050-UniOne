#!/usr/bin/env python
"""
Score whether stage 5 can reinforce structure instead of letting the fused
objective keep routing weight away from structure.
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


def inv_gap_score(value: float, lo: float, hi: float) -> float:
    return float(1.0 - normalize(value, lo, hi))


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5B structure reinforcement closure")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage5b_structure_reinforcement_closure_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage5 = load_json(ROOT / "tests" / "codex_temp" / "stage5_fused_unified_law_objective_20260311.json")
    task2 = load_json(ROOT / "tests" / "codex_temp" / "task_block_2_unified_training_closure_20260311.json")
    atlas = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_real_model_structure_atlas_20260310.json")
    shared = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_shared_support_head_bridge_20260310.json")

    qwen_atlas = atlas["models"]["qwen3_4b"]["global_summary"]
    deepseek_atlas = atlas["models"]["deepseek_7b"]["global_summary"]
    qwen_shared = shared["models"]["qwen3_4b"]["global_summary"]
    deepseek_shared = shared["models"]["deepseek_7b"]["global_summary"]
    stage5_best = stage5["best_config"]

    foundation = {
        "task2_structure_score": normalize(
            float(task2["headline_metrics"]["structure_training_score"]),
            0.54,
            0.64,
        ),
        "stage5_training_score": normalize(float(stage5_best["training_score"]), 0.60, 0.67),
        "structure_weight_floor": inv_gap_score(float(stage5_best["training_weights"]["structure"]), 0.15, 0.40),
    }
    foundation_score = mean(foundation.values())

    shared_support = {
        "qwen_soft_layer_overlap": normalize(
            float(qwen_shared["concept_relation_soft_layer_overlap_ratio"]),
            0.30,
            0.55,
        ),
        "deepseek_soft_layer_overlap": normalize(
            float(deepseek_shared["concept_relation_soft_layer_overlap_ratio"]),
            0.30,
            0.55,
        ),
        "qwen_shared_mass": normalize(float(qwen_shared["mean_shared_mass_ratio"]), 0.02, 0.06),
        "deepseek_shared_mass": normalize(float(deepseek_shared["mean_shared_mass_ratio"]), 0.02, 0.06),
    }
    shared_support_score = mean(shared_support.values())

    real_task_structure_gain = {
        "qwen_behavior_gain": normalize(float(qwen_atlas["mean_behavior_gain"]), 0.02, 0.05),
        "deepseek_behavior_gain": normalize(float(deepseek_atlas["mean_behavior_gain"]), 0.02, 0.05),
        "qwen_compact_gain": normalize(float(qwen_shared["compact_minus_diffuse_shared_mass"]), 0.01, 0.04),
        "deepseek_compact_gain": normalize(float(deepseek_shared["compact_minus_diffuse_shared_mass"]), 0.01, 0.04),
    }
    real_task_gain_score = mean(real_task_structure_gain.values())

    anti_collapse = {
        "qwen_orientation_guard": inv_gap_score(float(qwen_atlas["orientation_gap_abs"]), 0.05, 0.30),
        "deepseek_orientation_guard": inv_gap_score(float(deepseek_atlas["orientation_gap_abs"]), 0.20, 0.80),
        "qwen_mechanism_bridge": normalize(float(qwen_atlas["mechanism_bridge_score"]), 0.70, 0.92),
        "deepseek_mechanism_bridge": normalize(float(deepseek_atlas["mechanism_bridge_score"]), 0.70, 0.92),
    }
    anti_collapse_score = mean(anti_collapse.values())

    band_support = {
        "qwen_shared_band_count": normalize(float(qwen_atlas["shared_band_layer_count"]), 3.0, 5.0),
        "deepseek_shared_band_count": normalize(float(deepseek_atlas["shared_band_layer_count"]), 3.0, 5.0),
        "qwen_targeted_count": normalize(float(qwen_atlas["targeted_layer_count"]), 1.0, 2.0),
        "deepseek_targeted_count": normalize(float(deepseek_atlas["targeted_layer_count"]), 1.0, 2.0),
    }
    band_support_score = mean(band_support.values())

    overall_score = mean(
        [
            foundation_score,
            shared_support_score,
            real_task_gain_score,
            anti_collapse_score,
            band_support_score,
        ]
    )

    hypotheses = {
        "H1_structure_foundation_is_nontrivial": bool(foundation_score >= 0.56),
        "H2_shared_support_is_visible_on_both_models": bool(shared_support_score >= 0.55),
        "H3_structure_aware_gain_stays_positive": bool(real_task_gain_score >= 0.58),
        "H4_structure_reinforcement_does_not_fully_collapse_under_orientation_mismatch": bool(
            anti_collapse_score >= 0.54
        ),
        "H5_stage5b_structure_reinforcement_is_moderately_closed": bool(overall_score >= 0.61),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage5b_structure_reinforcement_closure",
        },
        "pillars": {
            "foundation": {"components": foundation, "score": float(foundation_score)},
            "shared_support": {"components": shared_support, "score": float(shared_support_score)},
            "real_task_structure_gain": {
                "components": real_task_structure_gain,
                "score": float(real_task_gain_score),
            },
            "anti_collapse": {"components": anti_collapse, "score": float(anti_collapse_score)},
            "band_support": {"components": band_support, "score": float(band_support_score)},
        },
        "headline_metrics": {
            "foundation_score": float(foundation_score),
            "shared_support_score": float(shared_support_score),
            "real_task_structure_gain_score": float(real_task_gain_score),
            "anti_collapse_score": float(anti_collapse_score),
            "band_support_score": float(band_support_score),
            "overall_stage5b_score": float(overall_score),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "Stage 5B is positive only if structure is no longer just a weak regularizer but remains "
                "visible in shared support, real-task gain, and model-side band structure."
            ),
            "next_question": (
                "If this stage holds, the next step is to push hard online failures directly into the same "
                "training view rather than evaluating them in a separate dashboard."
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
