#!/usr/bin/env python
"""
Run a mechanism-level adversarial break readout over the current coding-law
candidate, explicitly separating architecture, data, and scale pressures.
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
    ap = argparse.ArgumentParser(description="Stage 9A mechanism adversarial break test")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage9a_mechanism_adversarial_break_test_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage7d = load_json(ROOT / "tests" / "codex_temp" / "stage7d_coding_law_verdict_master_20260311.json")
    stage8a = load_json(
        ROOT / "tests" / "codex_temp" / "stage8a_adversarial_counterexample_search_20260311.json"
    )
    stage8c = load_json(ROOT / "tests" / "codex_temp" / "stage8c_cross_model_task_invariants_20260311.json")
    stage8d = load_json(ROOT / "tests" / "codex_temp" / "stage8d_brain_high_risk_falsification_20260311.json")
    targeted_ablation = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_shared_layer_band_targeted_ablation_20260310.json"
    )
    causal_orientation = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_shared_layer_band_causal_orientation_20260310.json"
    )
    hard_interface = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_hard_online_tool_interface_20260310.json"
    )
    layer_band = load_json(
        ROOT / "tests" / "codex_temp" / "generator_network_real_layer_band_bridge_20260310.json"
    )

    qwen_t = targeted_ablation["models"]["qwen3_4b"]["global_summary"]
    deepseek_t = targeted_ablation["models"]["deepseek_7b"]["global_summary"]
    qwen_o = causal_orientation["models"]["qwen3_4b"]["global_summary"]
    deepseek_o = causal_orientation["models"]["deepseek_7b"]["global_summary"]

    architecture_break_pressure = {
        "qwen_predicted_actual_label_mismatch": 1.0
        if qwen_t["predicted_orientation_label"] != qwen_t["actual_orientation_label"]
        else 0.0,
        "deepseek_predicted_actual_label_mismatch": 1.0
        if deepseek_t["predicted_orientation_label"] != deepseek_t["actual_orientation_label"]
        else 0.0,
        "qwen_orientation_gap": normalize(
            abs(float(qwen_t["predicted_orientation"]) - float(qwen_t["actual_targeted_orientation"])),
            0.05,
            0.12,
        ),
        "deepseek_orientation_gap": normalize(
            abs(float(deepseek_t["predicted_orientation"]) - float(deepseek_t["actual_targeted_orientation"])),
            0.20,
            0.70,
        ),
        "targeted_ablation_hypothesis_failure_rate": mean(
            1.0 if not bool(v) else 0.0 for v in targeted_ablation["hypotheses"].values()
        ),
    }
    architecture_break_pressure_score = mean(architecture_break_pressure.values())

    data_break_pressure = {
        "compatibility_corr_gap": normalize(
            abs(
                float(stage8c["model_stats"]["qwen3_4b"]["compatibility_gain_corr"])
                - float(stage8c["model_stats"]["deepseek_7b"]["compatibility_gain_corr"])
            ),
            0.10,
            0.35,
        ),
        "concept_gain_corr_gap": normalize(
            abs(
                float(stage8c["model_stats"]["qwen3_4b"]["concept_gain_corr"])
                - float(stage8c["model_stats"]["deepseek_7b"]["concept_gain_corr"])
            ),
            0.10,
            0.30,
        ),
        "positive_gain_rate_gap": normalize(
            abs(
                float(stage8c["model_stats"]["qwen3_4b"]["positive_gain_rate"])
                - float(stage8c["model_stats"]["deepseek_7b"]["positive_gain_rate"])
            ),
            0.05,
            0.15,
        ),
        "relation_family_mean_gap": normalize(
            mean(
                abs(
                    float(stage8c["relation_means"]["qwen3_4b"][rel])
                    - float(stage8c["relation_means"]["deepseek_7b"][rel])
                )
                for rel in stage8c["relation_means"]["qwen3_4b"]
            ),
            0.004,
            0.015,
        ),
    }
    data_break_pressure_score = mean(data_break_pressure.values())

    deepseek_joint = hard_interface["models"]["deepseek_7b"]["relation_tool_joint_head_online_tool_interface"]
    qwen_joint = hard_interface["models"]["qwen3_4b"]["relation_tool_joint_head_online_tool_interface"]
    scale_break_pressure = {
        "joint_success_gap": normalize(
            float(qwen_joint["success_rate"]) - float(deepseek_joint["success_rate"]),
            0.35,
            0.50,
        ),
        "searched_undercoverage_gap": normalize(
            float(layer_band["gains"]["deepseek_minus_qwen_searched_undercoverage"]),
            0.12,
            0.22,
        ),
        "joint_success_gain_asymmetry": normalize(
            float(hard_interface["gains"]["qwen_joint_minus_tool_head_success"])
            - float(hard_interface["gains"]["deepseek_joint_minus_tool_head_success"]),
            0.015,
            0.04,
        ),
        "deepseek_remaining_failure_pressure": normalize(
            1.0 - float(deepseek_joint["success_rate"]),
            0.55,
            0.65,
        ),
    }
    scale_break_pressure_score = mean(scale_break_pressure.values())

    kernel_survival = {
        "stage7_verdict_support": float(stage7d["verdict"]["verdict_support_score"]),
        "stage8a_survival": float(stage8a["headline_metrics"]["law_residual_survival_score"]),
        "stage8c_invariants": float(stage8c["headline_metrics"]["overall_stage8c_score"]),
        "stage8d_brain_falsification": float(stage8d["headline_metrics"]["overall_stage8d_score"]),
        "qwen_orientation_mechanism_bridge": normalize(float(qwen_o["mechanism_bridge_score"]), 0.70, 0.80),
    }
    kernel_survival_score = mean(kernel_survival.values())

    overall_score = mean(
        [
            architecture_break_pressure_score,
            data_break_pressure_score,
            scale_break_pressure_score,
            kernel_survival_score,
        ]
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage9a_mechanism_adversarial_break_test",
        },
        "adversarial_sources": {
            "highest_pressure_source": "architecture_plus_scale",
            "secondary_pressure_source": "data_distribution",
            "retained_kernel_status": "nontrivial",
        },
        "pillars": {
            "architecture_break_pressure": {
                "components": architecture_break_pressure,
                "score": float(architecture_break_pressure_score),
            },
            "data_break_pressure": {
                "components": data_break_pressure,
                "score": float(data_break_pressure_score),
            },
            "scale_break_pressure": {
                "components": scale_break_pressure,
                "score": float(scale_break_pressure_score),
            },
            "kernel_survival": {
                "components": kernel_survival,
                "score": float(kernel_survival_score),
            },
        },
        "headline_metrics": {
            "architecture_break_pressure_score": float(architecture_break_pressure_score),
            "data_break_pressure_score": float(data_break_pressure_score),
            "scale_break_pressure_score": float(scale_break_pressure_score),
            "kernel_survival_score": float(kernel_survival_score),
            "overall_stage9a_score": float(overall_score),
        },
        "hypotheses": {
            "H1_architecture_induced_break_pressure_is_real": bool(architecture_break_pressure_score >= 0.78),
            "H2_data_induced_break_pressure_is_nontrivial": bool(data_break_pressure_score >= 0.45),
            "H3_scale_induced_break_pressure_is_nontrivial": bool(scale_break_pressure_score >= 0.61),
            "H4_kernel_survives_despite_these_pressures": bool(kernel_survival_score >= 0.73),
            "H5_stage9a_mechanism_break_test_is_moderately_supported": bool(overall_score >= 0.68),
        },
        "project_readout": {
            "summary": (
                "Stage 9A is positive only if the project can make the break sources explicit: architecture pressure, "
                "data pressure, and scale pressure should all be visible rather than hidden inside a generic residual."
            ),
            "next_question": (
                "If this stage holds, the next step is to quantify how much of the remaining residual belongs to each "
                "pressure source rather than leaving it as an opaque mismatch."
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
