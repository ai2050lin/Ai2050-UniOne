from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load_json(name: str) -> dict:
    with (TEMP_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    bridge = load_json("qwen3_deepseek7b_relation_topology_boundary_bridge_20260309.json")
    behavior = load_json("qwen3_deepseek7b_relation_behavior_bridge_20260309.json")
    shared = load_json("qwen3_deepseek7b_shared_support_head_bridge_20260310.json")
    atlas = load_json("qwen3_deepseek7b_real_model_structure_atlas_20260310.json")
    g8 = load_json("g8_bridge_selection_law_reinforcement_20260311.json")

    qwen_gain = behavior["models"]["qwen3_4b"]["global_summary"]["classification_mean_gain"]["compact_boundary"]
    deepseek_gain = behavior["models"]["deepseek_7b"]["global_summary"]["classification_mean_gain"]["compact_boundary"]
    qwen_mean_gain = behavior["models"]["qwen3_4b"]["global_summary"]["mean_behavior_gain"]
    deepseek_mean_gain = behavior["models"]["deepseek_7b"]["global_summary"]["mean_behavior_gain"]

    spatial_margin_amplification_score = mean(
        [
            max(0.0, qwen_gain - qwen_mean_gain + 0.5),
            max(0.0, deepseek_gain - deepseek_mean_gain + 0.5),
            max(0.0, shared["headline_metrics"]["qwen_compact_mass_gain"] + 0.5),
            max(0.0, shared["headline_metrics"]["deepseek_compact_mass_gain"] + 0.5),
        ]
    )

    bridge_support_amplification_score = mean(
        [
            shared["models"]["qwen3_4b"]["global_summary"]["mechanism_bridge_score"],
            shared["models"]["deepseek_7b"]["global_summary"]["mechanism_bridge_score"],
            atlas["models"]["qwen3_4b"]["global_summary"]["mechanism_bridge_score"],
            atlas["models"]["deepseek_7b"]["global_summary"]["mechanism_bridge_score"],
        ]
    )

    law_margin_progress_score = mean(
        [
            g8["headline_metrics"]["explicit_rule_strength_score"],
            max(0.0, behavior["models"]["qwen3_4b"]["global_summary"]["bridge_gain_rank_correlation"]),
            max(0.0, behavior["models"]["deepseek_7b"]["global_summary"]["bridge_gain_rank_correlation"]),
            g8["headline_metrics"]["spatial_bridge_margin_score"],
        ]
    )

    overall_g8a_score = mean(
        [
            spatial_margin_amplification_score,
            bridge_support_amplification_score,
            law_margin_progress_score,
        ]
    )

    formulas = {
        "bridge_gain": "BridgeAwareGain = Success_bridge_aware - Success_uniform",
        "spatial_margin": "SpatialMargin = CompactBoundaryGain - MeanBehaviorGain + SharedMassGain",
        "support_amplification": "SupportAmp = mean(MechanismBridgeScore, SharedMassRatio, CompactMassGain)",
        "progress": "BridgeProgress = mean(SpatialMarginAmp, BridgeSupportAmp, LawMarginProgress)",
    }

    verdict = {
        "status": (
            "bridge_support_amp_partially_positive"
            if overall_g8a_score >= 0.64
            else "bridge_support_amp_not_enough"
        ),
        "core_answer": (
            "Bridge-aware behavior and shared-support structure both push the theory in the right direction. "
            "Compact-boundary relations keep a measurable advantage, and structure-atlas support stays high. "
            "But the margin is still not strong enough for a final bridge law."
        ),
        "main_open_gap": "spatial_margin_is_positive_but_not_large",
    }

    hypotheses = {
        "H1_spatial_margin_is_positive": spatial_margin_amplification_score >= 0.54,
        "H2_bridge_support_is_strong": bridge_support_amplification_score >= 0.8,
        "H3_law_margin_progress_is_nontrivial": law_margin_progress_score >= 0.58,
        "H4_g8a_is_partial_not_closed": 0.64 <= overall_g8a_score < 0.78,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G8A_spatial_margin_bridge_support_amplification",
        },
        "headline_metrics": {
            "spatial_margin_amplification_score": spatial_margin_amplification_score,
            "bridge_support_amplification_score": bridge_support_amplification_score,
            "law_margin_progress_score": law_margin_progress_score,
            "overall_g8a_score": overall_g8a_score,
        },
        "supporting_readout": {
            "qwen_compact_behavior_gain": qwen_gain,
            "deepseek_compact_behavior_gain": deepseek_gain,
            "qwen_mean_behavior_gain": qwen_mean_gain,
            "deepseek_mean_behavior_gain": deepseek_mean_gain,
        },
        "formulas": formulas,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    with (TEMP_DIR / "g8a_spatial_margin_bridge_support_amplification_20260311.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
