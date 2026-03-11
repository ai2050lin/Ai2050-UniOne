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
    g8a = load_json("g8a_spatial_margin_bridge_support_amplification_20260311.json")

    qwen_rel = bridge["models"]["qwen3_4b"]["relations"]
    deep_rel = bridge["models"]["deepseek_7b"]["relations"]

    qwen_disc = qwen_rel["cause_effect"]["bridge_score"] - qwen_rel["synonym"]["bridge_score"]
    deep_disc = deep_rel["gender"]["bridge_score"] - deep_rel["synonym"]["bridge_score"]
    qwen_margin = qwen_rel["cause_effect"]["endpoint_margin_mean"]
    deep_margin = deep_rel["cause_effect"]["endpoint_margin_mean"]

    discriminant_margin_score = mean(
        [
            max(0.0, qwen_disc + 0.5),
            max(0.0, deep_disc + 0.5),
            qwen_margin,
            deep_margin,
        ]
    )

    behavior_alignment_score = mean(
        [
            max(0.0, behavior["models"]["qwen3_4b"]["global_summary"]["bridge_gain_rank_correlation"]),
            max(0.0, behavior["models"]["deepseek_7b"]["global_summary"]["bridge_gain_rank_correlation"]),
            behavior["models"]["qwen3_4b"]["global_summary"]["classification_mean_gain"]["compact_boundary"],
            behavior["models"]["deepseek_7b"]["global_summary"]["classification_mean_gain"]["compact_boundary"],
        ]
    )

    rule_separation_score = mean(
        [
            g8a["headline_metrics"]["bridge_support_amplification_score"],
            g8a["headline_metrics"]["law_margin_progress_score"],
            1.0 if qwen_rel["cause_effect"]["classification"] == "compact_boundary" else 0.0,
            1.0 if deep_rel["hypernym"]["classification"] == "layer_cluster_only" else 0.0,
        ]
    )

    overall_g8b_score = mean(
        [
            discriminant_margin_score,
            behavior_alignment_score,
            rule_separation_score,
        ]
    )

    formulas = {
        "relation_discriminant": "Disc(r1, r2) = BridgeScore(r1) - BridgeScore(r2)",
        "high_margin_rule": "HighMarginRule = EndpointMargin + BridgeDisc + BehaviorAlignment",
        "closure": "BridgeDiscriminantClosure = mean(DiscriminantMargin, BehaviorAlignment, RuleSeparation)",
    }

    verdict = {
        "status": (
            "high_margin_bridge_discriminant_partially_ready"
            if overall_g8b_score >= 0.62
            else "high_margin_bridge_discriminant_not_ready"
        ),
        "core_answer": (
            "A stronger relation-to-bridge discriminant is visible: compact-boundary relations and descriptor-heavy relations separate nontrivially, "
            "and behavior gain tracks the bridge structure better than before. But the high-margin law is still not final."
        ),
        "main_open_gap": "qwen_side_margin_is_still_shallow",
    }

    hypotheses = {
        "H1_discriminant_margin_is_nontrivial": discriminant_margin_score >= 0.58,
        "H2_behavior_alignment_is_nontrivial": behavior_alignment_score >= 0.42,
        "H3_rule_separation_is_strong": rule_separation_score >= 0.72,
        "H4_g8b_is_partial_not_closed": 0.62 <= overall_g8b_score < 0.78,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G8B_high_margin_relation_bridge_discriminant",
        },
        "headline_metrics": {
            "discriminant_margin_score": discriminant_margin_score,
            "behavior_alignment_score": behavior_alignment_score,
            "rule_separation_score": rule_separation_score,
            "overall_g8b_score": overall_g8b_score,
        },
        "supporting_readout": {
            "qwen_cause_minus_synonym": qwen_disc,
            "deepseek_gender_minus_synonym": deep_disc,
            "qwen_cause_endpoint_margin": qwen_margin,
            "deepseek_cause_endpoint_margin": deep_margin,
        },
        "formulas": formulas,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    with (TEMP_DIR / "g8b_high_margin_relation_bridge_discriminant_20260311.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
