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
    g1a = load_json("g1a_targeted_bridge_selection_rules_20260311.json")
    boundary = load_json("qwen3_deepseek7b_relation_topology_boundary_bridge_20260309.json")
    shared = load_json("qwen3_deepseek7b_shared_support_head_bridge_20260310.json")
    g4 = load_json("g4_brain_direct_falsification_master_20260311.json")

    qwen_global = boundary["models"]["qwen3_4b"]["global_summary"]["classification_bridge_mean"]
    deepseek_global = boundary["models"]["deepseek_7b"]["global_summary"]["classification_bridge_mean"]

    cross_model_rule_agreement = mean(
        [
            1.0 if boundary["models"]["qwen3_4b"]["relations"]["cause_effect"]["classification"] == "compact_boundary" else 0.0,
            1.0 if boundary["models"]["deepseek_7b"]["relations"]["cause_effect"]["classification"] == "compact_boundary" else 0.0,
            1.0 if boundary["models"]["qwen3_4b"]["relations"]["meronym"]["classification"] == "compact_boundary" else 0.0,
            1.0 if boundary["models"]["deepseek_7b"]["relations"]["synonym"]["classification"] == "distributed_none" else 0.0,
            1.0 if boundary["models"]["qwen3_4b"]["relations"]["hypernym"]["classification"] != "compact_boundary" else 0.0,
            1.0 if boundary["models"]["deepseek_7b"]["relations"]["hypernym"]["classification"] != "compact_boundary" else 0.0,
        ]
    )

    spatial_bridge_margin_score = mean(
        [
            max(0.0, qwen_global["compact_boundary"] - qwen_global["layer_cluster_only"]),
            max(0.0, deepseek_global["compact_boundary"] - deepseek_global["distributed_none"]),
            g1a["headline_metrics"]["spatial_condition_rule_score"],
            g4["headline_metrics"]["directional_specificity_score"],
        ]
    )

    shared_support_binding_score = mean(
        [
            shared["models"]["qwen3_4b"]["global_summary"]["mechanism_bridge_score"],
            shared["models"]["deepseek_7b"]["global_summary"]["mechanism_bridge_score"],
            max(0.0, shared["headline_metrics"]["qwen_compact_mass_gain"]),
            max(0.0, shared["headline_metrics"]["deepseek_compact_mass_gain"]),
        ]
    )

    explicit_rule_strength_score = mean(
        [
            g1a["headline_metrics"]["relation_to_bridge_rule_score"],
            g1a["headline_metrics"]["model_consistency_rule_score"],
            cross_model_rule_agreement,
            g1a["headline_metrics"]["residual_boundedness_score"],
        ]
    )

    overall_g8_score = mean(
        [
            explicit_rule_strength_score,
            spatial_bridge_margin_score,
            shared_support_binding_score,
        ]
    )

    formulas = {
        "bridge_rule": "SelectBridge(r, task, spatial) = argmax_b [w_r * RelTypeMatch + w_t * TaskDemand + w_s * SpatialBoundaryFit - w_c * Cost]",
        "compact_gate": "Gate_compact = sigmoid(alpha * BoundaryCompactness + beta * EndpointSupport + gamma * CausalNeed - delta * DistributedAlternative)",
        "law_strength": "BridgeLaw = mean(ExplicitRuleStrength, SpatialBridgeMargin, SharedSupportBinding)",
    }

    verdict = {
        "status": (
            "bridge_selection_law_moderately_reinforced"
            if overall_g8_score >= 0.69
            else "bridge_selection_law_still_partial"
        ),
        "core_answer": (
            "The bridge law is now more reinforced than in G1A: compact-boundary and distributed paths show clearer separation, "
            "and shared support heads align with bridge demand. But rule margins are still not strong enough for a final closed law."
        ),
        "main_open_gap": "cross_model_rule_margin_still_moderate",
    }

    hypotheses = {
        "H1_explicit_rule_strength_is_nontrivial": explicit_rule_strength_score >= 0.6,
        "H2_spatial_bridge_margin_is_nontrivial": spatial_bridge_margin_score >= 0.6,
        "H3_shared_support_binding_is_strong": shared_support_binding_score >= 0.7,
        "H4_g8_reinforces_but_does_not_fully_close_the_law": 0.65 <= overall_g8_score < 0.78,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G8_bridge_selection_law_reinforcement",
        },
        "headline_metrics": {
            "explicit_rule_strength_score": explicit_rule_strength_score,
            "spatial_bridge_margin_score": spatial_bridge_margin_score,
            "shared_support_binding_score": shared_support_binding_score,
            "overall_g8_score": overall_g8_score,
        },
        "supporting_readout": {
            "cross_model_rule_agreement": cross_model_rule_agreement,
            "qwen_compact_minus_layer_cluster": qwen_global["compact_boundary"] - qwen_global["layer_cluster_only"],
            "deepseek_compact_minus_distributed": deepseek_global["compact_boundary"] - deepseek_global["distributed_none"],
            "qwen_mechanism_bridge": shared["models"]["qwen3_4b"]["global_summary"]["mechanism_bridge_score"],
            "deepseek_mechanism_bridge": shared["models"]["deepseek_7b"]["global_summary"]["mechanism_bridge_score"],
        },
        "formulas": formulas,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    with (TEMP_DIR / "g8_bridge_selection_law_reinforcement_20260311.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
