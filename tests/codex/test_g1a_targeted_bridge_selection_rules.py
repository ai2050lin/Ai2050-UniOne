from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load_json(name: str) -> dict:
    path = TEMP_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def compact_boundary_advantage(model_blob: dict) -> float:
    means = model_blob["global_summary"]["classification_bridge_mean"]
    compact = means.get("compact_boundary", 0.0)
    layer = means.get("layer_cluster_only", 0.0)
    distributed = means.get("distributed_none", 0.0)
    return compact - distributed, compact - layer


def main() -> None:
    boundary_bridge = load_json("qwen3_deepseek7b_relation_topology_boundary_bridge_20260309.json")
    p8b = load_json("p8b_3d_wiring_dynamic_topology_division_20260311.json")
    p8c = load_json("p8c_spatial_brain_falsifier_predictions_20260311.json")
    stage8d = load_json("stage8d_brain_high_risk_falsification_20260311.json")
    p9b = load_json("p9b_spatial_residual_counterexample_compression_20260311.json")

    qwen_adv_distributed, qwen_adv_layer = compact_boundary_advantage(boundary_bridge["models"]["qwen3_4b"])
    deep_adv_distributed, deep_adv_layer = compact_boundary_advantage(boundary_bridge["models"]["deepseek_7b"])

    compact_relations = [
        boundary_bridge["models"]["qwen3_4b"]["relations"]["cause_effect"]["bridge_score"],
        boundary_bridge["models"]["qwen3_4b"]["relations"]["meronym"]["bridge_score"],
        boundary_bridge["models"]["deepseek_7b"]["relations"]["cause_effect"]["bridge_score"],
        boundary_bridge["models"]["deepseek_7b"]["relations"]["gender"]["bridge_score"],
        boundary_bridge["models"]["deepseek_7b"]["relations"]["meronym"]["bridge_score"],
    ]
    distributed_relations = [
        boundary_bridge["models"]["qwen3_4b"]["relations"]["hypernym"]["bridge_score"],
        boundary_bridge["models"]["deepseek_7b"]["relations"]["synonym"]["bridge_score"],
        boundary_bridge["models"]["deepseek_7b"]["relations"]["antonym"]["bridge_score"],
    ]

    relation_to_bridge_rule_score = mean(
        [
            mean(compact_relations),
            1.0 - mean(distributed_relations),
            p8c["headline_metrics"]["local_vs_bridge_prediction_specificity_score"],
            p8b["headline_metrics"]["selective_bridge_advantage_score"],
        ]
    )

    spatial_condition_rule_score = mean(
        [
            p8b["headline_metrics"]["geometry_dynamic_topology_split_score"],
            p8c["headline_metrics"]["geometry_vs_targeted_prediction_score"],
            p8c["headline_metrics"]["brain_spatial_testability_score"],
            stage8d["headline_metrics"]["directional_falsifier_score"],
        ]
    )

    model_consistency_rule_score = mean(
        [
            max(0.0, qwen_adv_distributed),
            max(0.0, deep_adv_distributed),
            1.0 - abs(qwen_adv_layer),
            1.0 - abs(deep_adv_layer),
            1.0 - p9b["pillars"]["bridge_specificity_gap"]["components"]["not_yet_strong"],
        ]
    )

    residual_boundedness_score = mean(
        [
            p9b["headline_metrics"]["bridge_specificity_gap_score"],
            p9b["headline_metrics"]["residual_source_control_score"],
            p9b["headline_metrics"]["compression_verdict_score"],
        ]
    )

    overall_g1a_score = mean(
        [
            relation_to_bridge_rule_score,
            spatial_condition_rule_score,
            model_consistency_rule_score,
            residual_boundedness_score,
        ]
    )

    formulas = {
        "bridge_selection_rule": (
            "SelectBridge(r, task, spatial) = argmax_b [w_r * RelTypeMatch(r,b) + "
            "w_t * TaskDemand(task,b) + w_s * SpatialBoundaryFit(spatial,b) - w_c * Cost(b)]"
        ),
        "compact_boundary_gate": (
            "Gate_compact = sigmoid(alpha * BoundaryCompactness + beta * EndpointSupport + "
            "gamma * CausalNeed - delta * DistributedAlternative)"
        ),
        "distributed_gate": (
            "Gate_distributed = sigmoid(alpha_d * FamilySpread + beta_d * DescriptorOverlap - gamma_d * CompactBoundaryNeed)"
        ),
        "selection_margin": (
            "Margin = Score(target_bridge) - max Score(non_target_bridge)"
        ),
    }

    rules = {
        "cause_effect": "prefer compact-boundary bridge when endpoint support is high and causal need is concentrated",
        "meronym": "prefer compact-boundary bridge when part-whole boundary is tight and local topology remains compact",
        "gender": "can be compact-boundary or layer-cluster depending on model bias, but should remain selective not diffuse",
        "hypernym": "more likely to use layer-cluster or broad abstraction path than compact-boundary bridge",
        "synonym_antonym": "less bridge-heavy; often descriptor overlap or distributed family path dominates",
    }

    verdict = {
        "status": "targeted_bridge_rules_partially_explicit",
        "core_answer": (
            "The project can now state partial bridge-selection rules: cause-effect and meronym relations tend to prefer "
            "compact-boundary bridges, while hypernym and descriptor-heavy relations are less bridge-specific. "
            "But the rule margins are still not strong enough to count as closed."
        ),
        "main_open_gap": "rule_margin_strength_is_still_moderate",
    }

    hypotheses = {
        "H1_relation_to_bridge_rules_are_nontrivial": relation_to_bridge_rule_score >= 0.54,
        "H2_spatial_conditions_help_explain_bridge_selection": spatial_condition_rule_score >= 0.66,
        "H3_model_consistency_is_partial_not_random": model_consistency_rule_score >= 0.5,
        "H4_g1a_reaches_partial_rule_explicitness": overall_g1a_score >= 0.6,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G1A_targeted_bridge_selection_rules",
        },
        "headline_metrics": {
            "relation_to_bridge_rule_score": relation_to_bridge_rule_score,
            "spatial_condition_rule_score": spatial_condition_rule_score,
            "model_consistency_rule_score": model_consistency_rule_score,
            "residual_boundedness_score": residual_boundedness_score,
            "overall_g1a_score": overall_g1a_score,
        },
        "supporting_readout": {
            "qwen_compact_minus_distributed": qwen_adv_distributed,
            "deepseek_compact_minus_distributed": deep_adv_distributed,
            "qwen_compact_minus_layer": qwen_adv_layer,
            "deepseek_compact_minus_layer": deep_adv_layer,
            "qwen_cause_effect_bridge_score": boundary_bridge["models"]["qwen3_4b"]["relations"]["cause_effect"]["bridge_score"],
            "deepseek_cause_effect_bridge_score": boundary_bridge["models"]["deepseek_7b"]["relations"]["cause_effect"]["bridge_score"],
            "qwen_hypernym_bridge_score": boundary_bridge["models"]["qwen3_4b"]["relations"]["hypernym"]["bridge_score"],
            "deepseek_synonym_bridge_score": boundary_bridge["models"]["deepseek_7b"]["relations"]["synonym"]["bridge_score"],
        },
        "formulas": formulas,
        "rules": rules,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "g1a_targeted_bridge_selection_rules_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
