#!/usr/bin/env python
"""
Aggregate existing causal artifacts into a single homology scorecard.

The intent is to answer a narrower stage question:
do the current positive signals support one mechanism family,
or are they still just disconnected local wins?
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def true_ratio(flags: Dict[str, Any]) -> float:
    if not flags:
        return 0.0
    return float(sum(1.0 for value in flags.values() if bool(value)) / len(flags))


def normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return clamp01((float(value) - lo) / (hi - lo))


def score_shared_atom(payload: Dict[str, Any]) -> Dict[str, Any]:
    unified = payload["systems"]["unified_shared_atoms"]
    drops = unified["ablations"]["shared_top"]["drops"]
    gains = payload["gains"]
    usage = unified["usage"]
    hypotheses = payload["hypotheses"]

    components = {
        "shared_vs_random_joint_drop": normalize(float(gains["shared_vs_random_joint_drop"]), 0.03, 0.30),
        "shared_min_concept_relation_drop": normalize(
            min(float(drops["concept_drop"]), float(drops["relation_drop"])),
            0.05,
            0.30,
        ),
        "shared_recovery_drop": normalize(float(drops["recovery_drop"]), 0.05, 0.35),
        "cross_dim_corr": normalize(float(usage["cross_dim_corr"]), 0.85, 0.99),
        "hypothesis_ratio": true_ratio(hypotheses),
    }
    return {
        "score": mean(components.values()),
        "components": components,
        "headline": payload["headline_metrics"],
        "hypotheses": hypotheses,
    }


def score_phase_switch(payload: Dict[str, Any]) -> Dict[str, Any]:
    replay = payload["systems"]["heterogeneous_local_replay"]
    hypotheses = payload["hypotheses"]
    metrics = payload["headline_metrics"]

    components = {
        "baseline_accuracy": normalize(float(replay["baseline_accuracy"]), 0.50, 0.72),
        "comparison_phase_switch": normalize(
            float(metrics["comparison_phase_memory_comparator_advantage"]),
            0.01,
            0.08,
        ),
        "recovery_replay_dependence": normalize(
            float(metrics["recovery_phase_replay_memory_comparator_advantage"]),
            0.02,
            0.18,
        ),
        "distinct_top_regions": normalize(float(metrics["distinct_top_region_count"]), 1.0, 3.0),
        "hypothesis_ratio": true_ratio(
            {
                "H2": hypotheses.get("H2_comparison_phase_prefers_memory_comparator"),
                "H3": hypotheses.get("H3_recovery_phase_is_replay_dependent"),
                "H4": hypotheses.get("H4_no_single_global_core"),
            }
        ),
    }
    return {
        "score": mean(components.values()),
        "components": components,
        "headline": metrics,
        "hypotheses": hypotheses,
    }


def score_shared_support(payload: Dict[str, Any]) -> Dict[str, Any]:
    models = {}
    model_scores = []
    for model_name, model_row in payload["models"].items():
        summary = model_row["global_summary"]
        components = {
            "soft_layer_overlap": normalize(float(summary["concept_relation_soft_layer_overlap_ratio"]), 0.25, 0.60),
            "shared_mass_ratio": normalize(float(summary["mean_shared_mass_ratio"]), 0.02, 0.06),
            "compact_minus_diffuse_shared_mass": normalize(float(summary["compact_minus_diffuse_shared_mass"]), 0.01, 0.04),
            "mechanism_bridge_score": normalize(float(summary["mechanism_bridge_score"]), 0.70, 0.95),
        }
        score = mean(components.values())
        models[model_name] = {
            "score": score,
            "components": components,
            "global_summary": summary,
        }
        model_scores.append(score)
    return {
        "score": mean(model_scores),
        "models": models,
    }


def score_protocol_mesofield(payload: Dict[str, Any]) -> Dict[str, Any]:
    models = {}
    model_scores = []
    for model_name, model_row in payload["models"].items():
        summary = model_row["global_summary"]
        relation_count = max(1, len(model_row["relations"]))
        positive_k_ratio = mean(
            1.0 if float(value) > 0.0 else 0.0
            for value in summary["mean_causal_margin_by_k"].values()
        )
        collapse_delta = float(summary["mean_layer_cluster_collapse_ratio"]) - float(
            summary["mean_layer_cluster_control_collapse_ratio"]
        )
        stronger_ratio = float(summary["layer_cluster_stronger_than_control_count"]) / relation_count
        components = {
            "positive_k_ratio": positive_k_ratio,
            "collapse_delta": normalize(collapse_delta, 0.0, 0.06),
            "stronger_than_control_ratio": stronger_ratio,
            "layer_cluster_margin": normalize(float(summary["mean_layer_cluster_margin"]), -0.04, 0.10),
        }
        score = mean(components.values())
        models[model_name] = {
            "score": score,
            "components": components,
            "global_summary": summary,
        }
        model_scores.append(score)
    return {
        "score": mean(model_scores),
        "models": models,
    }


def score_online_recovery(payload: Dict[str, Any]) -> Dict[str, Any]:
    models = {}
    model_scores = []
    for model_name, model_row in payload["models"].items():
        systems = model_row["systems"]
        no_recovery = float(systems["online_no_recovery"]["success_rate"])
        recovery = systems["online_recovery_aware"]
        gain = float(recovery["success_rate"]) - no_recovery
        components = {
            "online_recovery_gain": normalize(gain, 0.05, 0.22),
            "rollback_recovery_rate": normalize(float(recovery["rollback_recovery_rate"]), 0.20, 0.65),
            "completion_progress": normalize(float(recovery["mean_completion_progress"]), 0.45, 0.80),
            "post_recovery_stability": normalize(float(recovery["mean_post_recovery_stability"]), 0.45, 0.78),
        }
        score = mean(components.values())
        models[model_name] = {
            "score": score,
            "components": components,
            "gain": gain,
            "systems": systems,
        }
        model_scores.append(score)
    return {
        "score": mean(model_scores),
        "models": models,
        "headline": payload["headline_metrics"],
        "hypotheses": payload["hypotheses"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate a unified causal homology scorecard from existing artifacts")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/unified_mechanism_causal_homology_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    shared_atom = load_json(ROOT / "tests" / "codex_temp" / "shared_atom_causal_unification_benchmark_20260310.json")
    phase_switch = load_json(ROOT / "tests" / "codex_temp" / "local_pulse_phase_conditioned_causal_atlas_20260310.json")
    shared_support = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_shared_support_head_bridge_20260310.json")
    protocol_mesofield = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_relation_protocol_mesofield_scale_20260309.json")
    online_recovery = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_online_recovery_chain_20260310.json")

    pillars = {
        "shared_atom_causal": score_shared_atom(shared_atom),
        "phase_conditioned_switching": score_phase_switch(phase_switch),
        "real_model_shared_support": score_shared_support(shared_support),
        "relation_protocol_mesofield": score_protocol_mesofield(protocol_mesofield),
        "online_recovery_bridge": score_online_recovery(online_recovery),
    }

    same_source_score = mean(
        [
            float(pillars["shared_atom_causal"]["score"]),
            float(pillars["real_model_shared_support"]["score"]),
            float(pillars["relation_protocol_mesofield"]["score"]),
        ]
    )
    stage_gating_score = mean(
        [
            float(pillars["phase_conditioned_switching"]["score"]),
            float(pillars["online_recovery_bridge"]["score"]),
        ]
    )
    overall_score = float(
        0.28 * float(pillars["shared_atom_causal"]["score"])
        + 0.18 * float(pillars["phase_conditioned_switching"]["score"])
        + 0.20 * float(pillars["real_model_shared_support"]["score"])
        + 0.16 * float(pillars["relation_protocol_mesofield"]["score"])
        + 0.18 * float(pillars["online_recovery_bridge"]["score"])
    )

    hypotheses = {
        "H1_shared_atoms_form_a_joint_causal_source": bool(pillars["shared_atom_causal"]["score"] >= 0.70),
        "H2_phase_specific_local_cores_exist_without_global_controller": bool(
            pillars["phase_conditioned_switching"]["score"] >= 0.62
        ),
        "H3_real_models_show_shared_support_between_concept_and_relation": bool(
            pillars["real_model_shared_support"]["score"] >= 0.45
        ),
        "H4_relation_protocol_requires_meso_scale_more_than_tiny_head_groups": bool(
            pillars["relation_protocol_mesofield"]["score"] >= 0.35
        ),
        "H5_recovery_structure_transfers_to_online_chain": bool(
            pillars["online_recovery_bridge"]["score"] >= 0.55
        ),
        "H6_unified_causal_homology_is_moderately_supported": bool(overall_score >= 0.55),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "unified_mechanism_causal_homology_scorecard",
            "sources": {
                "shared_atom_causal": "tests/codex_temp/shared_atom_causal_unification_benchmark_20260310.json",
                "phase_conditioned_switching": "tests/codex_temp/local_pulse_phase_conditioned_causal_atlas_20260310.json",
                "real_model_shared_support": "tests/codex_temp/qwen3_deepseek7b_shared_support_head_bridge_20260310.json",
                "relation_protocol_mesofield": "tests/codex_temp/qwen3_deepseek7b_relation_protocol_mesofield_scale_20260309.json",
                "online_recovery_bridge": "tests/codex_temp/qwen3_deepseek7b_online_recovery_chain_20260310.json",
            },
        },
        "pillars": pillars,
        "headline_metrics": {
            "same_source_support_score": float(same_source_score),
            "stage_gating_support_score": float(stage_gating_score),
            "overall_causal_homology_score": float(overall_score),
            "shared_atom_score": float(pillars["shared_atom_causal"]["score"]),
            "phase_switch_score": float(pillars["phase_conditioned_switching"]["score"]),
            "real_model_shared_support_score": float(pillars["real_model_shared_support"]["score"]),
            "relation_mesofield_score": float(pillars["relation_protocol_mesofield"]["score"]),
            "online_recovery_bridge_score": float(pillars["online_recovery_bridge"]["score"]),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "Current evidence supports a moderate same-family picture: shared atoms, phase-conditioned local cores, "
                "real-model shared support, meso-scale relation protocol structure, and online recovery gains are not yet "
                "a closed proof, but they do line up better with one mechanism family than with five unrelated modules."
            ),
            "next_question": (
                "The next stage should stop aggregating correlations and instead run a single joint intervention that "
                "perturbs shared support, protocol routing, and recovery-critical structure in one setting and measures "
                "whether representation, topology, and recovery fail together."
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
