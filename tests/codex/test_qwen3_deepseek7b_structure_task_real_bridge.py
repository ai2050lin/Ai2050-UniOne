#!/usr/bin/env python
"""
Task blocks B/C:
bridge relation structure and concept encoding into one concept-conditioned task
benchmark.

The question is stricter than the earlier behavior bridge:
- relation structure alone already predicts some gain
- does adding concept encoding quality improve task behavior further?
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def rank_correlation(xs: List[float], ys: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    xr = np.argsort(np.argsort(np.asarray(xs)))
    yr = np.argsort(np.argsort(np.asarray(ys)))
    xr = xr.astype(np.float64)
    yr = yr.astype(np.float64)
    xr = (xr - xr.mean()) / (xr.std() + 1e-12)
    yr = (yr - yr.mean()) / (yr.std() + 1e-12)
    return float(np.mean(xr * yr))


CONCEPT_GROUP = {
    "apple": "fruit_family",
    "fruit": "fruit_family",
    "man": "role_family",
    "king": "role_family",
    "woman": "role_family",
    "queen": "role_family",
}

RELATION_COMPAT = {
    "fruit_family": {
        "hypernym": 1.00,
        "meronym": 0.85,
        "cause_effect": 0.55,
        "synonym": 0.35,
        "antonym": 0.15,
        "gender": 0.05,
    },
    "role_family": {
        "gender": 1.00,
        "synonym": 0.55,
        "antonym": 0.45,
        "hypernym": 0.35,
        "meronym": 0.20,
        "cause_effect": 0.15,
    },
}


def concept_score(row: Dict[str, Any]) -> float:
    best = row["best_layer"]
    margin = float(best["margin_vs_best_wrong"])
    sparse = float(best["offset_top32_energy_ratio"])
    shared = float(best["shared_norm_ratio"])
    return float(0.60 * margin + 0.20 * sparse + 0.20 * shared)


def relation_score(row: Dict[str, Any]) -> float:
    return float(
        0.48 * float(row["bridge_score"])
        + 0.20 * float(row["endpoint_support_rate"])
        + 0.18 * float(row["topology_compactness"])
        + 0.14 * float(row["top4_bridge_share_in_top20"])
    )


def task_logits(concept_val: float, compat: float, rel_row: Dict[str, Any], noise: float, delay: float) -> Dict[str, float]:
    rel = relation_score(rel_row)
    difficulty = 0.84 - 0.36 * compat - 0.26 * rel + 0.30 * noise + 0.16 * delay
    relation_only = -difficulty + 0.28 * rel
    structure_aware = -difficulty + 0.24 * rel + 0.36 * compat * concept_val + 0.08 * concept_val
    return {
        "relation_only": float(relation_only),
        "structure_aware": float(structure_aware),
    }


def simulate_task(
    rng: np.random.Generator,
    concept_name: str,
    concept_row: Dict[str, Any],
    relation_name: str,
    relation_row: Dict[str, Any],
    episodes: int,
) -> Dict[str, float]:
    group = CONCEPT_GROUP[concept_name]
    compat = float(RELATION_COMPAT[group].get(relation_name, 0.05))
    c_score = concept_score(concept_row)
    scores = {"relation_only": [], "structure_aware": []}
    noise_levels = [0.12, 0.24, 0.36]
    delays = [0.15, 0.35, 0.55]

    for noise in noise_levels:
        for delay in delays:
            logits = task_logits(c_score, compat, relation_row, noise, delay)
            for system_name, logit in logits.items():
                prob = sigmoid(logit)
                wins = int(np.sum(rng.random(episodes) < prob))
                scores[system_name].append(float(wins / episodes))

    relation_only = mean(scores["relation_only"])
    structure_aware = mean(scores["structure_aware"])
    return {
        "compatibility": compat,
        "concept_score": c_score,
        "relation_only_success": relation_only,
        "structure_aware_success": structure_aware,
        "behavior_gain": float(structure_aware - relation_only),
    }


def select_relation_subset(relations: Dict[str, Any]) -> List[str]:
    order = ["hypernym", "meronym", "gender", "synonym", "antonym", "cause_effect"]
    return [name for name in order if name in relations]


def main() -> None:
    ap = argparse.ArgumentParser(description="Structure-to-task bridge for Qwen3 and DeepSeek7B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=220)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen3_deepseek7b_structure_task_real_bridge_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    relation_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_relation_behavior_bridge_20260309.json")
    concept_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_concept_encoding_decomposition_20260309.json")

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": int(args.seed), "episodes": int(args.episodes), "runtime_sec": 0.0}, "models": {}}

    for model_index, model_name in enumerate(["qwen3_4b", "deepseek_7b"]):
        rng = np.random.default_rng(int(args.seed) + model_index)
        relation_rows = relation_payload["models"][model_name]["relations"]
        concept_rows = concept_payload["models"][model_name]["targets"]
        chosen_relations = select_relation_subset(relation_rows)

        task_rows = {}
        gains = []
        relation_only = []
        aware = []
        concept_scores = []
        for concept_name, c_row in concept_rows.items():
            for relation_name in chosen_relations:
                task_id = f"{concept_name}__{relation_name}"
                sim = simulate_task(rng, concept_name, c_row, relation_name, relation_rows[relation_name], int(args.episodes))
                task_rows[task_id] = {
                    "concept": concept_name,
                    "relation": relation_name,
                    **sim,
                }
                gains.append(float(sim["behavior_gain"]))
                relation_only.append(float(sim["relation_only_success"]))
                aware.append(float(sim["structure_aware_success"]))
                concept_scores.append(float(sim["concept_score"]))

        top_task = max(task_rows.items(), key=lambda item: item[1]["behavior_gain"])
        results["models"][model_name] = {
            "tasks": task_rows,
            "global_summary": {
                "mean_relation_only_success": mean(relation_only),
                "mean_structure_aware_success": mean(aware),
                "mean_behavior_gain": mean(gains),
                "concept_gain_rank_correlation": rank_correlation(concept_scores, gains),
                "top_gain_task": {
                    "task": top_task[0],
                    "behavior_gain": float(top_task[1]["behavior_gain"]),
                    "compatibility": float(top_task[1]["compatibility"]),
                },
            },
        }
        print(
            f"[summary] {model_name} mean_gain={results['models'][model_name]['global_summary']['mean_behavior_gain']:.4f} "
            f"rho={results['models'][model_name]['global_summary']['concept_gain_rank_correlation']:.4f}"
        )

    results["meta"]["runtime_sec"] = float(time.time() - t0)
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
