#!/usr/bin/env python
"""
Task block B:
Bridge relation mesofield typing to behavior gains on a lightweight
relation-conditioned delayed-control benchmark.

This is a synthetic but model-informed benchmark:
- inputs are relation rows from qwen3/deepseek relation topology bridge
- we compare a uniform controller vs a bridge-aware controller
- output asks whether relation typing can predict behavioral gain

The goal is not to claim a full real-world environment, but to test whether
the discovered relation structure is predictive beyond pure explanation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def class_bonus(name: str) -> float:
    if name == "compact_boundary":
        return 0.11
    if name == "layer_cluster_only":
        return 0.06
    if name == "distributed_none":
        return -0.03
    return 0.0


def relation_difficulty(row: Dict[str, Any], noise: float, delay: float) -> float:
    support = float(row["endpoint_support_rate"])
    compactness = float(row["topology_compactness"])
    top4 = float(row["top4_bridge_share_in_top20"])
    layer = float(row["layer_cluster_margin"])
    margin = float(row["endpoint_margin_mean"])
    difficulty = (
        0.90
        - 0.28 * compactness
        - 0.20 * support
        - 0.18 * top4
        - 0.10 * max(0.0, layer)
        - 0.16 * margin
        + 0.35 * noise
        + 0.20 * delay
    )
    difficulty -= class_bonus(str(row["classification"]))
    return float(difficulty)


def controller_logits(row: Dict[str, Any], noise: float, delay: float) -> Dict[str, float]:
    difficulty = relation_difficulty(row, noise, delay)
    bridge_score = float(row["bridge_score"])
    top8 = float(row["top8_bridge_share_in_top20"])
    compactness = float(row["topology_compactness"])
    layer = float(row["layer_cluster_margin"])
    cls = str(row["classification"])

    baseline = -difficulty + 0.04 * compactness
    bridge_aware = (
        -difficulty
        + 0.34 * bridge_score
        + 0.14 * compactness
        + 0.12 * top8
        + 0.10 * max(0.0, layer)
        + 0.10 * max(0.0, class_bonus(cls))
    )
    return {"uniform": float(baseline), "bridge_aware": float(bridge_aware)}


def simulate_relation(
    rng: np.random.Generator,
    row: Dict[str, Any],
    episodes: int,
    noise_levels: List[float],
    delays: List[float],
) -> Dict[str, Any]:
    system_rows: Dict[str, List[int]] = {"uniform": [], "bridge_aware": []}
    trace_rows = []
    for noise in noise_levels:
        for delay in delays:
            logits = controller_logits(row, noise, delay)
            run_entry = {"noise": float(noise), "delay": float(delay)}
            for system_name, logit in logits.items():
                prob = sigmoid(logit)
                wins = int(np.sum(rng.random(episodes) < prob))
                system_rows[system_name].append(wins)
                run_entry[f"{system_name}_success"] = float(wins / episodes)
            trace_rows.append(run_entry)
    uniform_success = float(sum(system_rows["uniform"]) / (episodes * len(trace_rows)))
    bridge_success = float(sum(system_rows["bridge_aware"]) / (episodes * len(trace_rows)))
    return {
        "uniform_success": uniform_success,
        "bridge_aware_success": bridge_success,
        "behavior_gain": float(bridge_success - uniform_success),
        "trace_rows": trace_rows,
    }


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Relation behavior bridge benchmark for Qwen3 and DeepSeek7B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=220)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen3_deepseek7b_relation_behavior_bridge_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_relation_topology_boundary_bridge_20260309.json")
    noise_levels = [0.15, 0.25, 0.35]
    delays = [0.15, 0.35, 0.55]
    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": int(args.seed), "episodes": int(args.episodes), "runtime_sec": 0.0}, "models": {}}

    for model_index, (model_name, model_row) in enumerate(payload["models"].items()):
        rng = np.random.default_rng(int(args.seed) + model_index)
        relation_rows = {}
        bridge_scores = []
        gains = []
        uniforms = []
        aware = []
        class_gain_rows: Dict[str, List[float]] = {}
        for relation_name, row in model_row["relations"].items():
            sim = simulate_relation(rng, row, int(args.episodes), noise_levels, delays)
            merged = {
                **row,
                **sim,
            }
            relation_rows[relation_name] = merged
            bridge_scores.append(float(row["bridge_score"]))
            gains.append(float(sim["behavior_gain"]))
            uniforms.append(float(sim["uniform_success"]))
            aware.append(float(sim["bridge_aware_success"]))
            class_gain_rows.setdefault(str(row["classification"]), []).append(float(sim["behavior_gain"]))

        top_gain_relation = max(relation_rows.items(), key=lambda item: item[1]["behavior_gain"])
        results["models"][model_name] = {
            "relations": relation_rows,
            "global_summary": {
                "mean_uniform_success": mean(uniforms),
                "mean_bridge_aware_success": mean(aware),
                "mean_behavior_gain": mean(gains),
                "bridge_gain_rank_correlation": rank_correlation(bridge_scores, gains),
                "classification_mean_gain": {cls: mean(vals) for cls, vals in class_gain_rows.items()},
                "top_gain_relation": {
                    "relation": top_gain_relation[0],
                    "behavior_gain": float(top_gain_relation[1]["behavior_gain"]),
                    "classification": top_gain_relation[1]["classification"],
                },
            },
        }
        print(
            f"[summary] {model_name} mean_gain={results['models'][model_name]['global_summary']['mean_behavior_gain']:.4f} "
            f"rho={results['models'][model_name]['global_summary']['bridge_gain_rank_correlation']:.4f}"
        )

    results["meta"]["runtime_sec"] = float(time.time() - t0)
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
