#!/usr/bin/env python
"""
Search a stage-5 fused objective over task blocks 1-4.

This is a proxy objective, not a full end-to-end training loop. The goal is to
find a weighting that does not let structure, online metrics, or brain-side
constraints collapse each other.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def normalize_weights(a: float, b: float, c: float) -> Tuple[float, float, float]:
    s = float(a + b + c)
    return float(a / s), float(b / s), float(c / s)


def group_score(values: Dict[str, float], weights: Dict[str, float]) -> float:
    return float(sum(float(values[k]) * float(weights[k]) for k in weights))


def main() -> None:
    ap = argparse.ArgumentParser(description="Search a fused objective over task blocks 1-4")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage5_fused_unified_law_objective_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    task1 = load_json(ROOT / "tests" / "codex_temp" / "unified_mechanism_causal_homology_20260311.json")
    task2 = load_json(ROOT / "tests" / "codex_temp" / "task_block_2_unified_training_closure_20260311.json")
    task3 = load_json(ROOT / "tests" / "codex_temp" / "task_block_3_real_model_task_bridge_closure_20260311.json")
    task4 = load_json(ROOT / "tests" / "codex_temp" / "task_block_4_brain_constraint_closure_20260311.json")

    anchor = float(task1["headline_metrics"]["overall_causal_homology_score"])
    training_values = {
        "structure": float(task2["headline_metrics"]["structure_training_score"]),
        "recovery": float(task2["headline_metrics"]["recovery_training_score"]),
        "generator": float(task2["headline_metrics"]["generator_training_score"]),
    }
    online_values = {
        "qwen": float(task3["headline_metrics"]["qwen_task_bridge_score"]),
        "deepseek": float(task3["headline_metrics"]["deepseek_task_bridge_score"]),
        "upgrade": float(task3["headline_metrics"]["cross_model_upgrade_score"]),
    }
    brain_values = {
        "positive": float(task4["headline_metrics"]["brain_positive_score"]),
        "robustness": float(task4["headline_metrics"]["brain_robustness_score"]),
    }

    candidates: List[Dict[str, Any]] = []
    block_weight_grid = [0.8, 1.0, 1.2, 1.4]
    split_grid = [0.2, 0.3, 0.4, 0.5]
    balance_penalties = [0.00, 0.05, 0.10, 0.15, 0.20]
    guard_penalties = [0.20, 0.30, 0.40]

    for t_w in block_weight_grid:
        for o_w in block_weight_grid:
            for b_w in block_weight_grid:
                wt, wo, wb = normalize_weights(t_w, o_w, b_w)
                for structure_share in split_grid:
                    for recovery_share in split_grid:
                        if structure_share + recovery_share >= 0.95:
                            continue
                        generator_share = 1.0 - structure_share - recovery_share
                        training_weights = {
                            "structure": float(structure_share),
                            "recovery": float(recovery_share),
                            "generator": float(generator_share),
                        }
                        for qwen_share in split_grid:
                            for deepseek_share in split_grid:
                                if qwen_share + deepseek_share >= 0.95:
                                    continue
                                upgrade_share = 1.0 - qwen_share - deepseek_share
                                online_weights = {
                                    "qwen": float(qwen_share),
                                    "deepseek": float(deepseek_share),
                                    "upgrade": float(upgrade_share),
                                }
                                for brain_positive_share in [0.35, 0.5, 0.65]:
                                    brain_weights = {
                                        "positive": float(brain_positive_share),
                                        "robustness": float(1.0 - brain_positive_share),
                                    }
                                    for balance_penalty in balance_penalties:
                                        for guard_penalty in guard_penalties:
                                            training_score = group_score(training_values, training_weights)
                                            online_score = group_score(online_values, online_weights)
                                            brain_score = group_score(brain_values, brain_weights)
                                            group_values = [training_score, online_score, brain_score]
                                            imbalance = float(statistics.pstdev(group_values))

                                            guard_loss = 0.0
                                            guard_loss += max(0.0, 0.58 - training_values["structure"]) * guard_penalty
                                            guard_loss += max(0.0, 0.42 - online_values["qwen"]) * guard_penalty
                                            guard_loss += max(0.0, 0.50 - online_values["deepseek"]) * guard_penalty
                                            guard_loss += max(0.0, 0.64 - brain_values["positive"]) * guard_penalty

                                            fused_score = float(
                                                0.10 * anchor
                                                + wt * training_score
                                                + wo * online_score
                                                + wb * brain_score
                                                - balance_penalty * imbalance
                                                - guard_loss
                                            )
                                            candidates.append(
                                                {
                                                    "fused_score": fused_score,
                                                    "anchor_score": anchor,
                                                    "training_score": training_score,
                                                    "online_score": online_score,
                                                    "brain_score": brain_score,
                                                    "imbalance": imbalance,
                                                    "guard_loss": guard_loss,
                                                    "block_weights": {"training": wt, "online": wo, "brain": wb},
                                                    "training_weights": training_weights,
                                                    "online_weights": online_weights,
                                                    "brain_weights": brain_weights,
                                                    "balance_penalty": float(balance_penalty),
                                                    "guard_penalty": float(guard_penalty),
                                                }
                                            )

    ranked = sorted(candidates, key=lambda row: row["fused_score"], reverse=True)
    best = ranked[0]

    hypotheses = {
        "H1_best_fused_objective_keeps_training_above_0_62": bool(best["training_score"] >= 0.62),
        "H2_best_fused_objective_keeps_online_above_0_54": bool(best["online_score"] >= 0.54),
        "H3_best_fused_objective_keeps_brain_above_0_68": bool(best["brain_score"] >= 0.68),
        "H4_best_fused_objective_keeps_imbalance_below_0_07": bool(best["imbalance"] <= 0.07),
        "H5_stage5_proxy_objective_is_viable": bool(best["fused_score"] >= 0.60),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage5_fused_unified_law_objective_proxy",
            "candidate_count": len(ranked),
        },
        "inputs": {
            "task_block_1_anchor": anchor,
            "training_values": training_values,
            "online_values": online_values,
            "brain_values": brain_values,
        },
        "best_config": best,
        "top_configs": ranked[:10],
        "headline_metrics": {
            "best_fused_score": float(best["fused_score"]),
            "best_training_score": float(best["training_score"]),
            "best_online_score": float(best["online_score"]),
            "best_brain_score": float(best["brain_score"]),
            "best_imbalance": float(best["imbalance"]),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "This stage-5 proxy does not yet train a single model, but it does identify a fused objective regime "
                "where training-law, online bridge, and brain-side terms can stay jointly positive without one term "
                "obviously flattening the others."
            ),
            "next_question": (
                "The next step is to turn this proxy weighting into a real optimization loop so that structure, online, "
                "and brain-side penalties are updated together rather than selected from separate dashboards."
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
