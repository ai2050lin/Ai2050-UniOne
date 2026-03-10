#!/usr/bin/env python
"""
Focused search for task block D:
scan tunable dual-store routing to see whether we can beat the direct baseline
on both novel concepts and retention.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

import test_continuous_input_grounding_proto as proto


class TunableDualStoreGrounder:
    def __init__(self, margin: float, stable_tactile_weight: float, plastic_visual_weight: float) -> None:
        self.margin = float(margin)
        self.stable_tactile_weight = float(stable_tactile_weight)
        self.plastic_visual_weight = float(plastic_visual_weight)
        self.family_proto: Dict[str, np.ndarray] = {}
        self.family_count: Dict[str, int] = {}
        self.stable_concepts: Dict[str, Dict[str, np.ndarray]] = {}
        self.plastic_concepts: Dict[str, Dict[str, np.ndarray]] = {}
        self.stable_count: Dict[str, Dict[str, int]] = {}
        self.plastic_count: Dict[str, Dict[str, int]] = {}
        self.phase1_concepts: set[str] = set()

    @staticmethod
    def _ema(prev: np.ndarray | None, x: np.ndarray, count: int, alpha_cap: float) -> np.ndarray:
        if prev is None:
            return x.astype(np.float32).copy()
        alpha = min(alpha_cap, 1.0 / float(count + 1))
        return ((1.0 - alpha) * prev + alpha * x).astype(np.float32)

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        prev_family = self.family_proto.get(family)
        family_count = self.family_count.get(family, 0)
        self.family_proto[family] = self._ema(prev_family, x, family_count, 0.28)
        self.family_count[family] = family_count + 1

        if concept in self.phase1_concepts:
            store = self.stable_concepts.setdefault(family, {})
            counts = self.stable_count.setdefault(family, {})
            alpha_cap = 0.18
        elif family in self.stable_concepts and concept not in self.stable_concepts[family]:
            store = self.plastic_concepts.setdefault(family, {})
            counts = self.plastic_count.setdefault(family, {})
            alpha_cap = 0.42
        else:
            store = self.stable_concepts.setdefault(family, {})
            counts = self.stable_count.setdefault(family, {})
            alpha_cap = 0.22
        prev = store.get(concept)
        count = counts.get(concept, 0)
        store[concept] = self._ema(prev, x, count, alpha_cap)
        counts[concept] = count + 1

    def consolidate_phase1(self) -> None:
        self.phase1_concepts = {concept for family in proto.PHASE1 for concept in proto.PHASE1[family]}

    def predict(self, x: np.ndarray):
        family = min(self.family_proto, key=lambda name: proto.sq_dist(x, self.family_proto[name]))
        visual = x[:8]
        tactile = x[8:]
        stable = self.stable_concepts.get(family, {})
        plastic = self.plastic_concepts.get(family, {})

        stable_best = None
        stable_score = float("inf")
        for concept, center in stable.items():
            score = (1.0 - self.stable_tactile_weight) * proto.sq_dist(x, center) + self.stable_tactile_weight * proto.sq_dist(tactile, center[8:])
            if score < stable_score:
                stable_score = score
                stable_best = concept

        plastic_best = None
        plastic_score = float("inf")
        for concept, center in plastic.items():
            score = (1.0 - self.plastic_visual_weight) * proto.sq_dist(x, center) + self.plastic_visual_weight * proto.sq_dist(visual, center[:8])
            if score < plastic_score:
                plastic_score = score
                plastic_best = concept

        if plastic_best is not None and plastic_score + self.margin < stable_score:
            return family, plastic_best
        if stable_best is not None:
            return family, stable_best
        assert plastic_best is not None
        return family, plastic_best


def evaluate_model(model, groups, repeats, rng, noise, dropout_p):
    family_ok = 0
    concept_ok = 0
    total = 0
    for _ in range(repeats):
        for family, concepts in groups.items():
            for concept in concepts:
                x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                pred_family, pred_concept = model.predict(x)
                family_ok += int(pred_family == family)
                concept_ok += int(pred_concept == concept)
                total += 1
    return {
        "family_accuracy": float(family_ok / max(1, total)),
        "concept_accuracy": float(concept_ok / max(1, total)),
    }


def run_candidate(margin: float, stable_tactile_weight: float, plastic_visual_weight: float, seed: int, noise: float, dropout_p: float):
    rng = np.random.default_rng(seed)
    direct = proto.DirectPrototypeLearner(dim=16)
    dual = TunableDualStoreGrounder(margin, stable_tactile_weight, plastic_visual_weight)

    for _ in range(42):
        for family, concepts in proto.PHASE1.items():
            for concept in concepts:
                x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                direct.train(x, family, concept)
                dual.train(x, family, concept)
    dual.consolidate_phase1()

    for _ in range(2):
        for family, concepts in proto.PHASE2.items():
            for concept in concepts:
                x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                direct.train(x, family, concept)
                dual.train(x, family, concept)

    novel_direct = evaluate_model(direct, proto.PHASE2, 24, rng, noise, dropout_p)
    novel_dual = evaluate_model(dual, proto.PHASE2, 24, rng, noise, dropout_p)

    for _ in range(12):
        for family, concepts in proto.PHASE2.items():
            for concept in concepts:
                x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                direct.train(x, family, concept)
                dual.train(x, family, concept)

    retention_direct = evaluate_model(direct, proto.PHASE1, 22, rng, noise, dropout_p)
    retention_dual = evaluate_model(dual, proto.PHASE1, 22, rng, noise, dropout_p)
    overall_direct = evaluate_model(direct, {family: proto.PHASE1[family] + proto.PHASE2[family] for family in proto.FAMILIES}, 22, rng, noise, dropout_p)
    overall_dual = evaluate_model(dual, {family: proto.PHASE1[family] + proto.PHASE2[family] for family in proto.FAMILIES}, 22, rng, noise, dropout_p)

    return {
        "novel_direct": novel_direct["concept_accuracy"],
        "novel_dual": novel_dual["concept_accuracy"],
        "retention_direct": retention_direct["concept_accuracy"],
        "retention_dual": retention_dual["concept_accuracy"],
        "overall_direct": overall_direct["concept_accuracy"],
        "overall_dual": overall_dual["concept_accuracy"],
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Dual-store routing scan for continuous grounding")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/continuous_input_grounding_dual_store_scan_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    candidates = []
    for margin in [-0.04, -0.02, 0.0, 0.02]:
        for stable_tactile_weight in [0.50, 0.65, 0.80]:
            for plastic_visual_weight in [0.10, 0.25, 0.40]:
                rows = []
                for offset in range(int(args.num_seeds)):
                    rows.append(
                        run_candidate(
                            margin,
                            stable_tactile_weight,
                            plastic_visual_weight,
                            int(args.seed) + offset,
                            float(args.noise),
                            float(args.dropout_p),
                        )
                    )
                summary = summarize(rows)
                candidates.append(
                    {
                        "margin": float(margin),
                        "stable_tactile_weight": float(stable_tactile_weight),
                        "plastic_visual_weight": float(plastic_visual_weight),
                        **summary,
                        "novel_gain": float(summary["novel_dual"] - summary["novel_direct"]),
                        "retention_gain": float(summary["retention_dual"] - summary["retention_direct"]),
                        "overall_gain": float(summary["overall_dual"] - summary["overall_direct"]),
                    }
                )

    feasible = [
        row
        for row in candidates
        if row["novel_gain"] > 0.0 and row["retention_gain"] > 0.0
    ]
    best_overall = max(candidates, key=lambda row: row["overall_gain"] + 0.6 * row["retention_gain"] + 0.6 * row["novel_gain"])
    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "runtime_sec": float(time.time() - t0),
        },
        "best_overall": best_overall,
        "feasible_count": len(feasible),
        "best_feasible": max(feasible, key=lambda row: row["overall_gain"]) if feasible else None,
        "top_candidates": sorted(
            candidates,
            key=lambda row: row["overall_gain"] + 0.6 * row["retention_gain"] + 0.6 * row["novel_gain"],
            reverse=True,
        )[:12],
        "hypotheses": {
            "H1_exists_dual_store_that_beats_direct_on_novel_and_retention": bool(len(feasible) > 0),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"feasible_count": results["feasible_count"], "best_overall": best_overall}, ensure_ascii=False, indent=2))
    print(json.dumps(results["hypotheses"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
