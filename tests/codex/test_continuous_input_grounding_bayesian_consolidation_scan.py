#!/usr/bin/env python
"""
Task block D:
annealed Bayesian consolidation on top of direct stable prototypes.

This is a deliberately conservative method:
- phase-1 concepts keep direct raw prototypes unchanged
- phase-2 concepts use a posterior prototype with a family/sibling prior
- the prior strength decays with the number of observed updates

It asks a sharper question than the residual-gate scan:
if we preserve old concepts almost exactly and only improve the estimate of
new concepts, can we flip overall gain to positive?
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

import test_continuous_input_grounding_proto as proto


def eval_concept_accuracy(
    model,
    groups: Dict[str, List[str]],
    repeats: int,
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
) -> float:
    concept_ok = 0
    total = 0
    for _ in range(repeats):
        for family, concepts in groups.items():
            for concept in concepts:
                x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                _, pred_concept = model.predict(x)
                concept_ok += int(pred_concept == concept)
                total += 1
    return float(concept_ok / max(1, total))


class AnnealedBayesianNovelPrototype:
    def __init__(self, kappa0: float, decay: float) -> None:
        self.kappa0 = float(kappa0)
        self.decay = float(decay)
        self.phase1_concepts: set[str] = set()
        self.family_proto: Dict[str, np.ndarray] = {}
        self.family_count: Dict[str, int] = {}
        self.phase1_proto: Dict[str, np.ndarray] = {}
        self.phase1_count: Dict[str, int] = {}
        self.phase2_mean: Dict[str, np.ndarray] = {}
        self.phase2_count: Dict[str, int] = {}
        self.phase2_proto: Dict[str, np.ndarray] = {}

    @staticmethod
    def _ema(prev: np.ndarray | None, x: np.ndarray, count: int, alpha_cap: float) -> np.ndarray:
        if prev is None:
            return x.astype(np.float32).copy()
        alpha = min(alpha_cap, 1.0 / float(count + 1))
        return ((1.0 - alpha) * prev + alpha * x).astype(np.float32)

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        family_count = self.family_count.get(family, 0)
        self.family_proto[family] = self._ema(self.family_proto.get(family), x, family_count, 0.24)
        self.family_count[family] = family_count + 1

        if concept in self.phase1_concepts or not self.phase1_concepts:
            concept_count = self.phase1_count.get(concept, 0)
            self.phase1_proto[concept] = self._ema(self.phase1_proto.get(concept), x, concept_count, 0.24)
            self.phase1_count[concept] = concept_count + 1
            return

        concept_count = self.phase2_count.get(concept, 0)
        sample_mean = self._ema(self.phase2_mean.get(concept), x, concept_count, 0.55)
        siblings = [self.phase1_proto[name] for name in self.phase1_concepts if proto.concept_family(name) == family]
        prior = np.mean(np.stack(siblings, axis=0), axis=0).astype(np.float32) if siblings else self.family_proto[family]
        kappa = self.kappa0 / float((concept_count + 1) ** self.decay)
        posterior = (
            ((concept_count + 1) / float(concept_count + 1 + kappa)) * sample_mean
            + (kappa / float(concept_count + 1 + kappa)) * prior
        ).astype(np.float32)
        self.phase2_mean[concept] = sample_mean
        self.phase2_proto[concept] = posterior
        self.phase2_count[concept] = concept_count + 1

    def consolidate_phase1(self) -> None:
        self.phase1_concepts = {concept for family in proto.PHASE1 for concept in proto.PHASE1[family]}

    def predict(self, x: np.ndarray):
        best_concept = None
        best_score = float("inf")
        for concept, center in self.phase1_proto.items():
            score = proto.sq_dist(x, center)
            if score < best_score:
                best_score = score
                best_concept = concept
        for concept, center in self.phase2_proto.items():
            score = proto.sq_dist(x, center)
            if score < best_score:
                best_score = score
                best_concept = concept
        assert best_concept is not None
        return proto.concept_family(best_concept), best_concept


def run_candidate(kappa0: float, decay: float, seed: int, noise: float, dropout_p: float) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    direct = proto.DirectPrototypeLearner(dim=16)
    model = AnnealedBayesianNovelPrototype(kappa0=kappa0, decay=decay)

    for _ in range(42):
        for family, concepts in proto.PHASE1.items():
            for concept in concepts:
                x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                direct.train(x, family, concept)
                model.train(x, family, concept)

    model.consolidate_phase1()

    for _ in range(3):
        for family, concepts in proto.PHASE2.items():
            for concept in concepts:
                x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                direct.train(x, family, concept)
                model.train(x, family, concept)

    novel_direct = eval_concept_accuracy(direct, proto.PHASE2, 24, rng, noise, dropout_p)
    novel_model = eval_concept_accuracy(model, proto.PHASE2, 24, rng, noise, dropout_p)

    for _ in range(18):
        for family, concepts in proto.PHASE2.items():
            for concept in concepts:
                x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                direct.train(x, family, concept)
                model.train(x, family, concept)

    retention_direct = eval_concept_accuracy(direct, proto.PHASE1, 22, rng, noise, dropout_p)
    retention_model = eval_concept_accuracy(model, proto.PHASE1, 22, rng, noise, dropout_p)
    all_groups = {family: proto.PHASE1[family] + proto.PHASE2[family] for family in proto.FAMILIES}
    overall_direct = eval_concept_accuracy(direct, all_groups, 22, rng, noise, dropout_p)
    overall_model = eval_concept_accuracy(model, all_groups, 22, rng, noise, dropout_p)

    return {
        "novel_direct": novel_direct,
        "novel_model": novel_model,
        "retention_direct": retention_direct,
        "retention_model": retention_model,
        "overall_direct": overall_direct,
        "overall_model": overall_model,
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Annealed Bayesian consolidation scan for continuous grounding")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/continuous_input_grounding_bayesian_consolidation_scan_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    candidates = []
    for kappa0 in [6.0, 8.0, 10.0, 12.0, 16.0, 20.0]:
        for decay in [0.25, 0.50, 0.75, 1.00, 1.25]:
            rows = []
            for offset in range(int(args.num_seeds)):
                rows.append(
                    run_candidate(
                        kappa0=kappa0,
                        decay=decay,
                        seed=int(args.seed) + offset,
                        noise=float(args.noise),
                        dropout_p=float(args.dropout_p),
                    )
                )
            summary = summarize(rows)
            novel_gain = float(summary["novel_model"] - summary["novel_direct"])
            retention_gain = float(summary["retention_model"] - summary["retention_direct"])
            overall_gain = float(summary["overall_model"] - summary["overall_direct"])
            candidates.append(
                {
                    "kappa0": float(kappa0),
                    "decay": float(decay),
                    **summary,
                    "novel_gain": novel_gain,
                    "retention_gain": retention_gain,
                    "overall_gain": overall_gain,
                    "dual_score": float(novel_gain + retention_gain),
                }
            )

    dual_positive = [row for row in candidates if row["novel_gain"] > 0.0 and row["retention_gain"] >= 0.0]
    full_positive = [
        row
        for row in candidates
        if row["novel_gain"] > 0.0 and row["retention_gain"] >= 0.0 and row["overall_gain"] > 0.0
    ]

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "runtime_sec": float(time.time() - t0),
        },
        "best_dual_positive": max(dual_positive, key=lambda row: row["dual_score"]) if dual_positive else None,
        "best_full_positive": max(full_positive, key=lambda row: row["overall_gain"]) if full_positive else None,
        "dual_positive_count": len(dual_positive),
        "full_positive_count": len(full_positive),
        "top_dual_positive": sorted(dual_positive, key=lambda row: row["dual_score"], reverse=True)[:12],
        "top_overall": sorted(candidates, key=lambda row: row["overall_gain"], reverse=True)[:12],
        "hypotheses": {
            "H1_dual_positive_region_exists": bool(len(dual_positive) > 0),
            "H2_full_positive_region_exists": bool(len(full_positive) > 0),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "dual_positive_count": results["dual_positive_count"],
                "full_positive_count": results["full_positive_count"],
                "best_dual_positive": results["best_dual_positive"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(json.dumps(results["hypotheses"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
