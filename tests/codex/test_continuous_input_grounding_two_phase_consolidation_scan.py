#!/usr/bin/env python
"""
Task block D:
explicit two-phase consolidation scan.

The design separates:
- online plastic write for new concepts
- offline stable rewrite for old concepts

This tests whether D requires an explicit phase split, rather than a single
controller family that tries to solve writing and protection at once.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

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


class TwoPhaseConsolidationGrounder:
    def __init__(
        self,
        quantile: float,
        threshold_bonus: float,
        phase2_blend: float,
        stable_refresh: float,
        route_margin: float,
        replay_steps: int,
    ) -> None:
        self.quantile = float(quantile)
        self.threshold_bonus = float(threshold_bonus)
        self.phase2_blend = float(phase2_blend)
        self.stable_refresh = float(stable_refresh)
        self.route_margin = float(route_margin)
        self.replay_steps = int(replay_steps)

        self.family_basis: Dict[str, np.ndarray] = {}
        self.family_count: Dict[str, int] = {}
        self.phase1_concepts: set[str] = set()
        self.phase1_proto: Dict[str, np.ndarray] = {}
        self.phase1_count: Dict[str, int] = {}
        self.stable_offsets: Dict[str, np.ndarray] = {}
        self.stable_var: Dict[str, np.ndarray] = {}
        self.stable_count: Dict[str, int] = {}
        self.phase1_scores: Dict[str, List[float]] = {}
        self.family_threshold: Dict[str, float] = {}
        self.phase2_mean: Dict[str, np.ndarray] = {}
        self.phase2_count: Dict[str, int] = {}
        self.phase2_post: Dict[str, np.ndarray] = {}
        self.phase2_novelty: Dict[str, float] = {}
        self.family_novelty: Dict[str, List[float]] = {family: [] for family in proto.FAMILIES}
        self.candidate_proto: Dict[str, np.ndarray] = {}
        self.threshold_shift: Dict[str, float] = {}

    @staticmethod
    def _ema(prev: np.ndarray | None, x: np.ndarray, count: int, alpha_cap: float) -> np.ndarray:
        if prev is None:
            return x.astype(np.float32).copy()
        alpha = min(alpha_cap, 1.0 / float(count + 1))
        return ((1.0 - alpha) * prev + alpha * x).astype(np.float32)

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        family_count = self.family_count.get(family, 0)
        if concept in self.phase1_concepts or not self.phase1_concepts:
            self.family_basis[family] = self._ema(self.family_basis.get(family), x, family_count, 0.18)
            self.family_count[family] = family_count + 1

        base = self.family_basis.get(family, x)
        centered = (x - base).astype(np.float32)

        if concept in self.phase1_concepts or not self.phase1_concepts:
            concept_count = self.phase1_count.get(concept, 0)
            self.phase1_proto[concept] = self._ema(self.phase1_proto.get(concept), x, concept_count, 0.24)
            self.phase1_count[concept] = concept_count + 1

            stable_count = self.stable_count.get(concept, 0)
            offset = self._ema(self.stable_offsets.get(concept), centered, stable_count, 0.22)
            residual = (centered - offset).astype(np.float32)
            var = self._ema(self.stable_var.get(concept), np.square(residual).astype(np.float32), stable_count, 0.22)
            self.stable_offsets[concept] = offset
            self.stable_var[concept] = var
            self.stable_count[concept] = stable_count + 1
            score = float(np.sum(np.square(residual) / (var + 0.03)))
            self.phase1_scores.setdefault(family, []).append(score)
            return

        concept_count = self.phase2_count.get(concept, 0)
        sample_mean = self._ema(self.phase2_mean.get(concept), x, concept_count, 0.55)
        siblings = [self.phase1_proto[name] for name in self.phase1_concepts if proto.concept_family(name) == family]
        prior = np.mean(np.stack(siblings, axis=0), axis=0).astype(np.float32) if siblings else base
        kappa = 10.0 / float((concept_count + 1) ** 0.5)
        posterior = (
            ((concept_count + 1) / float(concept_count + 1 + kappa)) * sample_mean
            + (kappa / float(concept_count + 1 + kappa)) * prior
        ).astype(np.float32)
        self.phase2_mean[concept] = sample_mean
        self.phase2_post[concept] = posterior
        self.phase2_count[concept] = concept_count + 1

        sibling_distances = [proto.sq_dist(posterior, self.phase1_proto[name]) for name in self.phase1_concepts if proto.concept_family(name) == family]
        novelty = float(min(sibling_distances)) if sibling_distances else float(proto.sq_dist(posterior, prior))
        self.phase2_novelty[concept] = novelty
        self.family_novelty[family].append(novelty)

    def consolidate_phase1(self) -> None:
        self.phase1_concepts = {concept for family in proto.PHASE1 for concept in proto.PHASE1[family]}
        for family, scores in self.phase1_scores.items():
            arr = np.array(scores, dtype=np.float32)
            self.family_threshold[family] = float(np.quantile(arr, self.quantile) + self.threshold_bonus * np.std(arr))
        self._rebuild_candidate_proto()

    def _rebuild_candidate_proto(self) -> None:
        candidate_proto = {concept: center.copy() for concept, center in self.phase1_proto.items()}
        threshold_shift: Dict[str, float] = {family: 0.0 for family in proto.FAMILIES}

        for family in proto.FAMILIES:
            new_concepts = [concept for concept in self.phase2_post if proto.concept_family(concept) == family]
            if not new_concepts:
                continue

            family_phase2 = []
            for concept in new_concepts:
                raw = self.phase2_mean[concept]
                post = self.phase2_post[concept]
                blended = ((1.0 - self.phase2_blend) * raw + self.phase2_blend * post).astype(np.float32)
                candidate_proto[concept] = blended
                family_phase2.append(blended)

            if not family_phase2:
                continue

            family_phase2_mean = np.mean(np.stack(family_phase2, axis=0), axis=0)
            family_shift = float(proto.sq_dist(family_phase2_mean, self.family_basis[family]))
            threshold_shift[family] = self.stable_refresh * min(0.25, family_shift)

            for _ in range(self.replay_steps):
                for concept in self.phase1_concepts:
                    if proto.concept_family(concept) != family:
                        continue
                    stable_target = (self.family_basis[family] + self.stable_offsets[concept]).astype(np.float32)
                    prev = candidate_proto[concept]
                    candidate_proto[concept] = ((1.0 - self.stable_refresh) * prev + self.stable_refresh * stable_target).astype(np.float32)

        self.candidate_proto = candidate_proto
        self.threshold_shift = threshold_shift

    def _stable_best(self, x: np.ndarray, family: str) -> Tuple[str | None, float]:
        base = self.family_basis[family]
        centered = (x - base).astype(np.float32)
        best_concept = None
        best_score = float("inf")
        for concept in self.phase1_concepts:
            if proto.concept_family(concept) != family:
                continue
            raw_score = proto.sq_dist(x, self.phase1_proto[concept])
            resid_score = float(np.sum(np.square(centered - self.stable_offsets[concept]) / (self.stable_var[concept] + 0.03)))
            score = 0.68 * raw_score + 0.32 * resid_score
            if score < best_score:
                best_score = score
                best_concept = concept
        return best_concept, best_score

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        family = min(self.family_basis, key=lambda name: proto.sq_dist(x, self.family_basis[name]))
        stable_best, stable_score = self._stable_best(x, family)
        threshold = self.family_threshold.get(family, stable_score) + self.threshold_shift.get(family, 0.0)

        novel_best = None
        novel_score = float("inf")
        for concept, center in self.candidate_proto.items():
            if concept in self.phase1_concepts or proto.concept_family(concept) != family:
                continue
            score = proto.sq_dist(x, center)
            if score < novel_score:
                novel_score = score
                novel_best = concept

        if novel_best is not None and (stable_score > threshold or novel_score + self.route_margin < stable_score):
            return family, novel_best
        assert stable_best is not None
        return family, stable_best


def run_candidate(
    quantile: float,
    threshold_bonus: float,
    phase2_blend: float,
    stable_refresh: float,
    route_margin: float,
    replay_steps: int,
    seed: int,
    noise: float,
    dropout_p: float,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    direct = proto.DirectPrototypeLearner(dim=16)
    model = TwoPhaseConsolidationGrounder(
        quantile=quantile,
        threshold_bonus=threshold_bonus,
        phase2_blend=phase2_blend,
        stable_refresh=stable_refresh,
        route_margin=route_margin,
        replay_steps=replay_steps,
    )

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
    model._rebuild_candidate_proto()

    novel_direct = eval_concept_accuracy(direct, proto.PHASE2, 24, rng, noise, dropout_p)
    novel_model = eval_concept_accuracy(model, proto.PHASE2, 24, rng, noise, dropout_p)

    for _ in range(18):
        for family, concepts in proto.PHASE2.items():
            for concept in concepts:
                x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                direct.train(x, family, concept)
                model.train(x, family, concept)
    model._rebuild_candidate_proto()

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
    ap = argparse.ArgumentParser(description="Two-phase consolidation scan for continuous grounding")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/continuous_input_grounding_two_phase_consolidation_scan_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    candidates = []
    for quantile in [0.75, 0.82, 0.88]:
        for threshold_bonus in [0.10, 0.18, 0.26]:
            for phase2_blend in [0.35, 0.50, 0.65]:
                for stable_refresh in [0.08, 0.14]:
                    for route_margin in [-0.04, -0.02, 0.00]:
                        for replay_steps in [1, 2]:
                            rows = []
                            for offset in range(int(args.num_seeds)):
                                rows.append(
                                    run_candidate(
                                        quantile=quantile,
                                        threshold_bonus=threshold_bonus,
                                        phase2_blend=phase2_blend,
                                        stable_refresh=stable_refresh,
                                        route_margin=route_margin,
                                        replay_steps=replay_steps,
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
                                    "quantile": float(quantile),
                                    "threshold_bonus": float(threshold_bonus),
                                    "phase2_blend": float(phase2_blend),
                                    "stable_refresh": float(stable_refresh),
                                    "route_margin": float(route_margin),
                                    "replay_steps": int(replay_steps),
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
        "best_overall": max(candidates, key=lambda row: row["overall_gain"]),
        "dual_positive_count": len(dual_positive),
        "full_positive_count": len(full_positive),
        "top_dual_positive": sorted(dual_positive, key=lambda row: row["dual_score"], reverse=True)[:12],
        "top_overall": sorted(candidates, key=lambda row: row["overall_gain"], reverse=True)[:12],
        "hypotheses": {
            "H1_dual_positive_region_exists": bool(len(dual_positive) > 0),
            "H2_full_positive_region_exists": bool(len(full_positive) > 0),
            "H3_best_overall_beats_zero": bool(max(row["overall_gain"] for row in candidates) > 0.0),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "best_overall": results["best_overall"],
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
