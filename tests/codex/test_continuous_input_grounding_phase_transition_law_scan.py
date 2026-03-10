#!/usr/bin/env python
"""
Task block D:
Explicit phase-transition law scan.

Instead of only adding more stages, this script learns when a phase-2 concept
should transition from:
    plastic -> promoted stable-new

The transition score depends on:
- novelty confidence
- family retention pressure
- distance margin vs old stable concepts
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


class PhaseTransitionGrounder:
    def __init__(
        self,
        threshold_quantile: float,
        threshold_bonus: float,
        posterior_kappa: float,
        novelty_weight: float,
        retention_weight: float,
        margin_weight: float,
        transition_bias: float,
        plastic_penalty: float,
        promoted_bonus: float,
        route_margin: float,
    ) -> None:
        self.threshold_quantile = float(threshold_quantile)
        self.threshold_bonus = float(threshold_bonus)
        self.posterior_kappa = float(posterior_kappa)
        self.novelty_weight = float(novelty_weight)
        self.retention_weight = float(retention_weight)
        self.margin_weight = float(margin_weight)
        self.transition_bias = float(transition_bias)
        self.plastic_penalty = float(plastic_penalty)
        self.promoted_bonus = float(promoted_bonus)
        self.route_margin = float(route_margin)

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
        self.family_retention_pressure: Dict[str, float] = {}

        self.phase2_mean: Dict[str, np.ndarray] = {}
        self.phase2_sq: Dict[str, np.ndarray] = {}
        self.phase2_count: Dict[str, int] = {}
        self.phase2_post: Dict[str, np.ndarray] = {}
        self.phase2_confidence: Dict[str, float] = {}
        self.phase2_margin: Dict[str, float] = {}
        self.phase2_transition: Dict[str, float] = {}
        self.promoted_new: Dict[str, np.ndarray] = {}
        self.promoted_score: Dict[str, float] = {}

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

        base = self.family_basis[family]
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

        count = self.phase2_count.get(concept, 0)
        mean = self._ema(self.phase2_mean.get(concept), centered, count, 0.55)
        sq = self._ema(self.phase2_sq.get(concept), np.square(centered).astype(np.float32), count, 0.55)
        self.phase2_mean[concept] = mean
        self.phase2_sq[concept] = sq
        self.phase2_count[concept] = count + 1
        self._refresh_phase2(concept, family)

    def consolidate_phase1(self) -> None:
        self.phase1_concepts = {concept for family in proto.PHASE1 for concept in proto.PHASE1[family]}
        for family, scores in self.phase1_scores.items():
            arr = np.array(scores, dtype=np.float32)
            self.family_threshold[family] = float(np.quantile(arr, self.threshold_quantile) + self.threshold_bonus * np.std(arr))

        for family in proto.FAMILIES:
            vars_ = [float(np.mean(self.stable_var[name])) for name in self.phase1_concepts if proto.concept_family(name) == family]
            self.family_retention_pressure[family] = float(np.mean(vars_)) if vars_ else 0.0

    def _family_prior(self, family: str) -> np.ndarray:
        siblings = [self.stable_offsets[name] for name in self.phase1_concepts if proto.concept_family(name) == family]
        if not siblings:
            return np.zeros_like(next(iter(self.family_basis.values())))
        return np.mean(np.stack(siblings, axis=0), axis=0).astype(np.float32)

    def _refresh_phase2(self, concept: str, family: str) -> None:
        mean = self.phase2_mean[concept]
        sq = self.phase2_sq[concept]
        count = self.phase2_count[concept]
        prior = self._family_prior(family)

        weight = float(count / (count + self.posterior_kappa))
        posterior = (weight * mean + (1.0 - weight) * prior).astype(np.float32)
        variance = np.maximum(0.0, sq - np.square(mean)).astype(np.float32)

        sibling_scores = []
        for name in self.phase1_concepts:
            if proto.concept_family(name) != family:
                continue
            sibling_scores.append(proto.sq_dist(posterior, self.stable_offsets[name]))
        margin = float(min(sibling_scores)) if sibling_scores else float(np.linalg.norm(posterior - prior))
        confidence = float(margin / (0.05 + float(np.mean(variance))))
        retention_pressure = self.family_retention_pressure.get(family, 0.0)
        transition = (
            self.novelty_weight * confidence
            - self.retention_weight * retention_pressure
            + self.margin_weight * margin
            - self.transition_bias
        )

        self.phase2_post[concept] = posterior
        self.phase2_confidence[concept] = confidence
        self.phase2_margin[concept] = margin
        self.phase2_transition[concept] = transition
        if transition > 0.0:
            self.promoted_new[concept] = posterior
            self.promoted_score[concept] = transition

    def _stable_old_best(self, x: np.ndarray, family: str) -> Tuple[str | None, float]:
        base = self.family_basis[family]
        centered = (x - base).astype(np.float32)
        best_concept = None
        best_score = float("inf")
        for concept in self.phase1_concepts:
            if proto.concept_family(concept) != family:
                continue
            raw_score = proto.sq_dist(x, self.phase1_proto[concept])
            resid_score = float(np.sum(np.square(centered - self.stable_offsets[concept]) / (self.stable_var[concept] + 0.03)))
            score = 0.70 * raw_score + 0.30 * resid_score
            if score < best_score:
                best_score = score
                best_concept = concept
        return best_concept, best_score

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        family = min(self.family_basis, key=lambda name: proto.sq_dist(x, self.family_basis[name]))
        base = self.family_basis[family]
        old_best, old_score = self._stable_old_best(x, family)
        threshold = self.family_threshold.get(family, old_score)

        promoted_best = None
        promoted_score = float("inf")
        for concept, offset in self.promoted_new.items():
            if proto.concept_family(concept) != family:
                continue
            score = proto.sq_dist(x, base + offset) - self.promoted_bonus * self.promoted_score.get(concept, 0.0)
            if score < promoted_score:
                promoted_score = score
                promoted_best = concept

        plastic_best = None
        plastic_score = float("inf")
        for concept, offset in self.phase2_post.items():
            if proto.concept_family(concept) != family:
                continue
            score = proto.sq_dist(x, base + offset) + self.plastic_penalty / (0.10 + max(0.0, self.phase2_transition.get(concept, 0.0)))
            if score < plastic_score:
                plastic_score = score
                plastic_best = concept

        if promoted_best is not None and old_score > threshold and promoted_score + self.route_margin < old_score:
            return family, promoted_best
        if plastic_best is not None and old_score > threshold * 1.02 and plastic_score + self.route_margin < old_score:
            return family, plastic_best

        assert old_best is not None
        return family, old_best


def run_candidate(
    threshold_quantile: float,
    threshold_bonus: float,
    posterior_kappa: float,
    novelty_weight: float,
    retention_weight: float,
    margin_weight: float,
    transition_bias: float,
    plastic_penalty: float,
    promoted_bonus: float,
    route_margin: float,
    seed: int,
    noise: float,
    dropout_p: float,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    direct = proto.DirectPrototypeLearner(dim=16)
    model = PhaseTransitionGrounder(
        threshold_quantile=threshold_quantile,
        threshold_bonus=threshold_bonus,
        posterior_kappa=posterior_kappa,
        novelty_weight=novelty_weight,
        retention_weight=retention_weight,
        margin_weight=margin_weight,
        transition_bias=transition_bias,
        plastic_penalty=plastic_penalty,
        promoted_bonus=promoted_bonus,
        route_margin=route_margin,
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
    ap = argparse.ArgumentParser(description="Phase-transition law scan for continuous grounding")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/continuous_input_grounding_phase_transition_law_scan_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    candidates = []
    for threshold_quantile in [0.80, 0.85]:
        for threshold_bonus in [0.00]:
            for posterior_kappa in [0.5, 1.5]:
                for novelty_weight in [0.8, 1.2]:
                    for retention_weight in [0.2, 0.4, 0.6]:
                        for margin_weight in [0.0, 0.05]:
                            for transition_bias in [0.2, 0.5, 0.8]:
                                for plastic_penalty in [0.01, 0.03]:
                                    for promoted_bonus in [0.02, 0.05]:
                                        for route_margin in [0.00, 0.03]:
                                            rows = []
                                            for offset in range(int(args.num_seeds)):
                                                rows.append(
                                                    run_candidate(
                                                        threshold_quantile=threshold_quantile,
                                                        threshold_bonus=threshold_bonus,
                                                        posterior_kappa=posterior_kappa,
                                                        novelty_weight=novelty_weight,
                                                        retention_weight=retention_weight,
                                                        margin_weight=margin_weight,
                                                        transition_bias=transition_bias,
                                                        plastic_penalty=plastic_penalty,
                                                        promoted_bonus=promoted_bonus,
                                                        route_margin=route_margin,
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
                                                    "threshold_quantile": float(threshold_quantile),
                                                    "threshold_bonus": float(threshold_bonus),
                                                    "posterior_kappa": float(posterior_kappa),
                                                    "novelty_weight": float(novelty_weight),
                                                    "retention_weight": float(retention_weight),
                                                    "margin_weight": float(margin_weight),
                                                    "transition_bias": float(transition_bias),
                                                    "plastic_penalty": float(plastic_penalty),
                                                    "promoted_bonus": float(promoted_bonus),
                                                    "route_margin": float(route_margin),
                                                    **summary,
                                                    "novel_gain": novel_gain,
                                                    "retention_gain": retention_gain,
                                                    "overall_gain": overall_gain,
                                                    "dual_score": float(novel_gain + retention_gain),
                                                    "full_score": float(novel_gain + retention_gain + overall_gain),
                                                }
                                            )

    dual_positive = [row for row in candidates if row["novel_gain"] > 0.0 and row["retention_gain"] > 0.0]
    full_positive = [
        row
        for row in candidates
        if row["novel_gain"] > 0.0 and row["retention_gain"] > 0.0 and row["overall_gain"] > 0.0
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
        "best_full_positive": max(full_positive, key=lambda row: row["full_score"]) if full_positive else None,
        "dual_positive_count": len(dual_positive),
        "full_positive_count": len(full_positive),
        "top_dual_positive": sorted(dual_positive, key=lambda row: row["dual_score"], reverse=True)[:12],
        "top_overall": sorted(candidates, key=lambda row: row["overall_gain"], reverse=True)[:12],
        "hypotheses": {
            "H1_dual_positive_region_exists": bool(len(dual_positive) > 0),
            "H2_full_positive_region_exists": bool(len(full_positive) > 0),
            "H3_best_overall_positive": bool(sorted(candidates, key=lambda row: row["overall_gain"], reverse=True)[0]["overall_gain"] > 0.0),
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
                "best_full_positive": results["best_full_positive"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(json.dumps(results["hypotheses"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
