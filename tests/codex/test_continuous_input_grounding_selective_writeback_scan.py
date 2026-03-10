#!/usr/bin/env python
"""
Task block D:
selective write-back scan as a stricter negative control.

Compared with naive unified replay, this method tries to be more careful:
- keep a stable phase-1 manifold score
- only open write-back when stable residual exceeds a family threshold
- only allow phase-2 posterior concepts with sufficient confidence

If this still cannot produce full-positive closure, the missing piece is not
"more careful replay" alone, but a stronger state-dependent consolidation law.
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


class SelectiveWritebackConsolidator:
    def __init__(
        self,
        quantile: float,
        threshold_bonus: float,
        kappa0: float,
        decay: float,
        write_alpha: float,
        confidence_quantile: float,
        replay_steps: int,
    ) -> None:
        self.quantile = float(quantile)
        self.threshold_bonus = float(threshold_bonus)
        self.kappa0 = float(kappa0)
        self.decay = float(decay)
        self.write_alpha = float(write_alpha)
        self.confidence_quantile = float(confidence_quantile)
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
        self.phase2_posterior: Dict[str, np.ndarray] = {}
        self.phase2_confidence: Dict[str, float] = {}
        self.unified_proto: Dict[str, np.ndarray] = {}
        self.unified_count: Dict[str, int] = {}

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
            self.phase1_scores.setdefault(family, []).append(float(np.sum(np.square(residual) / (var + 0.03))))

            unified_count = self.unified_count.get(concept, 0)
            self.unified_proto[concept] = self._ema(self.unified_proto.get(concept), x, unified_count, 0.24)
            self.unified_count[concept] = unified_count + 1
            return

        concept_count = self.phase2_count.get(concept, 0)
        sample_mean = self._ema(self.phase2_mean.get(concept), x, concept_count, 0.55)
        siblings = [self.phase1_proto[name] for name in self.phase1_concepts if proto.concept_family(name) == family]
        prior = np.mean(np.stack(siblings, axis=0), axis=0).astype(np.float32) if siblings else self.family_basis[family]
        kappa = self.kappa0 / float((concept_count + 1) ** self.decay)
        posterior = (
            ((concept_count + 1) / float(concept_count + 1 + kappa)) * sample_mean
            + (kappa / float(concept_count + 1 + kappa)) * prior
        ).astype(np.float32)

        self.phase2_mean[concept] = sample_mean
        self.phase2_posterior[concept] = posterior
        self.phase2_count[concept] = concept_count + 1

        sibling_distances = [proto.sq_dist(posterior, self.phase1_proto[name]) for name in self.phase1_concepts if proto.concept_family(name) == family]
        self.phase2_confidence[concept] = float(min(sibling_distances)) if sibling_distances else float(proto.sq_dist(posterior, prior))

    def consolidate_phase1(self) -> None:
        self.phase1_concepts = {concept for family in proto.PHASE1 for concept in proto.PHASE1[family]}
        for family, scores in self.phase1_scores.items():
            arr = np.array(scores, dtype=np.float32)
            self.family_threshold[family] = float(np.quantile(arr, self.quantile) + self.threshold_bonus * np.std(arr))

    def selective_writeback(self) -> None:
        if not self.phase2_posterior:
            return

        conf_values = np.array(list(self.phase2_confidence.values()), dtype=np.float32)
        conf_threshold = float(np.quantile(conf_values, self.confidence_quantile)) if conf_values.size else float("inf")

        for _ in range(self.replay_steps):
            for concept in self.phase1_concepts:
                target = self.phase1_proto[concept]
                prev = self.unified_proto.get(concept)
                if prev is None:
                    self.unified_proto[concept] = target.astype(np.float32).copy()
                else:
                    self.unified_proto[concept] = ((1.0 - self.write_alpha) * prev + self.write_alpha * target).astype(np.float32)

            for concept, posterior in self.phase2_posterior.items():
                if self.phase2_confidence.get(concept, float("inf")) > conf_threshold:
                    continue
                prev = self.unified_proto.get(concept)
                alpha = self.write_alpha * 0.6
                if prev is None:
                    self.unified_proto[concept] = posterior.astype(np.float32).copy()
                else:
                    self.unified_proto[concept] = ((1.0 - alpha) * prev + alpha * posterior).astype(np.float32)

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        family = min(self.family_basis, key=lambda name: proto.sq_dist(x, self.family_basis[name]))
        base = self.family_basis[family]
        centered = (x - base).astype(np.float32)

        stable_best = None
        stable_score = float("inf")
        for concept in self.phase1_concepts:
            if proto.concept_family(concept) != family:
                continue
            raw_score = proto.sq_dist(x, self.phase1_proto[concept])
            resid_score = float(np.sum(np.square(centered - self.stable_offsets[concept]) / (self.stable_var[concept] + 0.03)))
            score = 0.8 * raw_score + 0.2 * resid_score
            if score < stable_score:
                stable_score = score
                stable_best = concept

        threshold = self.family_threshold.get(family, stable_score)
        if stable_score <= threshold:
            return family, stable_best

        best_concept = stable_best
        best_score = proto.sq_dist(x, self.unified_proto[stable_best]) if stable_best is not None else float("inf")
        for concept, center in self.unified_proto.items():
            if proto.concept_family(concept) != family:
                continue
            score = proto.sq_dist(x, center)
            if score < best_score:
                best_score = score
                best_concept = concept
        return family, best_concept


def run_candidate(
    quantile: float,
    threshold_bonus: float,
    kappa0: float,
    decay: float,
    write_alpha: float,
    confidence_quantile: float,
    replay_steps: int,
    seed: int,
    noise: float,
    dropout_p: float,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    direct = proto.DirectPrototypeLearner(dim=16)
    model = SelectiveWritebackConsolidator(
        quantile=quantile,
        threshold_bonus=threshold_bonus,
        kappa0=kappa0,
        decay=decay,
        write_alpha=write_alpha,
        confidence_quantile=confidence_quantile,
        replay_steps=replay_steps,
    )

    for _ in range(42):
        for family, concepts in proto.PHASE1.items():
            for concept in concepts:
                x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                direct.train(x, family, concept)
                model.train(x, family, concept)

    model.consolidate_phase1()
    model.selective_writeback()

    for _ in range(3):
        for family, concepts in proto.PHASE2.items():
            for concept in concepts:
                x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                direct.train(x, family, concept)
                model.train(x, family, concept)

    model.selective_writeback()
    novel_direct = eval_concept_accuracy(direct, proto.PHASE2, 24, rng, noise, dropout_p)
    novel_model = eval_concept_accuracy(model, proto.PHASE2, 24, rng, noise, dropout_p)

    for _ in range(18):
        for family, concepts in proto.PHASE2.items():
            for concept in concepts:
                x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                direct.train(x, family, concept)
                model.train(x, family, concept)

    model.selective_writeback()
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
    ap = argparse.ArgumentParser(description="Selective write-back scan for continuous grounding")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/continuous_input_grounding_selective_writeback_scan_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    candidates = []
    for quantile in [0.75, 0.80, 0.85]:
        for threshold_bonus in [0.10, 0.20]:
            for kappa0 in [8.0, 10.0, 12.0]:
                for decay in [0.25, 0.50]:
                    for write_alpha in [0.04, 0.08]:
                        for confidence_quantile in [0.50, 0.65, 0.80]:
                            for replay_steps in [1, 2]:
                                rows = []
                                for offset in range(int(args.num_seeds)):
                                    rows.append(
                                        run_candidate(
                                            quantile=quantile,
                                            threshold_bonus=threshold_bonus,
                                            kappa0=kappa0,
                                            decay=decay,
                                            write_alpha=write_alpha,
                                            confidence_quantile=confidence_quantile,
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
                                        "kappa0": float(kappa0),
                                        "decay": float(decay),
                                        "write_alpha": float(write_alpha),
                                        "confidence_quantile": float(confidence_quantile),
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
