#!/usr/bin/env python
"""
Task block D:
scan a residual-manifold novelty gate with explicit stable/plastic stores.

The method is more structural than the old threshold scan:
- stable store keeps phase-1 concept offsets in a family-centered manifold
- plastic store learns few-shot phase-2 offsets with a sibling-informed prior
- novelty gate uses the phase-1 residual score distribution to decide when a
  sample should leave the stable route and enter the plastic route

This tests whether a simple consolidation law can produce a region where
novel-concept gain and retention gain are simultaneously positive.
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


class ResidualGateConsolidationGrounder:
    def __init__(self, quantile: float, threshold_bonus: float) -> None:
        self.quantile = float(quantile)
        self.threshold_bonus = float(threshold_bonus)
        self.family_basis: Dict[str, np.ndarray] = {}
        self.family_count: Dict[str, int] = {}
        self.phase1_concepts: set[str] = set()
        self.stable_offsets: Dict[str, np.ndarray] = {}
        self.stable_var: Dict[str, np.ndarray] = {}
        self.stable_count: Dict[str, int] = {}
        self.plastic_offsets: Dict[str, np.ndarray] = {}
        self.plastic_count: Dict[str, int] = {}
        self.raw_phase2_proto: Dict[str, np.ndarray] = {}
        self.raw_phase2_count: Dict[str, int] = {}
        self.phase1_scores: Dict[str, List[float]] = {}
        self.family_threshold: Dict[str, float] = {}

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
            concept_count = self.stable_count.get(concept, 0)
            offset = self._ema(self.stable_offsets.get(concept), centered, concept_count, 0.22)
            residual = (centered - offset).astype(np.float32)
            var = self._ema(
                self.stable_var.get(concept),
                np.square(residual).astype(np.float32),
                concept_count,
                0.22,
            )
            self.stable_offsets[concept] = offset
            self.stable_var[concept] = var
            self.stable_count[concept] = concept_count + 1
            score = float(np.sum(np.square(residual) / (var + 0.03)))
            self.phase1_scores.setdefault(family, []).append(score)
            return

        plastic_count = self.plastic_count.get(concept, 0)
        sample_offset = self._ema(self.plastic_offsets.get(concept), centered, plastic_count, 0.55)
        siblings = [value for name, value in self.stable_offsets.items() if proto.concept_family(name) == family]
        sibling_scale = (
            np.mean(np.stack([np.abs(value) for value in siblings], axis=0), axis=0).astype(np.float32)
            if siblings
            else np.full_like(centered, 0.05, dtype=np.float32)
        )
        prior_strength = 1.5
        posterior = (
            ((plastic_count + 1) / float(plastic_count + 1 + prior_strength)) * sample_offset
            + (prior_strength / float(plastic_count + 1 + prior_strength)) * np.sign(sample_offset + 1e-6) * sibling_scale
        ).astype(np.float32)
        self.plastic_offsets[concept] = posterior
        self.plastic_count[concept] = plastic_count + 1

        raw_count = self.raw_phase2_count.get(concept, 0)
        self.raw_phase2_proto[concept] = self._ema(self.raw_phase2_proto.get(concept), x, raw_count, 0.50)
        self.raw_phase2_count[concept] = raw_count + 1

    def consolidate_phase1(self) -> None:
        self.phase1_concepts = {concept for family in proto.PHASE1 for concept in proto.PHASE1[family]}
        for family, scores in self.phase1_scores.items():
            arr = np.array(scores, dtype=np.float32)
            self.family_threshold[family] = float(np.quantile(arr, self.quantile) + self.threshold_bonus * np.std(arr))

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        family = min(self.family_basis, key=lambda name: proto.sq_dist(x, self.family_basis[name]))
        base = self.family_basis[family]
        centered = (x - base).astype(np.float32)

        stable_best = None
        stable_score = float("inf")
        for concept, offset in self.stable_offsets.items():
            if proto.concept_family(concept) != family:
                continue
            var = self.stable_var.get(concept, np.full_like(offset, 0.05, dtype=np.float32))
            score = float(np.sum(np.square(centered - offset) / (var + 0.03)))
            if score < stable_score:
                stable_score = score
                stable_best = concept

        plastic_best = None
        plastic_score = float("inf")
        for concept, offset in self.plastic_offsets.items():
            if proto.concept_family(concept) != family:
                continue
            raw = self.raw_phase2_proto.get(concept, base + offset)
            score = 0.65 * proto.sq_dist(x, raw) + 0.35 * proto.sq_dist(centered, offset)
            if score < plastic_score:
                plastic_score = score
                plastic_best = concept

        threshold = self.family_threshold.get(family, stable_score)
        if plastic_best is not None and stable_score > threshold:
            return family, plastic_best
        if stable_best is not None:
            return family, stable_best
        assert plastic_best is not None
        return family, plastic_best


def run_candidate(
    quantile: float,
    threshold_bonus: float,
    seed: int,
    noise: float,
    dropout_p: float,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    direct = proto.DirectPrototypeLearner(dim=16)
    model = ResidualGateConsolidationGrounder(quantile=quantile, threshold_bonus=threshold_bonus)

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
    ap = argparse.ArgumentParser(description="Residual-manifold consolidation scan for continuous grounding")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/continuous_input_grounding_consolidation_law_scan_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    candidates = []
    for quantile in [0.70, 0.75, 0.80, 0.85, 0.90]:
        for threshold_bonus in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            rows = []
            for offset in range(int(args.num_seeds)):
                rows.append(
                    run_candidate(
                        quantile=quantile,
                        threshold_bonus=threshold_bonus,
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
                    **summary,
                    "novel_gain": novel_gain,
                    "retention_gain": retention_gain,
                    "overall_gain": overall_gain,
                    "dual_score": float(novel_gain + retention_gain),
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
