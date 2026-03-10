#!/usr/bin/env python
"""
Task block D:
Scan an offset-stabilization layer on top of the base+offset law.

The new ingredient is a protected old-offset subspace:
- phase-1 concept offsets define a stable family subspace
- phase-2 novelty must pass an orthogonal residual gate before it can win

This directly tests whether D is blocked less by representation and more by
missing stabilization at write/read time.
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


def affine_basis(xs: List[np.ndarray], rank_k: int) -> Tuple[np.ndarray, np.ndarray]:
    mat = np.stack(xs, axis=0).astype(np.float32)
    mu = np.mean(mat, axis=0).astype(np.float32)
    centered = mat - mu[None, :]
    if centered.shape[0] <= 1:
        return mu, np.zeros((mat.shape[1], 0), dtype=np.float32)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    k = int(min(rank_k, vh.shape[0]))
    basis = vh[:k].T.astype(np.float32) if k > 0 else np.zeros((mat.shape[1], 0), dtype=np.float32)
    return mu, basis


def project(vec: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.zeros_like(vec, dtype=np.float32)
    coeff = basis.T @ vec
    return (basis @ coeff).astype(np.float32)


class OffsetStabilizedGrounder:
    def __init__(
        self,
        threshold_quantile: float,
        threshold_bonus: float,
        posterior_kappa: float,
        parallel_scale: float,
        orth_scale: float,
        orth_gate_scale: float,
        route_margin: float,
    ) -> None:
        self.threshold_quantile = float(threshold_quantile)
        self.threshold_bonus = float(threshold_bonus)
        self.posterior_kappa = float(posterior_kappa)
        self.parallel_scale = float(parallel_scale)
        self.orth_scale = float(orth_scale)
        self.orth_gate_scale = float(orth_gate_scale)
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
        self.family_orth_threshold: Dict[str, float] = {}

        self.phase2_mean: Dict[str, np.ndarray] = {}
        self.phase2_count: Dict[str, int] = {}
        self.phase2_candidate: Dict[str, np.ndarray] = {}
        self.phase2_novelty: Dict[str, float] = {}

    @staticmethod
    def _ema(prev: np.ndarray | None, x: np.ndarray, count: int, alpha_cap: float) -> np.ndarray:
        if prev is None:
            return x.astype(np.float32).copy()
        alpha = min(alpha_cap, 1.0 / float(count + 1))
        return ((1.0 - alpha) * prev + alpha * x).astype(np.float32)

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        family_count = self.family_count.get(family, 0)
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

        concept_count = self.phase2_count.get(concept, 0)
        sample_mean = self._ema(self.phase2_mean.get(concept), centered, concept_count, 0.55)
        self.phase2_mean[concept] = sample_mean
        self.phase2_count[concept] = concept_count + 1
        self._refresh_candidate(concept, family)

    def consolidate_phase1(self) -> None:
        self.phase1_concepts = {concept for family in proto.PHASE1 for concept in proto.PHASE1[family]}
        for family, scores in self.phase1_scores.items():
            arr = np.array(scores, dtype=np.float32)
            self.family_threshold[family] = float(np.quantile(arr, self.threshold_quantile) + self.threshold_bonus * np.std(arr))

        for family in proto.FAMILIES:
            mean_offset, basis = self._family_offset_basis(family)
            orth_norms = []
            for concept in self.phase1_concepts:
                if proto.concept_family(concept) != family:
                    continue
                centered = (self.stable_offsets[concept] - mean_offset).astype(np.float32)
                orth = centered - project(centered, basis)
                orth_norms.append(float(np.linalg.norm(orth)))
            arr = np.array(orth_norms or [0.0], dtype=np.float32)
            self.family_orth_threshold[family] = float(np.quantile(arr, 0.85) + 0.25 * np.std(arr))

    def _family_offset_basis(self, family: str) -> Tuple[np.ndarray, np.ndarray]:
        offsets = [self.stable_offsets[name] for name in self.phase1_concepts if proto.concept_family(name) == family]
        if not offsets:
            dim = next(iter(self.family_basis.values())).shape[0]
            return np.zeros(dim, dtype=np.float32), np.zeros((dim, 0), dtype=np.float32)
        return affine_basis(offsets, rank_k=min(2, len(offsets) - 1))

    def _refresh_candidate(self, concept: str, family: str) -> None:
        sample_mean = self.phase2_mean[concept]
        family_mean_offset, family_basis = self._family_offset_basis(family)
        centered = (sample_mean - family_mean_offset).astype(np.float32)
        parallel = project(centered, family_basis)
        orth = (centered - parallel).astype(np.float32)

        candidate_offset = (
            family_mean_offset
            + self.parallel_scale * parallel
            + self.orth_scale * orth
        ).astype(np.float32)
        count = self.phase2_count[concept]
        weight = float(count / (count + self.posterior_kappa))
        posterior_offset = (weight * candidate_offset + (1.0 - weight) * family_mean_offset).astype(np.float32)

        sibling_scores = []
        for name in self.phase1_concepts:
            if proto.concept_family(name) != family:
                continue
            sibling_scores.append(proto.sq_dist(posterior_offset, self.stable_offsets[name]))
        novelty = float(min(sibling_scores)) if sibling_scores else float(np.linalg.norm(orth))

        self.phase2_candidate[concept] = posterior_offset
        self.phase2_novelty[concept] = novelty

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
            score = 0.72 * raw_score + 0.28 * resid_score
            if score < best_score:
                best_score = score
                best_concept = concept
        return best_concept, best_score

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        family = min(self.family_basis, key=lambda name: proto.sq_dist(x, self.family_basis[name]))
        base = self.family_basis[family]
        stable_best, stable_score = self._stable_best(x, family)
        threshold = self.family_threshold.get(family, stable_score)

        mean_offset, basis = self._family_offset_basis(family)
        centered = (x - base - mean_offset).astype(np.float32)
        orth_norm = float(np.linalg.norm(centered - project(centered, basis)))
        orth_threshold = self.family_orth_threshold.get(family, 0.0) * self.orth_gate_scale

        novel_best = None
        novel_score = float("inf")
        for concept, offset in self.phase2_candidate.items():
            if proto.concept_family(concept) != family:
                continue
            proto_x = (base + offset).astype(np.float32)
            offset_centered = (offset - mean_offset).astype(np.float32)
            offset_orth = offset_centered - project(offset_centered, basis)
            score = (
                0.62 * proto.sq_dist(x, proto_x)
                + 0.18 * self.phase2_novelty.get(concept, 0.0)
                + 0.20 * abs(orth_norm - float(np.linalg.norm(offset_orth)))
            )
            if score < novel_score:
                novel_score = score
                novel_best = concept

        if novel_best is not None:
            if orth_norm > orth_threshold and stable_score > threshold and novel_score + self.route_margin < stable_score:
                return family, novel_best
            if orth_norm > 1.35 * orth_threshold and novel_score + self.route_margin < 0.90 * stable_score:
                return family, novel_best

        assert stable_best is not None
        return family, stable_best


def run_candidate(
    threshold_quantile: float,
    threshold_bonus: float,
    posterior_kappa: float,
    parallel_scale: float,
    orth_scale: float,
    orth_gate_scale: float,
    route_margin: float,
    seed: int,
    noise: float,
    dropout_p: float,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    direct = proto.DirectPrototypeLearner(dim=16)
    model = OffsetStabilizedGrounder(
        threshold_quantile=threshold_quantile,
        threshold_bonus=threshold_bonus,
        posterior_kappa=posterior_kappa,
        parallel_scale=parallel_scale,
        orth_scale=orth_scale,
        orth_gate_scale=orth_gate_scale,
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
    ap = argparse.ArgumentParser(description="Offset stabilization scan for continuous grounding")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/continuous_input_grounding_offset_stabilization_scan_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    candidates = []
    for threshold_quantile in [0.80, 0.85]:
        for threshold_bonus in [0.00, 0.10, 0.20]:
            for posterior_kappa in [0.5, 1.5]:
                for parallel_scale in [0.60, 0.80]:
                    for orth_scale in [1.00, 1.20, 1.40]:
                        for orth_gate_scale in [1.00, 1.20, 1.40]:
                            for route_margin in [0.00, 0.04]:
                                rows = []
                                for offset in range(int(args.num_seeds)):
                                    rows.append(
                                        run_candidate(
                                            threshold_quantile=threshold_quantile,
                                            threshold_bonus=threshold_bonus,
                                            posterior_kappa=posterior_kappa,
                                            parallel_scale=parallel_scale,
                                            orth_scale=orth_scale,
                                            orth_gate_scale=orth_gate_scale,
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
                                        "parallel_scale": float(parallel_scale),
                                        "orth_scale": float(orth_scale),
                                        "orth_gate_scale": float(orth_gate_scale),
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
