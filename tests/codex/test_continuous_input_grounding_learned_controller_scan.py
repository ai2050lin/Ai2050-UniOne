#!/usr/bin/env python
"""
Task block D:
learned-controller consolidation scan.

This is the first controller-level upgrade after the negative controls:
- keep stable store / plastic store split
- build a nonlinear controller over a higher-dimensional state vector
- adapt controller parameters by a small query-based coordinate search

The goal is to test whether a lightweight learned scheduler can turn the
existing dual-positive region into a full-positive region.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import test_continuous_input_grounding_proto as proto


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -12.0, 12.0)))


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


class LearnedControllerGrounder:
    def __init__(
        self,
        controller_seed: int,
        base_alpha: float,
        gate_bias: float,
        adapt_lr: float,
        adapt_steps: int,
    ) -> None:
        self.base_alpha = float(base_alpha)
        self.gate_bias = float(gate_bias)
        self.adapt_lr = float(adapt_lr)
        self.adapt_steps = int(adapt_steps)

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
        self.family_novelty: Dict[str, List[float]] = {family: [] for family in proto.FAMILIES}
        self.phase2_seen = 0

        rng = np.random.default_rng(controller_seed)
        self.controller_u = rng.normal(scale=0.55, size=(6, 6)).astype(np.float32)
        self.controller_w = rng.normal(scale=0.55, size=(4, 6)).astype(np.float32)
        self.controller_bias = np.zeros(4, dtype=np.float32)
        self.controller_gain = 1.0

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
        self.phase2_seen += 1

        sibling_distances = [
            proto.sq_dist(posterior, self.phase1_proto[name])
            for name in self.phase1_concepts
            if proto.concept_family(name) == family
        ]
        novelty = float(min(sibling_distances)) if sibling_distances else float(proto.sq_dist(posterior, prior))
        self.family_novelty[family].append(novelty)

    def consolidate_phase1(self) -> None:
        self.phase1_concepts = {concept for family in proto.PHASE1 for concept in proto.PHASE1[family]}
        for family, scores in self.phase1_scores.items():
            arr = np.array(scores, dtype=np.float32)
            self.family_threshold[family] = float(np.quantile(arr, 0.82) + 0.15 * np.std(arr))

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

    def _controller_state(self, family: str, concept: str, posterior: np.ndarray) -> np.ndarray:
        siblings = [name for name in self.phase1_concepts if proto.concept_family(name) == family]
        sibling_distances = [proto.sq_dist(posterior, self.phase1_proto[name]) for name in siblings]
        novelty = float(min(sibling_distances)) if sibling_distances else 0.0
        family_mean = np.mean(np.stack([self.phase1_proto[name] for name in siblings], axis=0), axis=0) if siblings else self.family_basis[family]
        family_drift = float(proto.sq_dist(family_mean, self.family_basis[family]))
        confidence = 1.0 / (1.0 + novelty)
        retention_pressure = float(np.mean([np.mean(self.stable_var[name]) for name in siblings] or [0.0]))
        phase_progress = float(np.clip(self.phase2_seen / 36.0, 0.0, 1.0))
        _, stable_score = self._stable_best(posterior, family)
        threshold_margin = float(stable_score - self.family_threshold.get(family, stable_score))
        state = np.array(
            [
                novelty,
                retention_pressure * 10.0,
                family_drift,
                confidence,
                threshold_margin,
                phase_progress,
            ],
            dtype=np.float32,
        )
        mean = np.array([0.50, 0.30, 0.08, 0.50, 0.20, 0.50], dtype=np.float32)
        scale = np.array([0.60, 0.25, 0.12, 0.30, 0.60, 0.35], dtype=np.float32)
        return np.clip((state - mean) / (scale + 1e-6), -3.0, 3.0)

    def _controller_outputs(
        self,
        state: np.ndarray,
        bias: np.ndarray | None = None,
        gain: float | None = None,
    ) -> np.ndarray:
        current_bias = self.controller_bias if bias is None else bias
        current_gain = self.controller_gain if gain is None else gain
        hidden = np.tanh(self.controller_u @ state)
        logits = current_gain * (self.controller_w @ hidden + current_bias)
        return sigmoid(logits.astype(np.float32))

    def _candidate_artifacts(
        self,
        bias: np.ndarray | None = None,
        gain: float | None = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        unified = {concept: proto_value.copy() for concept, proto_value in self.phase1_proto.items()}
        threshold_boost: Dict[str, float] = {family: 0.0 for family in proto.FAMILIES}

        for concept, posterior in self.phase2_post.items():
            family = proto.concept_family(concept)
            state = self._controller_state(family, concept, posterior)
            gate_open, trust, write_strength, protect = [float(v) for v in self._controller_outputs(state, bias=bias, gain=gain)]
            threshold_boost[family] = max(
                threshold_boost[family],
                0.18 * max(0.0, protect - 0.5),
            )
            if gate_open <= self.gate_bias:
                continue
            alpha = self.base_alpha * (0.25 + 1.10 * write_strength) * (0.35 + 0.90 * trust)
            alpha = float(np.clip(alpha, 0.02, 0.28))
            prev = unified.get(concept)
            unified[concept] = ((1.0 - alpha) * prev + alpha * posterior).astype(np.float32) if prev is not None else posterior.copy()

        return unified, threshold_boost

    def _predict_with_artifacts(
        self,
        x: np.ndarray,
        unified: Dict[str, np.ndarray],
        threshold_boost: Dict[str, float],
    ) -> Tuple[str, str]:
        family = min(self.family_basis, key=lambda name: proto.sq_dist(x, self.family_basis[name]))
        stable_best, stable_score = self._stable_best(x, family)
        threshold = self.family_threshold.get(family, stable_score) + threshold_boost.get(family, 0.0)
        if stable_best is not None and stable_score <= threshold:
            return family, stable_best

        best_concept = stable_best
        best_score = float("inf")
        for concept, center in unified.items():
            if proto.concept_family(concept) != family:
                continue
            score = proto.sq_dist(x, center)
            if score < best_score:
                best_score = score
                best_concept = concept
        assert best_concept is not None
        return family, best_concept

    def _objective(
        self,
        bias: np.ndarray,
        gain: float,
        rng: np.random.Generator,
        noise: float,
        dropout_p: float,
    ) -> float:
        unified, threshold_boost = self._candidate_artifacts(bias=bias, gain=gain)
        novel = 0
        novel_total = 0
        retention = 0
        retention_total = 0
        overall = 0
        overall_total = 0
        all_groups = {family: proto.PHASE1[family] + proto.PHASE2[family] for family in proto.FAMILIES}

        for _ in range(6):
            for family, concepts in proto.PHASE2.items():
                for concept in concepts:
                    x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                    _, pred_concept = self._predict_with_artifacts(x, unified, threshold_boost)
                    novel += int(pred_concept == concept)
                    novel_total += 1

        for _ in range(5):
            for family, concepts in proto.PHASE1.items():
                for concept in concepts:
                    x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                    _, pred_concept = self._predict_with_artifacts(x, unified, threshold_boost)
                    retention += int(pred_concept == concept)
                    retention_total += 1

        for _ in range(4):
            for family, concepts in all_groups.items():
                for concept in concepts:
                    x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                    _, pred_concept = self._predict_with_artifacts(x, unified, threshold_boost)
                    overall += int(pred_concept == concept)
                    overall_total += 1

        novel_acc = float(novel / max(1, novel_total))
        retention_acc = float(retention / max(1, retention_total))
        overall_acc = float(overall / max(1, overall_total))
        return float(1.45 * novel_acc + 0.95 * retention_acc + 1.20 * overall_acc)

    def adapt_controller(self, seed: int, noise: float, dropout_p: float) -> None:
        if not self.phase2_post:
            return
        rng = np.random.default_rng(seed)
        bias = self.controller_bias.copy()
        gain = float(self.controller_gain)
        best_score = self._objective(bias, gain, rng, noise, dropout_p)
        delta = float(self.adapt_lr)

        for _ in range(self.adapt_steps):
            improved = False
            for idx in range(len(bias)):
                for direction in (-1.0, 1.0):
                    cand = bias.copy()
                    cand[idx] += direction * delta
                    score = self._objective(cand, gain, rng, noise, dropout_p)
                    if score > best_score:
                        bias = cand
                        best_score = score
                        improved = True
            for direction in (-1.0, 1.0):
                cand_gain = float(np.clip(gain + direction * delta, 0.35, 2.50))
                score = self._objective(bias, cand_gain, rng, noise, dropout_p)
                if score > best_score:
                    gain = cand_gain
                    best_score = score
                    improved = True
            if not improved:
                delta *= 0.5
                if delta < 0.02:
                    break

        self.controller_bias = bias.astype(np.float32)
        self.controller_gain = float(gain)

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        unified, threshold_boost = self._candidate_artifacts()
        return self._predict_with_artifacts(x, unified, threshold_boost)


def run_candidate(
    controller_seed: int,
    base_alpha: float,
    gate_bias: float,
    adapt_lr: float,
    adapt_steps: int,
    seed: int,
    noise: float,
    dropout_p: float,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    direct = proto.DirectPrototypeLearner(dim=16)
    model = LearnedControllerGrounder(
        controller_seed=controller_seed,
        base_alpha=base_alpha,
        gate_bias=gate_bias,
        adapt_lr=adapt_lr,
        adapt_steps=adapt_steps,
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
    model.adapt_controller(seed=seed + 701, noise=noise, dropout_p=dropout_p)

    novel_direct = eval_concept_accuracy(direct, proto.PHASE2, 24, rng, noise, dropout_p)
    novel_model = eval_concept_accuracy(model, proto.PHASE2, 24, rng, noise, dropout_p)

    for _ in range(18):
        for family, concepts in proto.PHASE2.items():
            for concept in concepts:
                x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                direct.train(x, family, concept)
                model.train(x, family, concept)
    model.adapt_controller(seed=seed + 1701, noise=noise, dropout_p=dropout_p)

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
        "controller_gain": float(model.controller_gain),
        "controller_gate_bias_0": float(model.controller_bias[0]),
        "controller_trust_bias_1": float(model.controller_bias[1]),
        "controller_write_bias_2": float(model.controller_bias[2]),
        "controller_protect_bias_3": float(model.controller_bias[3]),
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Learned-controller consolidation scan for continuous grounding")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/continuous_input_grounding_learned_controller_scan_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    candidates = []
    for controller_seed in [5, 11]:
        for base_alpha in [0.08, 0.12, 0.16]:
            for gate_bias in [0.45, 0.55]:
                for adapt_lr in [0.14, 0.22]:
                    rows = []
                    for offset in range(int(args.num_seeds)):
                        rows.append(
                            run_candidate(
                                controller_seed=controller_seed,
                                base_alpha=base_alpha,
                                gate_bias=gate_bias,
                                adapt_lr=adapt_lr,
                                adapt_steps=6,
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
                            "controller_seed": int(controller_seed),
                            "base_alpha": float(base_alpha),
                            "gate_bias": float(gate_bias),
                            "adapt_lr": float(adapt_lr),
                            "adapt_steps": 6,
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
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(json.dumps(results["hypotheses"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
