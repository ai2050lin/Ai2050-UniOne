#!/usr/bin/env python
"""
Task block D:
phase-state dependent consolidation scan.

This extends D with an explicit hidden phase variable. The state is driven by
phase-2 novelty and then used to modulate write-back strength. It tests whether
"having a state variable at all" is enough, even before introducing a richer
learned controller.
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


class PhaseStateConsolidator:
    def __init__(
        self,
        kappa0: float,
        decay: float,
        gate_quantile: float,
        phase_alpha: float,
        write_alpha: float,
    ) -> None:
        self.kappa0 = float(kappa0)
        self.decay = float(decay)
        self.gate_quantile = float(gate_quantile)
        self.phase_alpha = float(phase_alpha)
        self.write_alpha = float(write_alpha)
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
        self.unified_proto: Dict[str, np.ndarray] = {}
        self.unified_count: Dict[str, int] = {}
        self.phase_state = 0.0
        self.phase_history: List[float] = []

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
        self.phase2_post[concept] = posterior
        self.phase2_count[concept] = concept_count + 1

        sibling_distances = [proto.sq_dist(posterior, self.phase1_proto[name]) for name in self.phase1_concepts if proto.concept_family(name) == family]
        novelty = float(min(sibling_distances)) if sibling_distances else float(proto.sq_dist(posterior, prior))
        self.phase_state = (1.0 - self.phase_alpha) * self.phase_state + self.phase_alpha * novelty
        self.phase_history.append(self.phase_state)

    def consolidate_phase1(self) -> None:
        self.phase1_concepts = {concept for family in proto.PHASE1 for concept in proto.PHASE1[family]}
        for family, scores in self.phase1_scores.items():
            arr = np.array(scores, dtype=np.float32)
            self.family_threshold[family] = float(np.quantile(arr, self.gate_quantile) + 0.15 * np.std(arr))

    def state_writeback(self) -> None:
        if not self.phase2_post:
            return
        state_level = float(np.quantile(np.array(self.phase_history, dtype=np.float32), self.gate_quantile)) if self.phase_history else 0.0

        for concept in self.phase1_concepts:
            prev = self.unified_proto.get(concept)
            target = self.phase1_proto[concept]
            alpha = self.write_alpha * (1.0 + min(1.0, state_level) * 0.25)
            self.unified_proto[concept] = ((1.0 - alpha) * prev + alpha * target).astype(np.float32) if prev is not None else target.copy()

        for concept, posterior in self.phase2_post.items():
            family = proto.concept_family(concept)
            sibling_distances = [proto.sq_dist(posterior, self.phase1_proto[name]) for name in self.phase1_concepts if proto.concept_family(name) == family]
            novelty = min(sibling_distances) if sibling_distances else 0.0
            if novelty < state_level:
                alpha = self.write_alpha * (0.5 + 0.5 * min(1.0, novelty / (state_level + 1e-6)))
            else:
                alpha = self.write_alpha * (0.15 + 0.25 * min(2.0, novelty / (state_level + 1e-6)))
            prev = self.unified_proto.get(concept)
            self.unified_proto[concept] = ((1.0 - alpha) * prev + alpha * posterior).astype(np.float32) if prev is not None else posterior.copy()

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
            score = 0.75 * raw_score + 0.25 * resid_score
            if score < stable_score:
                stable_score = score
                stable_best = concept

        threshold = self.family_threshold.get(family, stable_score)
        if stable_score <= threshold:
            return family, stable_best

        best_concept = stable_best
        best_score = proto.sq_dist(x, self.unified_proto.get(stable_best, self.phase1_proto[stable_best])) if stable_best is not None else float("inf")
        for concept, center in self.unified_proto.items():
            if proto.concept_family(concept) != family:
                continue
            score = proto.sq_dist(x, center)
            if score < best_score:
                best_score = score
                best_concept = concept
        return family, best_concept


def run_candidate(
    kappa0: float,
    decay: float,
    gate_quantile: float,
    phase_alpha: float,
    write_alpha: float,
    seed: int,
    noise: float,
    dropout_p: float,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    direct = proto.DirectPrototypeLearner(dim=16)
    model = PhaseStateConsolidator(
        kappa0=kappa0,
        decay=decay,
        gate_quantile=gate_quantile,
        phase_alpha=phase_alpha,
        write_alpha=write_alpha,
    )

    for _ in range(42):
        for family, concepts in proto.PHASE1.items():
            for concept in concepts:
                x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                direct.train(x, family, concept)
                model.train(x, family, concept)

    model.consolidate_phase1()
    model.state_writeback()

    for _ in range(3):
        for family, concepts in proto.PHASE2.items():
            for concept in concepts:
                x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                direct.train(x, family, concept)
                model.train(x, family, concept)

    model.state_writeback()
    novel_direct = eval_concept_accuracy(direct, proto.PHASE2, 24, rng, noise, dropout_p)
    novel_model = eval_concept_accuracy(model, proto.PHASE2, 24, rng, noise, dropout_p)

    for _ in range(18):
        for family, concepts in proto.PHASE2.items():
            for concept in concepts:
                x = proto.sample_continuous_input(rng, concept, noise, dropout_p)
                direct.train(x, family, concept)
                model.train(x, family, concept)

    model.state_writeback()
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
    ap = argparse.ArgumentParser(description="Phase-state consolidation scan for continuous grounding")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/continuous_input_grounding_phase_state_scan_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    candidates = []
    for kappa0 in [6.0, 8.0, 10.0, 12.0]:
        for decay in [0.25, 0.50, 0.75]:
            for gate_quantile in [0.70, 0.80, 0.90]:
                for phase_alpha in [0.15, 0.25, 0.35]:
                    for write_alpha in [0.04, 0.08, 0.12]:
                        rows = []
                        for offset in range(int(args.num_seeds)):
                            rows.append(
                                run_candidate(
                                    kappa0=kappa0,
                                    decay=decay,
                                    gate_quantile=gate_quantile,
                                    phase_alpha=phase_alpha,
                                    write_alpha=write_alpha,
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
                                "gate_quantile": float(gate_quantile),
                                "phase_alpha": float(phase_alpha),
                                "write_alpha": float(write_alpha),
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
