#!/usr/bin/env python
"""
Task block D upgrade:
precision-weighted shared-basis grounding on continuous inputs.

We test whether family-conditioned centered offsets plus per-dimension
precision can beat the direct prototype on both novel concepts and retention.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import test_continuous_input_grounding_proto as proto


def weighted_sq_dist(a: np.ndarray, b: np.ndarray, precision: np.ndarray) -> float:
    diff = a - b
    return float(np.sum(precision * np.square(diff)))


class PrecisionSharedOffsetGrounder:
    def __init__(
        self,
        dim: int,
        family_lr_scale: float,
        offset_lr_scale: float,
        replay_strength: float,
        shrinkage: float,
        adaptive_readout: bool,
    ) -> None:
        self.dim = dim
        self.family_lr_scale = family_lr_scale
        self.offset_lr_scale = offset_lr_scale
        self.replay_strength = replay_strength
        self.shrinkage = shrinkage
        self.adaptive_readout = adaptive_readout
        self.family_basis: Dict[str, np.ndarray] = {}
        self.family_var: Dict[str, np.ndarray] = {}
        self.family_count: Dict[str, int] = {}
        self.concept_offset: Dict[str, np.ndarray] = {}
        self.concept_var: Dict[str, np.ndarray] = {}
        self.concept_count: Dict[str, int] = {}
        self.concept_family: Dict[str, str] = {}

    def _ema(self, prev: np.ndarray | None, x: np.ndarray, alpha: float) -> np.ndarray:
        if prev is None:
            return x.astype(np.float32).copy()
        return ((1.0 - alpha) * prev + alpha * x).astype(np.float32)

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        family_count = self.family_count.get(family, 0)
        family_alpha = min(0.22, self.family_lr_scale / float(family_count + 1))
        prev_family = self.family_basis.get(family)
        family_mean = self._ema(prev_family, x, family_alpha)
        family_resid = (x - family_mean).astype(np.float32)
        family_var = self._ema(self.family_var.get(family), np.square(family_resid).astype(np.float32), max(0.08, family_alpha))
        self.family_basis[family] = family_mean
        self.family_var[family] = family_var
        self.family_count[family] = family_count + 1

        offset_count = self.concept_count.get(concept, 0)
        offset_alpha = min(0.26, self.offset_lr_scale / float(offset_count + 1))
        centered = (x - family_mean).astype(np.float32)
        prev_offset = self.concept_offset.get(concept)
        next_offset = self._ema(prev_offset, centered, offset_alpha)
        shrink = self.shrinkage * min(1.0, float(offset_count) / 20.0)
        next_offset = ((1.0 - shrink) * next_offset).astype(np.float32)
        concept_resid = (centered - next_offset).astype(np.float32)
        concept_var = self._ema(self.concept_var.get(concept), np.square(concept_resid).astype(np.float32), max(0.08, offset_alpha))
        self.concept_offset[concept] = next_offset
        self.concept_var[concept] = concept_var
        self.concept_count[concept] = offset_count + 1
        self.concept_family[concept] = family

        if self.replay_strength > 0.0:
            for old_concept, old_family in list(self.concept_family.items()):
                if old_concept == concept:
                    continue
                if old_family != family:
                    continue
                old_offset = self.concept_offset[old_concept]
                replay_target = self.family_basis[old_family] + old_offset
                replay_centered = (replay_target - self.family_basis[old_family]).astype(np.float32)
                self.concept_offset[old_concept] = (
                    (1.0 - self.replay_strength) * old_offset + self.replay_strength * replay_centered
                ).astype(np.float32)

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        best_family = None
        best_family_dist = float("inf")
        for family, family_mean in self.family_basis.items():
            precision = 1.0 / (self.family_var.get(family, np.ones(self.dim, dtype=np.float32)) + 0.03)
            dist = weighted_sq_dist(x, family_mean, precision.astype(np.float32))
            if dist < best_family_dist:
                best_family_dist = dist
                best_family = family
        assert best_family is not None

        centered = (x - self.family_basis[best_family]).astype(np.float32)
        candidates = [concept for concept, fam in self.concept_family.items() if fam == best_family]
        best_concept = None
        best_score = float("inf")
        for concept in candidates:
            precision = 1.0 / (self.concept_var.get(concept, np.ones(self.dim, dtype=np.float32)) + 0.03)
            offset = self.concept_offset[concept]
            plain_score = proto.sq_dist(centered, offset)
            precision_score = weighted_sq_dist(centered, offset, precision.astype(np.float32))
            mix = 1.0
            if self.adaptive_readout:
                count = self.concept_count.get(concept, 0)
                mix = float(np.clip((count - 6.0) / 20.0, 0.0, 1.0))
            score = (1.0 - mix) * plain_score + mix * precision_score
            score -= 0.08 * float(np.dot(centered, offset) / (np.linalg.norm(centered) * np.linalg.norm(offset) + 1e-6))
            if score < best_score:
                best_score = score
                best_concept = concept
        assert best_concept is not None
        return best_family, best_concept


class ProtectedPhaseSplitGrounder(PrecisionSharedOffsetGrounder):
    def __init__(self, dim: int) -> None:
        super().__init__(
            dim=dim,
            family_lr_scale=0.68,
            offset_lr_scale=1.02,
            replay_strength=0.0,
            shrinkage=0.02,
            adaptive_readout=False,
        )
        self.phase1_concepts: set[str] = set()
        self.stable_family_basis: Dict[str, np.ndarray] = {}
        self.stable_concept_offset: Dict[str, np.ndarray] = {}
        self.stable_concept_var: Dict[str, np.ndarray] = {}
        self.direct_proto: Dict[str, np.ndarray] = {}
        self.direct_count: Dict[str, int] = {}

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        super().train(x, family, concept)
        prev = self.direct_proto.get(concept)
        count = self.direct_count.get(concept, 0)
        alpha = min(0.30, 1.0 / float(count + 1))
        self.direct_proto[concept] = self._ema(prev, x, alpha)
        self.direct_count[concept] = count + 1

    def consolidate_phase1(self) -> None:
        self.phase1_concepts = set(self.concept_family.keys())
        self.stable_family_basis = {family: value.copy() for family, value in self.family_basis.items()}
        self.stable_concept_offset = {concept: value.copy() for concept, value in self.concept_offset.items()}
        self.stable_concept_var = {concept: value.copy() for concept, value in self.concept_var.items()}

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        family = min(self.family_basis, key=lambda name: proto.sq_dist(x, self.family_basis[name]))
        candidates = [concept for concept, fam in self.concept_family.items() if fam == family]
        best_concept = None
        best_score = float("inf")
        for concept in candidates:
            if concept in self.phase1_concepts:
                base = self.stable_family_basis.get(family, self.family_basis[family])
                offset = self.stable_concept_offset.get(concept, self.concept_offset[concept])
                precision = 1.0 / (self.stable_concept_var.get(concept, np.ones(self.dim, dtype=np.float32)) + 0.04)
                centered = (x - base).astype(np.float32)
                score = weighted_sq_dist(centered, offset, precision.astype(np.float32))
            else:
                proto_x = self.direct_proto.get(concept, self.family_basis[family] + self.concept_offset[concept])
                score = proto.sq_dist(x, proto_x)
            if score < best_score:
                best_score = score
                best_concept = concept
        assert best_concept is not None
        return family, best_concept


class DualStoreRouteGrounder:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.family_proto: Dict[str, np.ndarray] = {}
        self.family_count: Dict[str, int] = {}
        self.stable_concepts: Dict[str, Dict[str, np.ndarray]] = {}
        self.plastic_concepts: Dict[str, Dict[str, np.ndarray]] = {}
        self.stable_count: Dict[str, Dict[str, int]] = {}
        self.plastic_count: Dict[str, Dict[str, int]] = {}
        self.phase1_families: set[str] = set()

    @staticmethod
    def _ema(prev: np.ndarray | None, x: np.ndarray, count: int) -> np.ndarray:
        if prev is None:
            return x.astype(np.float32).copy()
        alpha = min(0.32, 1.0 / float(count + 1))
        return ((1.0 - alpha) * prev + alpha * x).astype(np.float32)

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        prev_family = self.family_proto.get(family)
        count = self.family_count.get(family, 0)
        self.family_proto[family] = self._ema(prev_family, x, count)
        self.family_count[family] = count + 1

        if family in self.phase1_families and concept not in self.stable_concepts.get(family, {}):
            store = self.plastic_concepts.setdefault(family, {})
            counts = self.plastic_count.setdefault(family, {})
        else:
            store = self.stable_concepts.setdefault(family, {})
            counts = self.stable_count.setdefault(family, {})
        prev = store.get(concept)
        concept_count = counts.get(concept, 0)
        store[concept] = self._ema(prev, x, concept_count)
        counts[concept] = concept_count + 1

    def consolidate_phase1(self) -> None:
        self.phase1_families = set(self.stable_concepts.keys())

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        family = min(self.family_proto, key=lambda name: proto.sq_dist(x, self.family_proto[name]))
        stable = self.stable_concepts.get(family, {})
        plastic = self.plastic_concepts.get(family, {})

        stable_best = None
        stable_score = float("inf")
        for concept, center in stable.items():
            score = proto.sq_dist(x, center)
            if score < stable_score:
                stable_score = score
                stable_best = concept

        plastic_best = None
        plastic_score = float("inf")
        for concept, center in plastic.items():
            score = proto.sq_dist(x, center)
            if score < plastic_score:
                plastic_score = score
                plastic_best = concept

        if plastic_best is not None and plastic_score + 0.015 < stable_score:
            return family, plastic_best
        if stable_best is not None:
            return family, stable_best
        assert plastic_best is not None
        return family, plastic_best


class CrossModalDualStoreGrounder(DualStoreRouteGrounder):
    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        family = min(self.family_proto, key=lambda name: proto.sq_dist(x, self.family_proto[name]))
        stable = self.stable_concepts.get(family, {})
        plastic = self.plastic_concepts.get(family, {})
        visual = x[:8]
        tactile = x[8:]

        stable_best = None
        stable_score = float("inf")
        for concept, center in stable.items():
            score = 0.35 * proto.sq_dist(x, center) + 0.65 * proto.sq_dist(tactile, center[8:])
            if score < stable_score:
                stable_score = score
                stable_best = concept

        plastic_best = None
        plastic_score = float("inf")
        for concept, center in plastic.items():
            score = 0.75 * proto.sq_dist(x, center) + 0.25 * proto.sq_dist(visual, center[:8])
            if score < plastic_score:
                plastic_score = score
                plastic_best = concept

        if plastic_best is not None and plastic_score + 0.010 < stable_score:
            return family, plastic_best
        if stable_best is not None:
            return family, stable_best
        assert plastic_best is not None
        return family, plastic_best


def run_system(system_name: str, seed: int, noise: float, dropout_p: float) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    if system_name == "direct_prototype":
        model = proto.DirectPrototypeLearner(dim=16)
    elif system_name == "shared_offset_grounder":
        model = proto.SharedOffsetGrounder(dim=16)
    elif system_name == "precision_shared_offset":
        model = PrecisionSharedOffsetGrounder(
            dim=16,
            family_lr_scale=0.70,
            offset_lr_scale=1.05,
            replay_strength=0.00,
            shrinkage=0.04,
            adaptive_readout=False,
        )
    elif system_name == "adaptive_precision_shared_offset":
        model = PrecisionSharedOffsetGrounder(
            dim=16,
            family_lr_scale=0.70,
            offset_lr_scale=1.00,
            replay_strength=0.00,
            shrinkage=0.03,
            adaptive_readout=True,
        )
    elif system_name == "adaptive_precision_shared_offset_replay":
        model = PrecisionSharedOffsetGrounder(
            dim=16,
            family_lr_scale=0.66,
            offset_lr_scale=1.05,
            replay_strength=0.03,
            shrinkage=0.02,
            adaptive_readout=True,
        )
    elif system_name == "protected_phase_split":
        model = ProtectedPhaseSplitGrounder(dim=16)
    elif system_name == "dual_store_route":
        model = DualStoreRouteGrounder(dim=16)
    elif system_name == "cross_modal_dual_store":
        model = CrossModalDualStoreGrounder(dim=16)
    else:
        model = PrecisionSharedOffsetGrounder(
            dim=16,
            family_lr_scale=0.60,
            offset_lr_scale=1.10,
            replay_strength=0.04,
            shrinkage=0.03,
            adaptive_readout=False,
        )

    for _ in range(42):
        for family, concepts in proto.PHASE1.items():
            for concept in concepts:
                model.train(proto.sample_continuous_input(rng, concept, noise, dropout_p), family, concept)

    if hasattr(model, "consolidate_phase1"):
        model.consolidate_phase1()

    phase1_eval = proto.evaluate_model(model, proto.PHASE1, repeats=22, rng=rng, noise=noise, dropout_p=dropout_p)

    for _ in range(3):
        for family, concepts in proto.PHASE2.items():
            for concept in concepts:
                model.train(proto.sample_continuous_input(rng, concept, noise, dropout_p), family, concept)

    novel_eval = proto.evaluate_model(model, proto.PHASE2, repeats=24, rng=rng, noise=noise, dropout_p=dropout_p)

    for _ in range(18):
        for family, concepts in proto.PHASE2.items():
            for concept in concepts:
                model.train(proto.sample_continuous_input(rng, concept, noise, dropout_p), family, concept)

    retention_eval = proto.evaluate_model(model, proto.PHASE1, repeats=22, rng=rng, noise=noise, dropout_p=dropout_p)
    overall_eval = proto.evaluate_model(
        model,
        {family: proto.PHASE1[family] + proto.PHASE2[family] for family in proto.FAMILIES},
        repeats=22,
        rng=rng,
        noise=noise,
        dropout_p=dropout_p,
    )

    grounding_score = float(
        (
            phase1_eval["family_accuracy"]
            + novel_eval["family_accuracy"]
            + overall_eval["family_accuracy"]
            + 2.0 * novel_eval["concept_accuracy"]
            + 1.5 * overall_eval["concept_accuracy"]
            + 1.0 * retention_eval["concept_accuracy"]
        )
        / 7.5
    )
    return {
        "phase1_family_accuracy": phase1_eval["family_accuracy"],
        "phase1_concept_accuracy": phase1_eval["concept_accuracy"],
        "novel_family_accuracy": novel_eval["family_accuracy"],
        "novel_concept_accuracy": novel_eval["concept_accuracy"],
        "retention_family_accuracy": retention_eval["family_accuracy"],
        "retention_concept_accuracy": retention_eval["concept_accuracy"],
        "overall_family_accuracy": overall_eval["family_accuracy"],
        "overall_concept_accuracy": overall_eval["concept_accuracy"],
        "grounding_score": grounding_score,
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(rows[0].keys())
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Precision-weighted continuous grounding scan")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/continuous_input_grounding_precision_scan_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    systems = {}
    ranking = []
    names = [
        "direct_prototype",
        "shared_offset_grounder",
        "precision_shared_offset",
        "adaptive_precision_shared_offset",
        "adaptive_precision_shared_offset_replay",
        "precision_shared_offset_replay",
        "protected_phase_split",
        "dual_store_route",
        "cross_modal_dual_store",
    ]
    for system_name in names:
        rows = []
        for offset in range(int(args.num_seeds)):
            rows.append(run_system(system_name, int(args.seed) + offset, float(args.noise), float(args.dropout_p)))
        systems[system_name] = summarize(rows)
        ranking.append({"system": system_name, **systems[system_name]})
        print(
            f"[summary] {system_name} grounding={systems[system_name]['grounding_score']:.4f} "
            f"novel={systems[system_name]['novel_concept_accuracy']:.4f} "
            f"retention={systems[system_name]['retention_concept_accuracy']:.4f}"
        )

    ranking.sort(key=lambda row: row["grounding_score"], reverse=True)
    baseline = systems["direct_prototype"]
    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "runtime_sec": float(time.time() - t0),
        },
        "systems": systems,
        "ranking": ranking,
        "gains_vs_direct": {
            key: {
                "grounding_score_gain": float(row["grounding_score"] - baseline["grounding_score"]),
                "novel_concept_gain": float(row["novel_concept_accuracy"] - baseline["novel_concept_accuracy"]),
                "retention_concept_gain": float(row["retention_concept_accuracy"] - baseline["retention_concept_accuracy"]),
                "overall_concept_gain": float(row["overall_concept_accuracy"] - baseline["overall_concept_accuracy"]),
            }
            for key, row in systems.items()
            if key != "direct_prototype"
        },
        "hypotheses": {
            "H1_some_precision_grounder_beats_direct_on_grounding": bool(
                any(row["grounding_score"] > baseline["grounding_score"] for row in ranking if row["system"] != "direct_prototype")
            ),
            "H2_some_precision_grounder_beats_direct_on_novel_and_retention": bool(
                any(
                    row["novel_concept_accuracy"] > baseline["novel_concept_accuracy"]
                    and row["retention_concept_accuracy"] > baseline["retention_concept_accuracy"]
                    for row in ranking
                    if row["system"] != "direct_prototype"
                )
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["hypotheses"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
