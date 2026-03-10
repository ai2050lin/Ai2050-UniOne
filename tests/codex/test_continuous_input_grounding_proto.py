#!/usr/bin/env python
"""
Task block D:
Continuous-input grounding prototype without token identities.

We compare:
- direct_prototype: memorizes concept prototypes directly
- shared_offset_grounder: learns family bases + concept offsets

The benchmark uses two continuous modalities and evaluates:
- family grounding
- concept identification
- few-shot adaptation to new concepts
- retention after phase-2 updates
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


FAMILIES = ["fruit", "animal", "abstract"]
PHASE1 = {
    "fruit": ["apple", "banana"],
    "animal": ["cat", "dog"],
    "abstract": ["truth", "logic"],
}
PHASE2 = {
    "fruit": ["pear"],
    "animal": ["horse"],
    "abstract": ["memory"],
}


def family_basis() -> Dict[str, np.ndarray]:
    return {
        "fruit": np.array([0.85, 0.82, 0.24, 0.18, 0.08, 0.11, 0.05, 0.07, 0.76, 0.72, 0.19, 0.12, 0.09, 0.06, 0.04, 0.05], dtype=np.float32),
        "animal": np.array([0.22, 0.18, 0.86, 0.88, 0.11, 0.09, 0.08, 0.06, 0.28, 0.26, 0.81, 0.84, 0.10, 0.07, 0.05, 0.04], dtype=np.float32),
        "abstract": np.array([0.14, 0.12, 0.19, 0.17, 0.82, 0.86, 0.28, 0.31, 0.18, 0.15, 0.24, 0.19, 0.88, 0.83, 0.33, 0.30], dtype=np.float32),
    }


def concept_offset() -> Dict[str, np.ndarray]:
    return {
        "apple": np.array([0.08, 0.03, 0.00, 0.01, 0.00, 0.02, 0.00, 0.01, 0.05, 0.02, 0.00, 0.01, 0.00, 0.01, 0.00, 0.00], dtype=np.float32),
        "banana": np.array([0.02, 0.09, 0.00, 0.00, 0.01, 0.00, 0.02, 0.00, 0.01, 0.08, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01], dtype=np.float32),
        "pear": np.array([0.05, 0.05, 0.00, 0.01, 0.00, 0.01, 0.01, 0.00, 0.03, 0.04, 0.00, 0.01, 0.00, 0.01, 0.00, 0.00], dtype=np.float32),
        "cat": np.array([0.00, 0.01, 0.09, 0.05, 0.00, 0.00, 0.01, 0.02, 0.00, 0.01, 0.08, 0.03, 0.01, 0.00, 0.02, 0.00], dtype=np.float32),
        "dog": np.array([0.01, 0.00, 0.05, 0.09, 0.00, 0.01, 0.00, 0.02, 0.01, 0.00, 0.04, 0.08, 0.00, 0.01, 0.01, 0.00], dtype=np.float32),
        "horse": np.array([0.00, 0.01, 0.07, 0.07, 0.00, 0.01, 0.01, 0.00, 0.00, 0.01, 0.06, 0.06, 0.01, 0.00, 0.01, 0.01], dtype=np.float32),
        "truth": np.array([0.01, 0.00, 0.00, 0.01, 0.08, 0.05, 0.03, 0.01, 0.00, 0.01, 0.01, 0.00, 0.09, 0.04, 0.05, 0.01], dtype=np.float32),
        "logic": np.array([0.00, 0.01, 0.01, 0.00, 0.05, 0.09, 0.01, 0.03, 0.01, 0.00, 0.00, 0.01, 0.04, 0.09, 0.01, 0.05], dtype=np.float32),
        "memory": np.array([0.01, 0.00, 0.00, 0.01, 0.06, 0.07, 0.02, 0.02, 0.00, 0.01, 0.01, 0.00, 0.07, 0.06, 0.03, 0.04], dtype=np.float32),
    }


def concept_family(concept: str) -> str:
    for family in FAMILIES:
        if concept in PHASE1.get(family, []) or concept in PHASE2.get(family, []):
            return family
    raise KeyError(concept)


def sample_continuous_input(rng: np.random.Generator, concept: str, noise: float, dropout_p: float) -> np.ndarray:
    family = concept_family(concept)
    base = family_basis()[family] + concept_offset()[concept]
    visual = base[:8] + rng.normal(scale=noise, size=8)
    tactile = base[8:] + rng.normal(scale=noise * 0.8, size=8)
    mask = (rng.random(16) > dropout_p).astype(np.float32)
    return (np.concatenate([visual, tactile], axis=0).astype(np.float32) * mask).astype(np.float32)


def sq_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.square(a - b)))


class DirectPrototypeLearner:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.family_proto: Dict[str, np.ndarray] = {}
        self.family_count: Dict[str, int] = {}
        self.concept_proto: Dict[str, np.ndarray] = {}
        self.concept_count: Dict[str, int] = {}

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        self.family_proto[family] = self._running_mean(self.family_proto.get(family), self.family_count.get(family, 0), x)
        self.family_count[family] = self.family_count.get(family, 0) + 1
        self.concept_proto[concept] = self._running_mean(self.concept_proto.get(concept), self.concept_count.get(concept, 0), x)
        self.concept_count[concept] = self.concept_count.get(concept, 0) + 1

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        family = min(self.family_proto, key=lambda name: sq_dist(x, self.family_proto[name]))
        concept = min(self.concept_proto, key=lambda name: sq_dist(x, self.concept_proto[name]))
        return family, concept

    @staticmethod
    def _running_mean(prev: np.ndarray | None, count: int, x: np.ndarray) -> np.ndarray:
        if prev is None:
            return x.astype(np.float32).copy()
        alpha = 1.0 / float(count + 1)
        return ((1.0 - alpha) * prev + alpha * x).astype(np.float32)


class SharedOffsetGrounder:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.family_basis: Dict[str, np.ndarray] = {}
        self.family_count: Dict[str, int] = {}
        self.concept_offset: Dict[str, np.ndarray] = {}
        self.concept_count: Dict[str, int] = {}
        self.concept_family: Dict[str, str] = {}

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        family_mean = self._running_mean(self.family_basis.get(family), self.family_count.get(family, 0), x)
        self.family_basis[family] = family_mean
        self.family_count[family] = self.family_count.get(family, 0) + 1
        offset = (x - family_mean).astype(np.float32)
        prev_offset = self.concept_offset.get(concept)
        self.concept_offset[concept] = self._running_mean(prev_offset, self.concept_count.get(concept, 0), offset)
        self.concept_count[concept] = self.concept_count.get(concept, 0) + 1
        self.concept_family[concept] = family

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        family = min(self.family_basis, key=lambda name: sq_dist(x, self.family_basis[name]))
        candidates = [concept for concept, fam in self.concept_family.items() if fam == family]
        if not candidates:
            candidates = list(self.concept_family.keys())
        best_concept = None
        best_dist = float("inf")
        for concept in candidates:
            fam = self.concept_family[concept]
            proto = self.family_basis[fam] + self.concept_offset[concept]
            dist = sq_dist(x, proto)
            if dist < best_dist:
                best_dist = dist
                best_concept = concept
        assert best_concept is not None
        return family, best_concept

    @staticmethod
    def _running_mean(prev: np.ndarray | None, count: int, x: np.ndarray) -> np.ndarray:
        if prev is None:
            return x.astype(np.float32).copy()
        alpha = 1.0 / float(count + 1)
        return ((1.0 - alpha) * prev + alpha * x).astype(np.float32)


def evaluate_model(
    model,
    concept_groups: Dict[str, List[str]],
    repeats: int,
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
) -> Dict[str, float]:
    family_ok = 0
    concept_ok = 0
    total = 0
    for _ in range(repeats):
        for family, concepts in concept_groups.items():
            for concept in concepts:
                x = sample_continuous_input(rng, concept, noise, dropout_p)
                pred_family, pred_concept = model.predict(x)
                family_ok += int(pred_family == family)
                concept_ok += int(pred_concept == concept)
                total += 1
    return {
        "family_accuracy": float(family_ok / max(1, total)),
        "concept_accuracy": float(concept_ok / max(1, total)),
    }


def run_system(system_name: str, seed: int, noise: float, dropout_p: float) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    if system_name == "direct_prototype":
        model = DirectPrototypeLearner(dim=16)
    else:
        model = SharedOffsetGrounder(dim=16)

    for _ in range(42):
        for family, concepts in PHASE1.items():
            for concept in concepts:
                model.train(sample_continuous_input(rng, concept, noise, dropout_p), family, concept)

    phase1_eval = evaluate_model(model, PHASE1, repeats=22, rng=rng, noise=noise, dropout_p=dropout_p)

    for _ in range(3):
        for family, concepts in PHASE2.items():
            for concept in concepts:
                model.train(sample_continuous_input(rng, concept, noise, dropout_p), family, concept)

    novel_eval = evaluate_model(model, PHASE2, repeats=24, rng=rng, noise=noise, dropout_p=dropout_p)

    for _ in range(18):
        for family, concepts in PHASE2.items():
            for concept in concepts:
                model.train(sample_continuous_input(rng, concept, noise, dropout_p), family, concept)

    retention_eval = evaluate_model(model, PHASE1, repeats=22, rng=rng, noise=noise, dropout_p=dropout_p)
    overall_eval = evaluate_model(
        model,
        {family: PHASE1[family] + PHASE2[family] for family in FAMILIES},
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
            + 0.5 * retention_eval["concept_accuracy"]
        )
        / 7.0
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
    ap = argparse.ArgumentParser(description="Continuous-input grounding prototype")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/continuous_input_grounding_proto_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    systems = {}
    for system_name in ["direct_prototype", "shared_offset_grounder"]:
        rows = []
        for offset in range(int(args.num_seeds)):
            rows.append(run_system(system_name, int(args.seed) + offset, float(args.noise), float(args.dropout_p)))
        systems[system_name] = summarize(rows)
        print(
            f"[summary] {system_name} grounding={systems[system_name]['grounding_score']:.4f} "
            f"novel={systems[system_name]['novel_concept_accuracy']:.4f} "
            f"retention={systems[system_name]['retention_concept_accuracy']:.4f}"
        )

    baseline = systems["direct_prototype"]
    grounded = systems["shared_offset_grounder"]
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
        "gains": {
            "grounding_score_gain": float(grounded["grounding_score"] - baseline["grounding_score"]),
            "novel_concept_gain": float(grounded["novel_concept_accuracy"] - baseline["novel_concept_accuracy"]),
            "retention_concept_gain": float(grounded["retention_concept_accuracy"] - baseline["retention_concept_accuracy"]),
            "overall_concept_gain": float(grounded["overall_concept_accuracy"] - baseline["overall_concept_accuracy"]),
        },
        "hypotheses": {
            "H1_grounded_beats_direct_on_grounding_score": bool(grounded["grounding_score"] > baseline["grounding_score"]),
            "H2_grounded_beats_direct_on_novel_concepts": bool(grounded["novel_concept_accuracy"] > baseline["novel_concept_accuracy"]),
            "H3_grounded_beats_direct_on_retention": bool(grounded["retention_concept_accuracy"] > baseline["retention_concept_accuracy"]),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
