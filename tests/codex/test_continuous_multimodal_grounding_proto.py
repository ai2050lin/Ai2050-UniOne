#!/usr/bin/env python
"""
Task block D:
continuous multimodal grounding prototype.

This extends the old continuous-input grounding proto from two modalities to
three modalities:
- visual
- tactile
- language-like continuous cue

The benchmark measures:
- novel concept gain
- retention gain
- overall grounding gain
- cross-modal consistency
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import test_continuous_input_grounding_proto as proto


FAMILIES = proto.FAMILIES
PHASE1 = proto.PHASE1
PHASE2 = proto.PHASE2


def lang_family_basis() -> Dict[str, np.ndarray]:
    return {
        "fruit": np.array([0.72, 0.68, 0.18, 0.15, 0.10, 0.12, 0.08, 0.06], dtype=np.float32),
        "animal": np.array([0.20, 0.16, 0.74, 0.79, 0.08, 0.07, 0.09, 0.05], dtype=np.float32),
        "abstract": np.array([0.10, 0.12, 0.20, 0.18, 0.76, 0.78, 0.24, 0.26], dtype=np.float32),
    }


def lang_concept_offset() -> Dict[str, np.ndarray]:
    return {
        "apple": np.array([0.06, 0.03, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01], dtype=np.float32),
        "banana": np.array([0.02, 0.07, 0.00, 0.00, 0.00, 0.01, 0.01, 0.00], dtype=np.float32),
        "pear": np.array([0.04, 0.04, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01], dtype=np.float32),
        "cat": np.array([0.00, 0.01, 0.06, 0.03, 0.00, 0.00, 0.01, 0.00], dtype=np.float32),
        "dog": np.array([0.01, 0.00, 0.03, 0.06, 0.00, 0.01, 0.00, 0.00], dtype=np.float32),
        "horse": np.array([0.00, 0.00, 0.05, 0.05, 0.00, 0.00, 0.01, 0.01], dtype=np.float32),
        "truth": np.array([0.00, 0.01, 0.00, 0.00, 0.06, 0.04, 0.03, 0.01], dtype=np.float32),
        "logic": np.array([0.01, 0.00, 0.00, 0.01, 0.04, 0.06, 0.01, 0.03], dtype=np.float32),
        "memory": np.array([0.00, 0.01, 0.01, 0.00, 0.05, 0.05, 0.02, 0.03], dtype=np.float32),
    }


def concept_family(concept: str) -> str:
    return proto.concept_family(concept)


def sample_multimodal_input(rng: np.random.Generator, concept: str, noise: float, dropout_p: float, missing_modality_p: float) -> np.ndarray:
    family = concept_family(concept)
    base16 = proto.family_basis()[family] + proto.concept_offset()[concept]
    visual = base16[:8] + rng.normal(scale=noise, size=8)
    tactile = base16[8:] + rng.normal(scale=noise * 0.8, size=8)
    language = lang_family_basis()[family] + lang_concept_offset()[concept] + rng.normal(scale=noise * 0.6, size=8)

    visual = visual.astype(np.float32)
    tactile = tactile.astype(np.float32)
    language = language.astype(np.float32)

    if rng.random() < missing_modality_p:
        visual *= 0.0
    if rng.random() < missing_modality_p:
        tactile *= 0.0
    if rng.random() < missing_modality_p:
        language *= 0.0

    x = np.concatenate([visual, tactile, language], axis=0).astype(np.float32)
    mask = (rng.random(x.shape[0]) > dropout_p).astype(np.float32)
    return (x * mask).astype(np.float32)


def sq_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.square(a - b)))


class DirectMultimodalPrototype:
    def __init__(self) -> None:
        self.family_proto: Dict[str, np.ndarray] = {}
        self.family_count: Dict[str, int] = {}
        self.concept_proto: Dict[str, np.ndarray] = {}
        self.concept_count: Dict[str, int] = {}

    @staticmethod
    def _mean(prev: np.ndarray | None, count: int, x: np.ndarray) -> np.ndarray:
        if prev is None:
            return x.astype(np.float32).copy()
        alpha = 1.0 / float(count + 1)
        return ((1.0 - alpha) * prev + alpha * x).astype(np.float32)

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        self.family_proto[family] = self._mean(self.family_proto.get(family), self.family_count.get(family, 0), x)
        self.family_count[family] = self.family_count.get(family, 0) + 1
        self.concept_proto[concept] = self._mean(self.concept_proto.get(concept), self.concept_count.get(concept, 0), x)
        self.concept_count[concept] = self.concept_count.get(concept, 0) + 1

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        family = min(self.family_proto, key=lambda name: sq_dist(x, self.family_proto[name]))
        concept = min(self.concept_proto, key=lambda name: sq_dist(x, self.concept_proto[name]))
        return family, concept


class SharedOffsetMultimodalGrounder:
    def __init__(self) -> None:
        self.family_basis: Dict[str, np.ndarray] = {}
        self.family_count: Dict[str, int] = {}
        self.concept_offset: Dict[str, np.ndarray] = {}
        self.concept_count: Dict[str, int] = {}
        self.concept_family: Dict[str, str] = {}

    @staticmethod
    def _mean(prev: np.ndarray | None, count: int, x: np.ndarray) -> np.ndarray:
        if prev is None:
            return x.astype(np.float32).copy()
        alpha = 1.0 / float(count + 1)
        return ((1.0 - alpha) * prev + alpha * x).astype(np.float32)

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        family_mean = self._mean(self.family_basis.get(family), self.family_count.get(family, 0), x)
        self.family_basis[family] = family_mean
        self.family_count[family] = self.family_count.get(family, 0) + 1
        offset = (x - family_mean).astype(np.float32)
        self.concept_offset[concept] = self._mean(self.concept_offset.get(concept), self.concept_count.get(concept, 0), offset)
        self.concept_count[concept] = self.concept_count.get(concept, 0) + 1
        self.concept_family[concept] = family

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        family = min(self.family_basis, key=lambda name: sq_dist(x, self.family_basis[name]))
        candidates = [concept for concept, fam in self.concept_family.items() if fam == family]
        best_concept = None
        best_dist = float("inf")
        for concept in candidates:
            proto_x = self.family_basis[family] + self.concept_offset[concept]
            dist = sq_dist(x, proto_x)
            if dist < best_dist:
                best_dist = dist
                best_concept = concept
        assert best_concept is not None
        return family, best_concept


def evaluate_model(
    model,
    concept_groups: Dict[str, List[str]],
    repeats: int,
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
) -> Dict[str, float]:
    family_ok = 0
    concept_ok = 0
    total = 0
    for _ in range(repeats):
        for family, concepts in concept_groups.items():
            for concept in concepts:
                x = sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p)
                pred_family, pred_concept = model.predict(x)
                family_ok += int(pred_family == family)
                concept_ok += int(pred_concept == concept)
                total += 1
    return {
        "family_accuracy": float(family_ok / max(1, total)),
        "concept_accuracy": float(concept_ok / max(1, total)),
    }


def crossmodal_consistency(
    model,
    concepts: List[str],
    repeats: int,
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
) -> float:
    ok = 0
    total = 0
    for _ in range(repeats):
        for concept in concepts:
            family = concept_family(concept)
            full = sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p=0.0)
            visual_only = full.copy()
            visual_only[8:] = 0.0
            lang_only = full.copy()
            lang_only[:16] = 0.0
            fam_v, con_v = model.predict(visual_only)
            fam_l, con_l = model.predict(lang_only)
            ok += int(fam_v == family and fam_l == family and con_v == con_l == concept)
            total += 1
    return float(ok / max(1, total))


def run_system(system_name: str, seed: int, noise: float, dropout_p: float, missing_modality_p: float) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    if system_name == "direct_multimodal":
        model = DirectMultimodalPrototype()
    else:
        model = SharedOffsetMultimodalGrounder()

    for _ in range(42):
        for family, concepts in PHASE1.items():
            for concept in concepts:
                model.train(sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p), family, concept)

    phase1_eval = evaluate_model(model, PHASE1, repeats=22, rng=rng, noise=noise, dropout_p=dropout_p, missing_modality_p=missing_modality_p)

    for _ in range(3):
        for family, concepts in PHASE2.items():
            for concept in concepts:
                model.train(sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p), family, concept)

    novel_eval = evaluate_model(model, PHASE2, repeats=24, rng=rng, noise=noise, dropout_p=dropout_p, missing_modality_p=missing_modality_p)

    for _ in range(18):
        for family, concepts in PHASE2.items():
            for concept in concepts:
                model.train(sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p), family, concept)

    retention_eval = evaluate_model(model, PHASE1, repeats=22, rng=rng, noise=noise, dropout_p=dropout_p, missing_modality_p=missing_modality_p)
    overall_eval = evaluate_model(
        model,
        {family: PHASE1[family] + PHASE2[family] for family in FAMILIES},
        repeats=22,
        rng=rng,
        noise=noise,
        dropout_p=dropout_p,
        missing_modality_p=missing_modality_p,
    )
    consistency = crossmodal_consistency(
        model,
        [concept for family in FAMILIES for concept in PHASE1[family] + PHASE2[family]],
        repeats=10,
        rng=rng,
        noise=noise,
        dropout_p=dropout_p,
    )

    grounding_score = float(
        (
            phase1_eval["family_accuracy"]
            + novel_eval["family_accuracy"]
            + overall_eval["family_accuracy"]
            + 1.8 * novel_eval["concept_accuracy"]
            + 1.3 * retention_eval["concept_accuracy"]
            + 1.5 * overall_eval["concept_accuracy"]
            + 1.0 * consistency
        )
        / 8.6
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
        "crossmodal_consistency": consistency,
        "grounding_score": grounding_score,
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(rows[0].keys())
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Continuous multimodal grounding prototype")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.22)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/continuous_multimodal_grounding_proto_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    systems = {}
    for system_name in ["direct_multimodal", "shared_offset_multimodal"]:
        rows = []
        for offset in range(int(args.num_seeds)):
            rows.append(
                run_system(
                    system_name,
                    seed=int(args.seed) + offset,
                    noise=float(args.noise),
                    dropout_p=float(args.dropout_p),
                    missing_modality_p=float(args.missing_modality_p),
                )
            )
        systems[system_name] = summarize(rows)

    direct = systems["direct_multimodal"]
    shared = systems["shared_offset_multimodal"]
    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "missing_modality_p": float(args.missing_modality_p),
            "runtime_sec": float(time.time() - t0),
        },
        "systems": systems,
        "gains_vs_direct": {
            "novel_concept_gain": float(shared["novel_concept_accuracy"] - direct["novel_concept_accuracy"]),
            "retention_concept_gain": float(shared["retention_concept_accuracy"] - direct["retention_concept_accuracy"]),
            "overall_concept_gain": float(shared["overall_concept_accuracy"] - direct["overall_concept_accuracy"]),
            "crossmodal_consistency_gain": float(shared["crossmodal_consistency"] - direct["crossmodal_consistency"]),
            "grounding_score_gain": float(shared["grounding_score"] - direct["grounding_score"]),
        },
        "hypotheses": {
            "H1_shared_offset_beats_direct_on_grounding": bool(shared["grounding_score"] > direct["grounding_score"]),
            "H2_shared_offset_beats_direct_on_consistency": bool(shared["crossmodal_consistency"] > direct["crossmodal_consistency"]),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["gains_vs_direct"], ensure_ascii=False, indent=2))
    print(json.dumps(results["hypotheses"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
