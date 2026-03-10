#!/usr/bin/env python
"""
Open-world-like continuous grounding stream benchmark.

Compared with the old phase-based grounding prototypes, this benchmark adds:
- continuous stream updates instead of strict train/eval phases
- background drift
- missing modalities
- distractor/noise chunks
- old-concept revisits after novel concept insertion

The goal is not to prove AGI, but to test whether a grounding system can keep
concept structure alive inside a harder continuous environment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import test_continuous_multimodal_grounding_proto as multi


FAMILIES = multi.FAMILIES
PHASE1 = multi.PHASE1
PHASE2 = multi.PHASE2


def concept_family(concept: str) -> str:
    return multi.concept_family(concept)


def sample_stream_input(
    rng: np.random.Generator,
    concept: str,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
) -> np.ndarray:
    x = multi.sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p).astype(np.float32)
    drift = rng.normal(scale=drift_scale, size=x.shape[0]).astype(np.float32)
    return (x + drift).astype(np.float32)


def sample_noise_chunk(
    rng: np.random.Generator,
    dim: int,
    noise_scale: float,
) -> np.ndarray:
    return rng.normal(scale=noise_scale, size=dim).astype(np.float32)


def sq_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.square(a - b)))


class DirectStreamPrototype:
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


class SharedOffsetStreamGrounder:
    def __init__(self) -> None:
        self.family_basis: Dict[str, np.ndarray] = {}
        self.family_count: Dict[str, int] = {}
        self.concept_offset: Dict[str, np.ndarray] = {}
        self.concept_count: Dict[str, int] = {}
        self.concept_family: Dict[str, str] = {}

    @staticmethod
    def _mean(prev: np.ndarray | None, count: int, x: np.ndarray, alpha_override: float | None = None) -> np.ndarray:
        if prev is None:
            return x.astype(np.float32).copy()
        alpha = alpha_override if alpha_override is not None else 1.0 / float(count + 1)
        return ((1.0 - alpha) * prev + alpha * x).astype(np.float32)

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        family_prev = self.family_basis.get(family)
        family_alpha = 0.05 if family_prev is not None else None
        family_mean = self._mean(family_prev, self.family_count.get(family, 0), x, alpha_override=family_alpha)
        self.family_basis[family] = family_mean
        self.family_count[family] = self.family_count.get(family, 0) + 1

        offset = (x - family_mean).astype(np.float32)
        offset_prev = self.concept_offset.get(concept)
        offset_alpha = 0.12 if offset_prev is not None else None
        self.concept_offset[concept] = self._mean(offset_prev, self.concept_count.get(concept, 0), offset, alpha_override=offset_alpha)
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
            proto = self.family_basis[family] + self.concept_offset[concept]
            dist = sq_dist(x, proto)
            if dist < best_dist:
                best_dist = dist
                best_concept = concept
        assert best_concept is not None
        return family, best_concept


def build_stream(seed: int) -> List[Dict[str, str]]:
    rng = np.random.default_rng(seed)
    stream: List[Dict[str, str]] = []

    phase1_concepts = [concept for family in FAMILIES for concept in PHASE1[family]]
    phase2_concepts = [concept for family in FAMILIES for concept in PHASE2[family]]

    for _ in range(36):
        rng.shuffle(phase1_concepts)
        for concept in phase1_concepts:
            stream.append({"kind": "concept", "concept": concept, "family": concept_family(concept), "segment": "phase1"})
        for _ in range(6):
            stream.append({"kind": "noise", "segment": "phase1_noise"})

    for _ in range(14):
        rng.shuffle(phase2_concepts)
        for concept in phase2_concepts:
            stream.append({"kind": "concept", "concept": concept, "family": concept_family(concept), "segment": "novel"})
        revisit = rng.choice(phase1_concepts, size=3, replace=False)
        for concept in revisit.tolist():
            stream.append({"kind": "concept", "concept": concept, "family": concept_family(concept), "segment": "revisit"})
        for _ in range(8):
            stream.append({"kind": "noise", "segment": "novel_noise"})

    return stream


def evaluate_concepts(
    model,
    concepts: List[str],
    repeats: int,
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
) -> Dict[str, float]:
    family_ok = 0
    concept_ok = 0
    total = 0
    for _ in range(repeats):
        for concept in concepts:
            family = concept_family(concept)
            x = sample_stream_input(rng, concept, noise, dropout_p, missing_modality_p, drift_scale)
            pred_family, pred_concept = model.predict(x)
            family_ok += int(pred_family == family)
            concept_ok += int(pred_concept == concept)
            total += 1
    return {
        "family_accuracy": float(family_ok / max(1, total)),
        "concept_accuracy": float(concept_ok / max(1, total)),
    }


def run_system(
    system_name: str,
    seed: int,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    model = DirectStreamPrototype() if system_name == "direct_stream" else SharedOffsetStreamGrounder()
    dim = 24

    stream = build_stream(seed)
    update_count = 0
    for item in stream:
        if item["kind"] == "concept":
            x = sample_stream_input(
                rng,
                item["concept"],
                noise=noise,
                dropout_p=dropout_p,
                missing_modality_p=missing_modality_p,
                drift_scale=drift_scale,
            )
            model.train(x, item["family"], item["concept"])
            update_count += 1
        else:
            _ = sample_noise_chunk(rng, dim=dim, noise_scale=noise * 1.2)

    phase1_concepts = [concept for family in FAMILIES for concept in PHASE1[family]]
    phase2_concepts = [concept for family in FAMILIES for concept in PHASE2[family]]
    all_concepts = phase1_concepts + phase2_concepts

    stable_old = evaluate_concepts(model, phase1_concepts, 24, rng, noise, dropout_p, missing_modality_p, drift_scale)
    novel = evaluate_concepts(model, phase2_concepts, 28, rng, noise, dropout_p, missing_modality_p, drift_scale)
    drifted = evaluate_concepts(model, all_concepts, 24, rng, noise * 1.1, dropout_p, missing_modality_p + 0.05, drift_scale * 1.8)

    closure_score = float(
        (
            1.7 * novel["concept_accuracy"]
            + 1.5 * stable_old["concept_accuracy"]
            + 1.1 * drifted["concept_accuracy"]
            + 1.0 * novel["family_accuracy"]
            + 0.9 * stable_old["family_accuracy"]
            + 0.8 * drifted["family_accuracy"]
        )
        / 7.0
    )

    return {
        "stable_old_family_accuracy": stable_old["family_accuracy"],
        "stable_old_concept_accuracy": stable_old["concept_accuracy"],
        "novel_family_accuracy": novel["family_accuracy"],
        "novel_concept_accuracy": novel["concept_accuracy"],
        "drifted_family_accuracy": drifted["family_accuracy"],
        "drifted_concept_accuracy": drifted["concept_accuracy"],
        "closure_score": closure_score,
        "update_count": float(update_count),
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(rows[0].keys())
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Open-world-like continuous grounding stream benchmark")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.25)
    ap.add_argument("--drift-scale", type=float, default=0.06)
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/open_world_continuous_grounding_stream_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    systems = {}
    for system_name in ["direct_stream", "shared_offset_stream"]:
        rows = []
        for offset in range(int(args.num_seeds)):
            rows.append(
                run_system(
                    system_name=system_name,
                    seed=int(args.seed) + offset,
                    noise=float(args.noise),
                    dropout_p=float(args.dropout_p),
                    missing_modality_p=float(args.missing_modality_p),
                    drift_scale=float(args.drift_scale),
                )
            )
        systems[system_name] = summarize(rows)

    direct = systems["direct_stream"]
    shared = systems["shared_offset_stream"]
    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "missing_modality_p": float(args.missing_modality_p),
            "drift_scale": float(args.drift_scale),
            "runtime_sec": float(time.time() - t0),
        },
        "systems": systems,
        "gains_vs_direct": {
            "stable_old_concept_gain": float(shared["stable_old_concept_accuracy"] - direct["stable_old_concept_accuracy"]),
            "novel_concept_gain": float(shared["novel_concept_accuracy"] - direct["novel_concept_accuracy"]),
            "drifted_concept_gain": float(shared["drifted_concept_accuracy"] - direct["drifted_concept_accuracy"]),
            "closure_score_gain": float(shared["closure_score"] - direct["closure_score"]),
        },
        "hypotheses": {
            "H1_shared_offset_beats_direct_on_open_world_closure": bool(shared["closure_score"] > direct["closure_score"]),
            "H2_shared_offset_beats_direct_on_drifted_concepts": bool(shared["drifted_concept_accuracy"] > direct["drifted_concept_accuracy"]),
            "H3_shared_offset_beats_direct_on_old_concept_stability": bool(shared["stable_old_concept_accuracy"] > direct["stable_old_concept_accuracy"]),
        },
        "project_readout": {
            "summary": "这一版把连续接地从阶段化原型测试推进到连续流环境，开始直接测概念在背景漂移、噪声片段和新旧概念交错出现时能否维持稳定。",
            "next_question": "如果流式 closure 仍然有正增益，下一步就该把动作回路和长期代理状态接进来，而不是继续停在静态原型层。",
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["gains_vs_direct"], ensure_ascii=False, indent=2))
    print(json.dumps(results["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
