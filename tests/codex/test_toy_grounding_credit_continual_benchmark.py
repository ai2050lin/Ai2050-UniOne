#!/usr/bin/env python
"""
Toy benchmark for:
- symbol grounding
- delayed credit assignment
- continual learning

This benchmark is intentionally small, but unlike the earlier version it is
non-trivial:
- grounding learns noisy concept -> family mapping
- delayed query asks at t=2 whether the concept at t=1 belongs to a queried family
- continual learning introduces abstract concepts in phase 2 and measures
  interference on earlier concrete families

Two systems are compared:
- plain_local: grounding updates the encoder, but delayed reward does not
  propagate back into the encoder
- trace_gated_local: delayed reward updates the encoder through a local trace,
  and phase-1 knowledge is stabilized before phase 2
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


FAMILIES = ["fruit", "animal", "abstract"]
CONCEPTS = {
    "fruit": ["apple", "banana"],
    "animal": ["cat", "dog"],
    "abstract": ["truth", "logic"],
}
FAMILY_INDEX = {name: idx for idx, name in enumerate(FAMILIES)}


def one_hot(index: int, size: int) -> np.ndarray:
    row = np.zeros(size, dtype=np.float32)
    row[index] = 1.0
    return row


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - float(np.max(x))
    exp = np.exp(shifted)
    return (exp / (float(np.sum(exp)) + 1e-12)).astype(np.float32)


def concept_centers() -> Dict[str, np.ndarray]:
    # Families overlap on several channels so a tiny learner cannot solve the
    # task by a single near-orthogonal axis.
    return {
        "apple": np.array([0.95, 0.82, 0.33, 0.28, 0.20, 0.06, 0.04, 0.00], dtype=np.float32),
        "banana": np.array([0.88, 0.94, 0.26, 0.22, 0.15, 0.05, 0.03, 0.02], dtype=np.float32),
        "cat": np.array([0.28, 0.24, 0.93, 0.79, 0.18, 0.08, 0.05, 0.01], dtype=np.float32),
        "dog": np.array([0.24, 0.30, 0.81, 0.92, 0.17, 0.06, 0.04, 0.02], dtype=np.float32),
        "truth": np.array([0.22, 0.12, 0.26, 0.17, 0.86, 0.78, 0.36, 0.24], dtype=np.float32),
        "logic": np.array([0.18, 0.10, 0.21, 0.24, 0.80, 0.89, 0.29, 0.31], dtype=np.float32),
    }


def concept_family(concept: str) -> str:
    for family, words in CONCEPTS.items():
        if concept in words:
            return family
    raise KeyError(concept)


def sample_grounding_input(rng: np.random.Generator, concept: str, noise: float, dropout_p: float) -> np.ndarray:
    center = concept_centers()[concept]
    x = center + rng.normal(scale=noise, size=center.shape)
    mask = (rng.random(center.shape[0]) > dropout_p).astype(np.float32)
    return (x * mask).astype(np.float32)


def sample_delayed_episode(
    rng: np.random.Generator,
    concept: str,
    noise: float,
    dropout_p: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    x1 = sample_grounding_input(rng, concept, noise, dropout_p)
    true_family = concept_family(concept)
    asked_family = FAMILIES[int(rng.integers(0, len(FAMILIES)))]
    q = one_hot(FAMILY_INDEX[asked_family], len(FAMILIES))
    y = 1.0 if asked_family == true_family else 0.0
    return x1, q, y


@dataclass
class AssociativeLearner:
    input_dim: int
    hidden_dim: int
    family_dim: int
    lr_enc: float
    lr_family: float
    lr_query: float
    use_trace: bool
    stability: float
    seed: int

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w_enc = rng.normal(scale=0.20, size=(self.input_dim, self.hidden_dim)).astype(np.float32)
        self.w_family = rng.normal(scale=0.20, size=(self.hidden_dim, self.family_dim)).astype(np.float32)
        self.w_query = rng.normal(scale=0.20, size=(self.family_dim, self.hidden_dim)).astype(np.float32)
        self.b_family = np.zeros(self.family_dim, dtype=np.float32)
        self.b_query = 0.0
        self.enc_ref = self.w_enc.copy()
        self.enc_importance = np.zeros_like(self.w_enc)

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pre = x @ self.w_enc
        h = np.tanh(pre).astype(np.float32)
        return pre.astype(np.float32), h

    def family_predict(self, x: np.ndarray) -> np.ndarray:
        _pre, h = self.encode(x)
        logits = h @ self.w_family + self.b_family
        return softmax(logits)

    def delayed_predict(self, x: np.ndarray, q: np.ndarray) -> float:
        _pre, h = self.encode(x)
        q_proj = q @ self.w_query
        return sigmoid(float(np.dot(h, q_proj) + self.b_query))

    def apply_encoder_update(self, update: np.ndarray) -> None:
        if self.stability > 0.0:
            update = update - self.stability * self.enc_importance * (self.w_enc - self.enc_ref)
        self.w_enc += (self.lr_enc * update).astype(np.float32)

    def grounding_step(self, x: np.ndarray, y: np.ndarray) -> None:
        _pre, h = self.encode(x)
        probs = softmax(h @ self.w_family + self.b_family)
        err = (y - probs).astype(np.float32)
        grad_family = np.outer(h, err).astype(np.float32)
        grad_h = (self.w_family @ err).astype(np.float32)
        grad_pre = (grad_h * (1.0 - np.square(h))).astype(np.float32)
        grad_enc = np.outer(x, grad_pre).astype(np.float32)
        self.apply_encoder_update(grad_enc)
        self.w_family += (self.lr_family * grad_family).astype(np.float32)
        self.b_family += (self.lr_family * err).astype(np.float32)

    def delayed_step(self, x: np.ndarray, q: np.ndarray, y: float) -> None:
        _pre, h = self.encode(x)
        q_proj = (q @ self.w_query).astype(np.float32)
        pred = sigmoid(float(np.dot(h, q_proj) + self.b_query))
        err = float(y - pred)
        grad_query = np.outer(q, err * h).astype(np.float32)
        self.w_query += (self.lr_query * grad_query).astype(np.float32)
        self.b_query += self.lr_query * err
        if self.use_trace:
            grad_h = (err * q_proj).astype(np.float32)
            grad_pre = (grad_h * (1.0 - np.square(h))).astype(np.float32)
            grad_enc = np.outer(x, grad_pre).astype(np.float32)
            self.apply_encoder_update(grad_enc)

    def consolidate(self, families: List[str], rng: np.random.Generator, noise: float, dropout_p: float, repeats: int) -> None:
        importance = np.zeros_like(self.w_enc)
        count = 0
        for _ in range(repeats):
            for family in families:
                for concept in CONCEPTS[family]:
                    x = sample_grounding_input(rng, concept, noise, dropout_p)
                    y = one_hot(FAMILY_INDEX[family], self.family_dim)
                    _pre, h = self.encode(x)
                    probs = softmax(h @ self.w_family + self.b_family)
                    err = (y - probs).astype(np.float32)
                    grad_h = (self.w_family @ err).astype(np.float32)
                    grad_pre = (grad_h * (1.0 - np.square(h))).astype(np.float32)
                    importance += np.abs(np.outer(x, grad_pre)).astype(np.float32)
                    count += 1
        if count > 0:
            importance /= float(count)
        mean_importance = float(np.mean(importance)) + 1e-6
        self.enc_ref = self.w_enc.copy()
        self.enc_importance = (importance / mean_importance).astype(np.float32)


def grounding_dataset(
    families: List[str],
    repeats: int,
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    rows = []
    for _ in range(repeats):
        for family in families:
            for concept in CONCEPTS[family]:
                rows.append(
                    (
                        sample_grounding_input(rng, concept, noise, dropout_p),
                        one_hot(FAMILY_INDEX[family], len(FAMILIES)),
                    )
                )
    rng.shuffle(rows)
    return rows


def delayed_dataset(
    families: List[str],
    repeats: int,
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    rows = []
    for _ in range(repeats):
        for family in families:
            for concept in CONCEPTS[family]:
                rows.append(sample_delayed_episode(rng, concept, noise, dropout_p))
    rng.shuffle(rows)
    return rows


def evaluate_grounding(model: AssociativeLearner, families: List[str], rng: np.random.Generator, noise: float, dropout_p: float, n: int) -> float:
    correct = 0
    total = 0
    for _ in range(n):
        for family in families:
            for concept in CONCEPTS[family]:
                x = sample_grounding_input(rng, concept, noise, dropout_p)
                pred = int(np.argmax(model.family_predict(x)))
                if pred == FAMILY_INDEX[family]:
                    correct += 1
                total += 1
    return float(correct / max(1, total))


def evaluate_delayed(model: AssociativeLearner, families: List[str], rng: np.random.Generator, noise: float, dropout_p: float, n: int) -> float:
    correct = 0
    total = 0
    for _ in range(n):
        for family in families:
            for concept in CONCEPTS[family]:
                for asked_family in FAMILIES:
                    x = sample_grounding_input(rng, concept, noise, dropout_p)
                    q = one_hot(FAMILY_INDEX[asked_family], len(FAMILIES))
                    y = 1.0 if asked_family == family else 0.0
                    pred = model.delayed_predict(x, q)
                    ok = int(pred >= 0.5) == int(y >= 0.5)
                    correct += int(ok)
                    total += 1
    return float(correct / max(1, total))


def run_system(use_trace: bool, seed: int, noise: float, dropout_p: float) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    model = AssociativeLearner(
        input_dim=8,
        hidden_dim=4,
        family_dim=len(FAMILIES),
        lr_enc=0.050,
        lr_family=0.140,
        lr_query=0.120,
        use_trace=use_trace,
        stability=0.18 if use_trace else 0.0,
        seed=seed,
    )

    phase1_families = ["fruit", "animal"]
    phase2_families = ["abstract"]
    all_families = ["fruit", "animal", "abstract"]
    phase1_ground_memory = grounding_dataset(phase1_families, repeats=6, rng=rng, noise=noise, dropout_p=dropout_p)
    phase1_delayed_memory = delayed_dataset(phase1_families, repeats=8, rng=rng, noise=noise, dropout_p=dropout_p)

    for x, y in grounding_dataset(phase1_families, repeats=18, rng=rng, noise=noise, dropout_p=dropout_p):
        model.grounding_step(x, y)
    grounding_phase1 = evaluate_grounding(model, phase1_families, rng, noise, dropout_p, n=24)

    for x, q, y in delayed_dataset(phase1_families, repeats=34, rng=rng, noise=noise, dropout_p=dropout_p):
        model.delayed_step(x, q, y)
    delayed_phase1 = evaluate_delayed(model, phase1_families, rng, noise, dropout_p, n=24)

    retention_before_phase2 = evaluate_grounding(model, phase1_families, rng, noise, dropout_p, n=24)

    if use_trace:
        model.consolidate(phase1_families, rng, noise, dropout_p, repeats=12)

    phase2_ground_rows = grounding_dataset(phase2_families, repeats=26, rng=rng, noise=noise, dropout_p=dropout_p)
    for idx, (x, y) in enumerate(phase2_ground_rows):
        model.grounding_step(x, y)
        if use_trace and idx % 2 == 0:
            mx, my = phase1_ground_memory[idx % len(phase1_ground_memory)]
            model.grounding_step(mx, my)
    grounding_phase2 = evaluate_grounding(model, phase2_families, rng, noise, dropout_p, n=24)

    phase2_delayed_rows = delayed_dataset(phase2_families, repeats=42, rng=rng, noise=noise, dropout_p=dropout_p)
    for idx, (x, q, y) in enumerate(phase2_delayed_rows):
        model.delayed_step(x, q, y)
        if use_trace and idx % 2 == 0:
            mx, mq, my = phase1_delayed_memory[idx % len(phase1_delayed_memory)]
            model.delayed_step(mx, mq, my)
    delayed_phase2 = evaluate_delayed(model, phase2_families, rng, noise, dropout_p, n=24)

    retention_after_phase2 = evaluate_grounding(model, phase1_families, rng, noise, dropout_p, n=24)
    overall_grounding = evaluate_grounding(model, all_families, rng, noise, dropout_p, n=24)
    overall_delayed = evaluate_delayed(model, all_families, rng, noise, dropout_p, n=24)

    return {
        "grounding_phase1_accuracy": grounding_phase1,
        "delayed_phase1_accuracy": delayed_phase1,
        "retention_before_phase2": retention_before_phase2,
        "grounding_phase2_accuracy": grounding_phase2,
        "delayed_phase2_accuracy": delayed_phase2,
        "retention_after_phase2": retention_after_phase2,
        "retention_drop": float(retention_before_phase2 - retention_after_phase2),
        "overall_grounding_accuracy": overall_grounding,
        "overall_delayed_accuracy": overall_delayed,
    }


def summarize_runs(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = rows[0].keys()
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
        }
        for key in keys
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Toy benchmark for grounding, credit assignment, and continual learning")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.46)
    ap.add_argument("--dropout-p", type=float, default=0.20)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/toy_grounding_credit_continual_benchmark_20260308.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    plain_runs = []
    trace_runs = []
    for offset in range(int(args.num_seeds)):
        seed = int(args.seed) + offset
        plain_runs.append(run_system(use_trace=False, seed=seed, noise=float(args.noise), dropout_p=float(args.dropout_p)))
        trace_runs.append(run_system(use_trace=True, seed=seed, noise=float(args.noise), dropout_p=float(args.dropout_p)))

    plain = summarize_runs(plain_runs)
    trace = summarize_runs(trace_runs)
    improvements = {
        key: float(trace[key]["mean"] - plain[key]["mean"])
        for key in plain.keys()
        if key != "retention_drop"
    }
    improvements["retention_drop_reduction"] = float(plain["retention_drop"]["mean"] - trace["retention_drop"]["mean"])

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "runtime_sec": float(time.time() - t0),
        },
        "systems": {
            "plain_local": plain,
            "trace_gated_local": trace,
        },
        "raw_runs": {
            "plain_local": plain_runs,
            "trace_gated_local": trace_runs,
        },
        "improvements": improvements,
        "hypotheses": {
            "H1_trace_improves_delayed_credit": bool(trace["overall_delayed_accuracy"]["mean"] > plain["overall_delayed_accuracy"]["mean"]),
            "H2_trace_reduces_retention_drop": bool(trace["retention_drop"]["mean"] < plain["retention_drop"]["mean"]),
            "H3_grounding_remains_nontrivial": bool(0.45 < trace["overall_grounding_accuracy"]["mean"] < 0.95),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["systems"], ensure_ascii=False, indent=2))
    print(json.dumps(results["improvements"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
