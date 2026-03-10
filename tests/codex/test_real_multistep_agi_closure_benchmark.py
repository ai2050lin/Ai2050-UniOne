#!/usr/bin/env python
"""
Real-ish multi-step benchmark for AGI closure proxies.

Compared with the earlier toy benchmark, this task is closer to a genuine
sequence-control setting:
- step 0: ground a noisy concept and choose the correct tool
- step 1: carry concept state through recurrence and choose the correct route
- step 2: use only recurrent memory plus a generic state cue to choose the
  correct final act

All supervision is applied at the end of the episode. This makes the benchmark
multi-step and delayed, while still staying lightweight enough for repeated
local experiments.

Two systems are compared:
- plain_local: delayed supervision updates output heads only
- trace_gated_local: delayed supervision also updates encoder/recurrent weights
  via local eligibility traces, plus stabilization and replay in phase 2
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
TOOLS = ["crate", "food", "proof"]
ROUTES = ["pantry", "yard", "library"]
FINALS = ["store", "feed", "verify"]
FAMILY_TO_ACTIONS = {
    "fruit": {"tool": "crate", "route": "pantry", "final": "store"},
    "animal": {"tool": "food", "route": "yard", "final": "feed"},
    "abstract": {"tool": "proof", "route": "library", "final": "verify"},
}
TOOL_INDEX = {name: idx for idx, name in enumerate(TOOLS)}
ROUTE_INDEX = {name: idx for idx, name in enumerate(ROUTES)}
FINAL_INDEX = {name: idx for idx, name in enumerate(FINALS)}


def one_hot(index: int, size: int) -> np.ndarray:
    row = np.zeros(size, dtype=np.float32)
    row[index] = 1.0
    return row


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - float(np.max(x))
    exp = np.exp(shifted)
    return (exp / (float(np.sum(exp)) + 1e-12)).astype(np.float32)


def concept_centers() -> Dict[str, np.ndarray]:
    return {
        "apple": np.array([0.94, 0.86, 0.31, 0.24, 0.11, 0.07, 0.06, 0.04, 0.12, 0.08], dtype=np.float32),
        "banana": np.array([0.89, 0.91, 0.25, 0.18, 0.10, 0.08, 0.05, 0.05, 0.13, 0.10], dtype=np.float32),
        "cat": np.array([0.30, 0.22, 0.92, 0.86, 0.19, 0.09, 0.05, 0.03, 0.11, 0.06], dtype=np.float32),
        "dog": np.array([0.24, 0.26, 0.86, 0.95, 0.17, 0.10, 0.06, 0.02, 0.09, 0.05], dtype=np.float32),
        "truth": np.array([0.18, 0.11, 0.24, 0.21, 0.86, 0.82, 0.33, 0.27, 0.58, 0.63], dtype=np.float32),
        "logic": np.array([0.16, 0.10, 0.19, 0.23, 0.81, 0.90, 0.29, 0.31, 0.62, 0.59], dtype=np.float32),
    }


def concept_family(concept: str) -> str:
    for family, words in CONCEPTS.items():
        if concept in words:
            return family
    raise KeyError(concept)


def sample_concept_state(rng: np.random.Generator, concept: str, noise: float, dropout_p: float) -> np.ndarray:
    center = concept_centers()[concept]
    x = center + rng.normal(scale=noise, size=center.shape)
    mask = (rng.random(center.shape[0]) > dropout_p).astype(np.float32)
    stage = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return np.concatenate([(x * mask).astype(np.float32), stage], axis=0)


def sample_transition_state(rng: np.random.Generator, step: int, noise: float) -> np.ndarray:
    if step == 1:
        base = np.array([0.14, 0.20, 0.15, 0.18, 0.12, 0.10, 0.06, 0.08, 0.05, 0.04], dtype=np.float32)
        stage = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        base = np.array([0.12, 0.17, 0.13, 0.14, 0.09, 0.11, 0.05, 0.06, 0.03, 0.07], dtype=np.float32)
        stage = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    x = base + rng.normal(scale=noise * 0.35, size=base.shape)
    return np.concatenate([x.astype(np.float32), stage], axis=0)


@dataclass
class MultiStepLearner:
    input_dim: int
    hidden_dim: int
    lr_head: float
    lr_enc: float
    lr_rec: float
    use_trace: bool
    stability: float
    seed: int

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w_enc = rng.normal(scale=0.18, size=(self.input_dim, self.hidden_dim)).astype(np.float32)
        self.w_rec = rng.normal(scale=0.16, size=(self.hidden_dim, self.hidden_dim)).astype(np.float32)
        self.w_tool = rng.normal(scale=0.22, size=(self.hidden_dim, len(TOOLS))).astype(np.float32)
        self.w_route = rng.normal(scale=0.22, size=(self.hidden_dim, len(ROUTES))).astype(np.float32)
        self.w_final = rng.normal(scale=0.22, size=(self.hidden_dim, len(FINALS))).astype(np.float32)
        self.b_tool = np.zeros(len(TOOLS), dtype=np.float32)
        self.b_route = np.zeros(len(ROUTES), dtype=np.float32)
        self.b_final = np.zeros(len(FINALS), dtype=np.float32)
        self.enc_ref = self.w_enc.copy()
        self.rec_ref = self.w_rec.copy()
        self.enc_importance = np.zeros_like(self.w_enc)
        self.rec_importance = np.zeros_like(self.w_rec)

    def step_hidden(self, x: np.ndarray, prev_h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pre = x @ self.w_enc + prev_h @ self.w_rec
        h = np.tanh(pre).astype(np.float32)
        return pre.astype(np.float32), h

    def _apply_regularized_update(self, weights: np.ndarray, update: np.ndarray, ref: np.ndarray, importance: np.ndarray, lr: float) -> np.ndarray:
        adj = update.astype(np.float32)
        if self.stability > 0.0:
            adj = adj - self.stability * importance * (weights - ref)
        return weights + lr * adj

    def train_episode(self, states: List[np.ndarray], targets: Dict[str, int]) -> None:
        prev_h = np.zeros(self.hidden_dim, dtype=np.float32)
        traces = []
        head_specs = [
            ("tool", self.w_tool, self.b_tool, len(TOOLS)),
            ("route", self.w_route, self.b_route, len(ROUTES)),
            ("final", self.w_final, self.b_final, len(FINALS)),
        ]

        for state, (head_name, head_w, head_b, head_dim) in zip(states, head_specs):
            pre, h = self.step_hidden(state, prev_h)
            probs = softmax(h @ head_w + head_b)
            target_vec = one_hot(int(targets[head_name]), head_dim)
            err = (target_vec - probs).astype(np.float32)
            traces.append(
                {
                    "head": head_name,
                    "state": state,
                    "prev_h": prev_h.copy(),
                    "h": h,
                    "pre": pre,
                    "err": err,
                }
            )
            prev_h = h

        for trace in traces:
            head_name = trace["head"]
            h = trace["h"]
            err = trace["err"]
            grad_head = np.outer(h, err).astype(np.float32)
            grad_b = err.astype(np.float32)
            if head_name == "tool":
                self.w_tool += self.lr_head * grad_head
                self.b_tool += self.lr_head * grad_b
                head_w = self.w_tool
            elif head_name == "route":
                self.w_route += self.lr_head * grad_head
                self.b_route += self.lr_head * grad_b
                head_w = self.w_route
            else:
                self.w_final += self.lr_head * grad_head
                self.b_final += self.lr_head * grad_b
                head_w = self.w_final

            if self.use_trace:
                grad_h = (head_w @ err).astype(np.float32)
                grad_pre = (grad_h * (1.0 - np.square(h))).astype(np.float32)
                grad_enc = np.outer(trace["state"], grad_pre).astype(np.float32)
                grad_rec = np.outer(trace["prev_h"], grad_pre).astype(np.float32)
                self.w_enc = self._apply_regularized_update(self.w_enc, grad_enc, self.enc_ref, self.enc_importance, self.lr_enc)
                self.w_rec = self._apply_regularized_update(self.w_rec, grad_rec, self.rec_ref, self.rec_importance, self.lr_rec)

    def predict_episode(self, states: List[np.ndarray]) -> Dict[str, int]:
        prev_h = np.zeros(self.hidden_dim, dtype=np.float32)
        outputs: Dict[str, int] = {}
        head_specs = [
            ("tool", self.w_tool, self.b_tool),
            ("route", self.w_route, self.b_route),
            ("final", self.w_final, self.b_final),
        ]
        for state, (head_name, head_w, head_b) in zip(states, head_specs):
            _pre, h = self.step_hidden(state, prev_h)
            probs = softmax(h @ head_w + head_b)
            outputs[head_name] = int(np.argmax(probs))
            prev_h = h
        return outputs

    def consolidate(self, episodes: List[Tuple[List[np.ndarray], Dict[str, int]]]) -> None:
        enc_imp = np.zeros_like(self.w_enc)
        rec_imp = np.zeros_like(self.w_rec)
        count = 0
        for states, targets in episodes:
            prev_h = np.zeros(self.hidden_dim, dtype=np.float32)
            head_specs = [
                ("tool", self.w_tool, self.b_tool, len(TOOLS)),
                ("route", self.w_route, self.b_route, len(ROUTES)),
                ("final", self.w_final, self.b_final, len(FINALS)),
            ]
            for state, (head_name, head_w, head_b, head_dim) in zip(states, head_specs):
                pre, h = self.step_hidden(state, prev_h)
                probs = softmax(h @ head_w + head_b)
                target_vec = one_hot(int(targets[head_name]), head_dim)
                err = (target_vec - probs).astype(np.float32)
                grad_h = (head_w @ err).astype(np.float32)
                grad_pre = (grad_h * (1.0 - np.square(h))).astype(np.float32)
                enc_imp += np.abs(np.outer(state, grad_pre)).astype(np.float32)
                rec_imp += np.abs(np.outer(prev_h, grad_pre)).astype(np.float32)
                prev_h = h
                count += 1
        if count > 0:
            enc_imp /= float(count)
            rec_imp /= float(count)
        self.enc_ref = self.w_enc.copy()
        self.rec_ref = self.w_rec.copy()
        self.enc_importance = enc_imp / (float(np.mean(enc_imp)) + 1e-6)
        self.rec_importance = rec_imp / (float(np.mean(rec_imp)) + 1e-6)


def build_episode(rng: np.random.Generator, concept: str, noise: float, dropout_p: float) -> Tuple[List[np.ndarray], Dict[str, int]]:
    family = concept_family(concept)
    actions = FAMILY_TO_ACTIONS[family]
    states = [
        sample_concept_state(rng, concept, noise, dropout_p),
        sample_transition_state(rng, step=1, noise=noise),
        sample_transition_state(rng, step=2, noise=noise),
    ]
    targets = {
        "tool": TOOL_INDEX[actions["tool"]],
        "route": ROUTE_INDEX[actions["route"]],
        "final": FINAL_INDEX[actions["final"]],
    }
    return states, targets


def episode_pool(families: List[str], repeats: int, rng: np.random.Generator, noise: float, dropout_p: float) -> List[Tuple[List[np.ndarray], Dict[str, int]]]:
    rows = []
    for _ in range(repeats):
        for family in families:
            for concept in CONCEPTS[family]:
                rows.append(build_episode(rng, concept, noise, dropout_p))
    rng.shuffle(rows)
    return rows


def evaluate_system(
    model: MultiStepLearner,
    families: List[str],
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
    repeats: int,
) -> Dict[str, float]:
    tool_correct = 0
    route_correct = 0
    final_correct = 0
    episode_correct = 0
    total = 0
    for _ in range(repeats):
        for family in families:
            for concept in CONCEPTS[family]:
                states, targets = build_episode(rng, concept, noise, dropout_p)
                pred = model.predict_episode(states)
                tool_ok = int(pred["tool"] == targets["tool"])
                route_ok = int(pred["route"] == targets["route"])
                final_ok = int(pred["final"] == targets["final"])
                tool_correct += tool_ok
                route_correct += route_ok
                final_correct += final_ok
                episode_correct += int(tool_ok and route_ok and final_ok)
                total += 1
    return {
        "tool_accuracy": float(tool_correct / max(1, total)),
        "route_accuracy": float(route_correct / max(1, total)),
        "final_accuracy": float(final_correct / max(1, total)),
        "episode_success": float(episode_correct / max(1, total)),
    }


def run_system(use_trace: bool, seed: int, noise: float, dropout_p: float) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    model = MultiStepLearner(
        input_dim=13,
        hidden_dim=6,
        lr_head=0.13,
        lr_enc=0.045,
        lr_rec=0.055,
        use_trace=use_trace,
        stability=0.12 if use_trace else 0.0,
        seed=seed,
    )

    phase1_families = ["fruit", "animal"]
    phase2_families = ["abstract"]
    all_families = ["fruit", "animal", "abstract"]

    phase1_memory = episode_pool(phase1_families, repeats=8, rng=rng, noise=noise, dropout_p=dropout_p)
    phase2_memory = episode_pool(phase2_families, repeats=10, rng=rng, noise=noise, dropout_p=dropout_p)

    for states, targets in episode_pool(phase1_families, repeats=44, rng=rng, noise=noise, dropout_p=dropout_p):
        model.train_episode(states, targets)

    phase1_eval = evaluate_system(model, phase1_families, rng, noise, dropout_p, repeats=18)

    if use_trace:
        model.consolidate(phase1_memory)

    phase2_rows = episode_pool(phase2_families, repeats=62, rng=rng, noise=noise, dropout_p=dropout_p)
    for idx, (states, targets) in enumerate(phase2_rows):
        model.train_episode(states, targets)
        if use_trace and idx % 2 == 0:
            mem_states, mem_targets = phase1_memory[idx % len(phase1_memory)]
            model.train_episode(mem_states, mem_targets)
        if use_trace and idx % 5 == 0:
            abs_states, abs_targets = phase2_memory[idx % len(phase2_memory)]
            model.train_episode(abs_states, abs_targets)

    phase2_eval = evaluate_system(model, phase2_families, rng, noise, dropout_p, repeats=18)
    retention_eval = evaluate_system(model, phase1_families, rng, noise, dropout_p, repeats=18)
    overall_eval = evaluate_system(model, all_families, rng, noise, dropout_p, repeats=18)

    return {
        "phase1_tool_accuracy": phase1_eval["tool_accuracy"],
        "phase1_route_accuracy": phase1_eval["route_accuracy"],
        "phase1_final_accuracy": phase1_eval["final_accuracy"],
        "phase1_episode_success": phase1_eval["episode_success"],
        "phase2_tool_accuracy": phase2_eval["tool_accuracy"],
        "phase2_route_accuracy": phase2_eval["route_accuracy"],
        "phase2_final_accuracy": phase2_eval["final_accuracy"],
        "phase2_episode_success": phase2_eval["episode_success"],
        "retention_after_phase2": retention_eval["episode_success"],
        "retention_drop": float(phase1_eval["episode_success"] - retention_eval["episode_success"]),
        "overall_tool_accuracy": overall_eval["tool_accuracy"],
        "overall_route_accuracy": overall_eval["route_accuracy"],
        "overall_final_accuracy": overall_eval["final_accuracy"],
        "overall_episode_success": overall_eval["episode_success"],
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


def build_real_closure_score(summary: Dict[str, Dict[str, float]]) -> float:
    def v(key: str) -> float:
        return float(summary[key]["mean"])

    tool = v("overall_tool_accuracy")
    route = v("overall_route_accuracy")
    final = v("overall_final_accuracy")
    episode = v("overall_episode_success")
    retention = v("retention_after_phase2")
    forgetting = 1.0 - v("retention_drop")
    return float(
        0.15 * tool
        + 0.15 * route
        + 0.20 * final
        + 0.25 * episode
        + 0.15 * retention
        + 0.10 * forgetting
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-ish multi-step benchmark for grounding, credit, and continual learning")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.34)
    ap.add_argument("--dropout-p", type=float, default=0.14)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/real_multistep_agi_closure_benchmark_20260308.json",
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

    real_scores = {
        "plain_local": build_real_closure_score(plain),
        "trace_gated_local": build_real_closure_score(trace),
        "score_gain": float(build_real_closure_score(trace) - build_real_closure_score(plain)),
    }

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
        "real_closure_score": real_scores,
        "hypotheses": {
            "H1_trace_improves_episode_success": bool(trace["overall_episode_success"]["mean"] > plain["overall_episode_success"]["mean"]),
            "H2_trace_reduces_retention_drop": bool(trace["retention_drop"]["mean"] < plain["retention_drop"]["mean"]),
            "H3_trace_improves_route_memory": bool(trace["overall_route_accuracy"]["mean"] > plain["overall_route_accuracy"]["mean"]),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["systems"], ensure_ascii=False, indent=2))
    print(json.dumps(results["real_closure_score"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
