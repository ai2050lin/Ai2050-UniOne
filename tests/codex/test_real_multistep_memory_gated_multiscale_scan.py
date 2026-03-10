#!/usr/bin/env python
"""
Compare gated multi-timescale memory readout against single-anchor and ungated
multi-anchor baselines on long-horizon real multi-step closure.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import test_real_multistep_agi_closure_benchmark as base
import test_real_multistep_agi_closure_memory_boost_scan as memory_scan


def softmax(vec: np.ndarray) -> np.ndarray:
    x = vec.astype(np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return (ex / np.sum(ex)).astype(np.float32)


def normalized_entropy(weights: np.ndarray) -> float:
    if weights.size <= 1:
        return 0.0
    probs = np.clip(weights.astype(np.float64), 1e-8, 1.0)
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / math.log(float(weights.size))


def system_configs() -> Dict[str, Dict[str, object]]:
    return {
        "trace_gated_local": {
            "betas": [],
            "use_gate": False,
            "stability": 0.12,
            "phase1_replay_stride": 2.0,
            "phase2_replay_stride": 5.0,
            "head_lr_scale": 1.0,
            "gate_lr": 0.0,
        },
        "single_anchor_beta_086": {
            "betas": [0.86],
            "use_gate": False,
            "stability": 0.16,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 3.0,
            "head_lr_scale": 1.0,
            "gate_lr": 0.0,
        },
        "dual_anchor_beta_050_086": {
            "betas": [0.50, 0.86],
            "use_gate": False,
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.92,
            "gate_lr": 0.0,
        },
        "gated_dual_anchor_beta_050_086": {
            "betas": [0.50, 0.86],
            "use_gate": True,
            "stability": 0.16,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.94,
            "gate_lr": 0.085,
        },
        "gated_triple_anchor_beta_050_080_092": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
    }


@dataclass
class ContextGatedMemoryLearner:
    input_dim: int
    hidden_dim: int
    lr_head: float
    lr_enc: float
    lr_rec: float
    gate_lr: float
    use_trace: bool
    memory_betas: List[float]
    use_gate: bool
    stability: float
    seed: int

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w_enc = rng.normal(scale=0.18, size=(self.input_dim, self.hidden_dim)).astype(np.float32)
        self.w_rec = rng.normal(scale=0.16, size=(self.hidden_dim, self.hidden_dim)).astype(np.float32)
        self.w_tool = rng.normal(scale=0.22, size=(self.hidden_dim, len(base.TOOLS))).astype(np.float32)
        self.w_route = rng.normal(scale=0.22, size=(self.hidden_dim, len(base.ROUTES))).astype(np.float32)
        self.w_final = rng.normal(scale=0.22, size=(self.hidden_dim, len(base.FINALS))).astype(np.float32)
        self.b_tool = np.zeros(len(base.TOOLS), dtype=np.float32)
        self.b_route = np.zeros(len(base.ROUTES), dtype=np.float32)
        self.b_final = np.zeros(len(base.FINALS), dtype=np.float32)
        self.enc_ref = self.w_enc.copy()
        self.rec_ref = self.w_rec.copy()
        self.enc_importance = np.zeros_like(self.w_enc)
        self.rec_importance = np.zeros_like(self.w_rec)
        self.memory_betas = [float(beta) for beta in self.memory_betas]
        self.w_mem_tool = [
            rng.normal(scale=0.10, size=(self.hidden_dim, len(base.TOOLS))).astype(np.float32)
            for _ in self.memory_betas
        ]
        self.w_mem_route = [
            rng.normal(scale=0.10, size=(self.hidden_dim, len(base.ROUTES))).astype(np.float32)
            for _ in self.memory_betas
        ]
        self.w_mem_final = [
            rng.normal(scale=0.10, size=(self.hidden_dim, len(base.FINALS))).astype(np.float32)
            for _ in self.memory_betas
        ]
        gate_count = max(1, len(self.memory_betas))
        self.w_gate_tool = rng.normal(scale=0.05, size=(self.hidden_dim, gate_count)).astype(np.float32)
        self.w_gate_route = rng.normal(scale=0.05, size=(self.hidden_dim, gate_count)).astype(np.float32)
        self.w_gate_final = rng.normal(scale=0.05, size=(self.hidden_dim, gate_count)).astype(np.float32)
        self.b_gate_tool = np.zeros(gate_count, dtype=np.float32)
        self.b_gate_route = np.zeros(gate_count, dtype=np.float32)
        self.b_gate_final = np.zeros(gate_count, dtype=np.float32)

    def _head_params(self, head_name: str) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        if head_name == "tool":
            return self.w_tool, self.w_mem_tool, self.b_tool
        if head_name == "route":
            return self.w_route, self.w_mem_route, self.b_route
        return self.w_final, self.w_mem_final, self.b_final

    def _set_head_params(self, head_name: str, w_head: np.ndarray, w_mem_list: List[np.ndarray], bias: np.ndarray) -> None:
        if head_name == "tool":
            self.w_tool, self.w_mem_tool, self.b_tool = w_head, w_mem_list, bias
        elif head_name == "route":
            self.w_route, self.w_mem_route, self.b_route = w_head, w_mem_list, bias
        else:
            self.w_final, self.w_mem_final, self.b_final = w_head, w_mem_list, bias

    def _gate_params(self, head_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if head_name == "tool":
            return self.w_gate_tool, self.b_gate_tool
        if head_name == "route":
            return self.w_gate_route, self.b_gate_route
        return self.w_gate_final, self.b_gate_final

    def _set_gate_params(self, head_name: str, w_gate: np.ndarray, b_gate: np.ndarray) -> None:
        if head_name == "tool":
            self.w_gate_tool, self.b_gate_tool = w_gate, b_gate
        elif head_name == "route":
            self.w_gate_route, self.b_gate_route = w_gate, b_gate
        else:
            self.w_gate_final, self.b_gate_final = w_gate, b_gate

    def step_hidden(self, x: np.ndarray, prev_h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pre = x @ self.w_enc + prev_h @ self.w_rec
        h = np.tanh(pre).astype(np.float32)
        return pre.astype(np.float32), h

    def update_memories(self, prev_memories: List[np.ndarray], h: np.ndarray) -> List[np.ndarray]:
        if not self.memory_betas:
            return []
        next_memories = []
        for prev_memory, beta in zip(prev_memories, self.memory_betas):
            next_memories.append((beta * prev_memory + (1.0 - beta) * h).astype(np.float32))
        return next_memories

    def gates(self, head_name: str, h: np.ndarray, memories: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if len(memories) == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        if len(memories) == 1 or not self.use_gate:
            ones = np.ones(len(memories), dtype=np.float32)
            return ones, np.zeros(len(memories), dtype=np.float32)
        w_gate, b_gate = self._gate_params(head_name)
        mem_strength = np.asarray([float(np.linalg.norm(memory)) for memory in memories], dtype=np.float32)
        gate_logits = (h @ w_gate + b_gate + 0.05 * mem_strength).astype(np.float32)
        return softmax(gate_logits), gate_logits

    def logits(self, head_name: str, h: np.ndarray, memories: List[np.ndarray], gates: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        w_head, w_mem_list, bias = self._head_params(head_name)
        logits = h @ w_head + bias
        contributions = []
        for idx, (memory, w_mem) in enumerate(zip(memories, w_mem_list)):
            contrib = memory @ w_mem
            contributions.append(contrib.astype(np.float32))
            logits = logits + gates[idx] * contrib
        return logits.astype(np.float32), contributions

    def _apply_regularized_update(
        self,
        weights: np.ndarray,
        update: np.ndarray,
        ref: np.ndarray,
        importance: np.ndarray,
        lr: float,
    ) -> np.ndarray:
        adj = update.astype(np.float32)
        if self.stability > 0.0:
            adj = adj - self.stability * importance * (weights - ref)
        return weights + lr * adj

    def train_episode(self, states: List[np.ndarray], head_targets: List[Tuple[str, int]]) -> None:
        prev_h = np.zeros(self.hidden_dim, dtype=np.float32)
        prev_memories = [np.zeros(self.hidden_dim, dtype=np.float32) for _ in self.memory_betas]
        traces = []

        for state, (head_name, target_idx) in zip(states, head_targets):
            _pre, h = self.step_hidden(state, prev_h)
            memories = self.update_memories(prev_memories, h)
            gates, gate_logits = self.gates(head_name, h, memories)
            logits, contributions = self.logits(head_name, h, memories, gates)
            probs = base.softmax(logits)
            w_head, w_mem_list, _bias = self._head_params(head_name)
            target_vec = base.one_hot(int(target_idx), logits.shape[0])
            err = (target_vec - probs).astype(np.float32)
            traces.append(
                {
                    "head": head_name,
                    "state": state,
                    "prev_h": prev_h.copy(),
                    "h": h,
                    "memories": [memory.copy() for memory in memories],
                    "gates": gates.copy(),
                    "gate_logits": gate_logits.copy(),
                    "contributions": [c.copy() for c in contributions],
                    "err": err,
                    "w_head": w_head.copy(),
                    "w_mem_list": [w.copy() for w in w_mem_list],
                }
            )
            prev_h = h
            prev_memories = memories

        for trace in traces:
            head_name = trace["head"]
            h = trace["h"]
            memories = trace["memories"]
            gates = trace["gates"]
            contributions = trace["contributions"]
            err = trace["err"]
            w_head, w_mem_list, bias = self._head_params(head_name)
            w_head = w_head + self.lr_head * np.outer(h, err).astype(np.float32)
            updated_mem = []
            for idx, (memory, w_mem) in enumerate(zip(memories, w_mem_list)):
                updated_mem.append(w_mem + self.lr_head * gates[idx] * np.outer(memory, err).astype(np.float32))
            bias = bias + self.lr_head * err.astype(np.float32)
            self._set_head_params(head_name, w_head, updated_mem, bias)

            gate_grad_logits = np.zeros(len(gates), dtype=np.float32)
            if self.use_gate and len(gates) > 1:
                gate_signal = np.asarray([float(np.dot(contrib, err)) for contrib in contributions], dtype=np.float32)
                gate_grad_logits = (gates * (gate_signal - float(np.sum(gates * gate_signal)))).astype(np.float32)
                w_gate, b_gate = self._gate_params(head_name)
                w_gate = w_gate + self.gate_lr * np.outer(h, gate_grad_logits).astype(np.float32)
                b_gate = b_gate + self.gate_lr * gate_grad_logits.astype(np.float32)
                self._set_gate_params(head_name, w_gate, b_gate)

            if self.use_trace:
                grad_h = (w_head @ err).astype(np.float32)
                for gate_value, beta, w_mem in zip(gates, self.memory_betas, updated_mem):
                    grad_h = grad_h + gate_value * (1.0 - beta) * (w_mem @ err).astype(np.float32)
                if self.use_gate and len(gates) > 1:
                    w_gate, _ = self._gate_params(head_name)
                    grad_h = grad_h + (w_gate @ gate_grad_logits).astype(np.float32)
                grad_pre = (grad_h * (1.0 - np.square(h))).astype(np.float32)
                grad_enc = np.outer(trace["state"], grad_pre).astype(np.float32)
                grad_rec = np.outer(trace["prev_h"], grad_pre).astype(np.float32)
                self.w_enc = self._apply_regularized_update(self.w_enc, grad_enc, self.enc_ref, self.enc_importance, self.lr_enc)
                self.w_rec = self._apply_regularized_update(self.w_rec, grad_rec, self.rec_ref, self.rec_importance, self.lr_rec)

    def predict_episode(self, states: List[np.ndarray], head_targets: List[Tuple[str, int]]) -> Tuple[List[Tuple[str, int, int]], List[Dict[str, float]]]:
        prev_h = np.zeros(self.hidden_dim, dtype=np.float32)
        prev_memories = [np.zeros(self.hidden_dim, dtype=np.float32) for _ in self.memory_betas]
        rows = []
        gate_rows = []
        for state, (head_name, target_idx) in zip(states, head_targets):
            _pre, h = self.step_hidden(state, prev_h)
            memories = self.update_memories(prev_memories, h)
            gates, _gate_logits = self.gates(head_name, h, memories)
            probs = base.softmax(self.logits(head_name, h, memories, gates)[0])
            rows.append((head_name, int(np.argmax(probs)), int(target_idx)))
            if len(gates) > 0:
                gate_rows.append(
                    {
                        "entropy": normalized_entropy(gates),
                        "peak": float(np.max(gates)),
                    }
                )
            prev_h = h
            prev_memories = memories
        return rows, gate_rows

    def consolidate(self, episodes: List[Tuple[List[np.ndarray], List[Tuple[str, int]]]]) -> None:
        enc_imp = np.zeros_like(self.w_enc)
        rec_imp = np.zeros_like(self.w_rec)
        count = 0
        for states, head_targets in episodes:
            prev_h = np.zeros(self.hidden_dim, dtype=np.float32)
            prev_memories = [np.zeros(self.hidden_dim, dtype=np.float32) for _ in self.memory_betas]
            for state, (head_name, target_idx) in zip(states, head_targets):
                _pre, h = self.step_hidden(state, prev_h)
                memories = self.update_memories(prev_memories, h)
                gates, gate_logits = self.gates(head_name, h, memories)
                w_head, w_mem_list, _bias = self._head_params(head_name)
                logits, contributions = self.logits(head_name, h, memories, gates)
                probs = base.softmax(logits)
                target_vec = base.one_hot(int(target_idx), probs.shape[0])
                err = (target_vec - probs).astype(np.float32)
                grad_h = (w_head @ err).astype(np.float32)
                for gate_value, beta, w_mem in zip(gates, self.memory_betas, w_mem_list):
                    grad_h = grad_h + gate_value * (1.0 - beta) * (w_mem @ err).astype(np.float32)
                if self.use_gate and len(gates) > 1:
                    gate_signal = np.asarray([float(np.dot(contrib, err)) for contrib in contributions], dtype=np.float32)
                    gate_grad_logits = (gates * (gate_signal - float(np.sum(gates * gate_signal)))).astype(np.float32)
                    w_gate, _ = self._gate_params(head_name)
                    grad_h = grad_h + (w_gate @ gate_grad_logits).astype(np.float32)
                grad_pre = (grad_h * (1.0 - np.square(h))).astype(np.float32)
                enc_imp += np.abs(np.outer(state, grad_pre)).astype(np.float32)
                rec_imp += np.abs(np.outer(prev_h, grad_pre)).astype(np.float32)
                prev_h = h
                prev_memories = memories
                count += 1
        if count > 0:
            enc_imp /= float(count)
            rec_imp /= float(count)
        self.enc_ref = self.w_enc.copy()
        self.rec_ref = self.w_rec.copy()
        self.enc_importance = enc_imp / (float(np.mean(enc_imp)) + 1e-6)
        self.rec_importance = rec_imp / (float(np.mean(rec_imp)) + 1e-6)


def evaluate_system(
    model: ContextGatedMemoryLearner,
    families: List[str],
    length: int,
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
    repeats: int,
) -> Dict[str, float]:
    tool_correct = 0
    route_correct = 0
    final_correct = 0
    episode_correct = 0
    route_total = 0
    total = 0
    gate_entropy = []
    gate_peak = []
    for _ in range(repeats):
        for family in families:
            for concept in base.CONCEPTS[family]:
                states, targets = memory_scan.build_episode(rng, concept, length, noise, dropout_p)
                preds, gate_rows = model.predict_episode(states, targets)
                step_ok = []
                for head_name, pred_idx, target_idx in preds:
                    ok = int(pred_idx == target_idx)
                    if head_name == "tool":
                        tool_correct += ok
                    elif head_name == "route":
                        route_correct += ok
                        route_total += 1
                    else:
                        final_correct += ok
                    step_ok.append(ok)
                for row in gate_rows:
                    gate_entropy.append(float(row["entropy"]))
                    gate_peak.append(float(row["peak"]))
                episode_correct += int(all(step_ok))
                total += 1
    return {
        "tool_accuracy": float(tool_correct / max(1, total)),
        "route_accuracy": float(route_correct / max(1, route_total)),
        "final_accuracy": float(final_correct / max(1, total)),
        "episode_success": float(episode_correct / max(1, total)),
        "gate_entropy": float(np.mean(gate_entropy)) if gate_entropy else 0.0,
        "gate_peak": float(np.mean(gate_peak)) if gate_peak else 0.0,
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


def summarize_system(rows_by_length: Dict[int, List[Dict[str, float]]], lengths: List[int]) -> Dict[str, object]:
    per_length = {}
    closure_curve = []
    retention_curve = []
    gate_entropy_curve = []
    gate_peak_curve = []
    for length in lengths:
        summary = summarize_runs(rows_by_length[length])
        score = memory_scan.closure_score(summary)
        per_length[str(length)] = {
            "summary": summary,
            "real_closure_score": score,
        }
        closure_curve.append(score)
        retention_curve.append(float(summary["retention_after_phase2"]["mean"]))
        gate_entropy_curve.append(float(summary["gate_entropy"]["mean"]))
        gate_peak_curve.append(float(summary["gate_peak"]["mean"]))

    return {
        "per_length": per_length,
        "global_summary": {
            "lengths": lengths,
            "closure_curve": closure_curve,
            "retention_curve": retention_curve,
            "gate_entropy_curve": gate_entropy_curve,
            "gate_peak_curve": gate_peak_curve,
            "closure_decay_slope": memory_scan.fit_slope(lengths, closure_curve),
            "retention_decay_slope": memory_scan.fit_slope(lengths, retention_curve),
            "closure_relative_drop": memory_scan.relative_drop(closure_curve),
            "retention_relative_drop": memory_scan.relative_drop(retention_curve),
            "mean_closure_score": float(np.mean(closure_curve)),
            "mean_retention_score": float(np.mean(retention_curve)),
            "mean_gate_entropy": float(np.mean(gate_entropy_curve)),
            "mean_gate_peak": float(np.mean(gate_peak_curve)),
        },
    }


def run_for_length(
    system_name: str,
    length: int,
    seed: int,
    noise: float,
    dropout_p: float,
) -> Dict[str, float]:
    cfg = system_configs()[system_name]
    rng = np.random.default_rng(seed)
    model = ContextGatedMemoryLearner(
        input_dim=13,
        hidden_dim=6,
        lr_head=0.13 * float(cfg["head_lr_scale"]),
        lr_enc=0.045,
        lr_rec=0.055,
        gate_lr=float(cfg["gate_lr"]),
        use_trace=True,
        memory_betas=[float(beta) for beta in cfg["betas"]],
        use_gate=bool(cfg["use_gate"]),
        stability=float(cfg["stability"]),
        seed=seed,
    )

    phase1_families = ["fruit", "animal"]
    phase2_families = ["abstract"]
    all_families = ["fruit", "animal", "abstract"]

    phase1_memory = memory_scan.episode_pool(phase1_families, length, repeats=8, rng=rng, noise=noise, dropout_p=dropout_p)
    phase2_memory = memory_scan.episode_pool(phase2_families, length, repeats=10, rng=rng, noise=noise, dropout_p=dropout_p)

    for states, targets in memory_scan.episode_pool(phase1_families, length, repeats=44, rng=rng, noise=noise, dropout_p=dropout_p):
        model.train_episode(states, targets)
    phase1_eval = evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18)

    model.consolidate(phase1_memory)

    phase2_rows = memory_scan.episode_pool(phase2_families, length, repeats=62, rng=rng, noise=noise, dropout_p=dropout_p)
    phase1_stride = int(cfg["phase1_replay_stride"])
    phase2_stride = int(cfg["phase2_replay_stride"])
    for idx, (states, targets) in enumerate(phase2_rows):
        model.train_episode(states, targets)
        if phase1_stride > 0 and idx % phase1_stride == 0:
            mem_states, mem_targets = phase1_memory[idx % len(phase1_memory)]
            model.train_episode(mem_states, mem_targets)
        if phase2_stride > 0 and idx % phase2_stride == 0:
            abs_states, abs_targets = phase2_memory[idx % len(phase2_memory)]
            model.train_episode(abs_states, abs_targets)

    phase2_eval = evaluate_system(model, phase2_families, length, rng, noise, dropout_p, repeats=18)
    retention_eval = evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18)
    overall_eval = evaluate_system(model, all_families, length, rng, noise, dropout_p, repeats=18)

    return {
        "phase1_episode_success": phase1_eval["episode_success"],
        "phase2_episode_success": phase2_eval["episode_success"],
        "retention_after_phase2": retention_eval["episode_success"],
        "retention_drop": float(phase1_eval["episode_success"] - retention_eval["episode_success"]),
        "overall_tool_accuracy": overall_eval["tool_accuracy"],
        "overall_route_accuracy": overall_eval["route_accuracy"],
        "overall_final_accuracy": overall_eval["final_accuracy"],
        "overall_episode_success": overall_eval["episode_success"],
        "gate_entropy": overall_eval["gate_entropy"],
        "gate_peak": overall_eval["gate_peak"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Context-gated multi-timescale memory scan")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.34)
    ap.add_argument("--dropout-p", type=float, default=0.14)
    ap.add_argument("--lengths", type=int, nargs="+", default=[6, 8, 10, 12])
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/real_multistep_memory_gated_multiscale_scan_20260308.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    lengths = [int(v) for v in args.lengths]
    systems = {}
    ranking = []
    max_length = max(lengths)
    for system_name in system_configs().keys():
        rows_by_length = {length: [] for length in lengths}
        for length in lengths:
            for offset in range(int(args.num_seeds)):
                seed = int(args.seed) + offset
                rows_by_length[length].append(run_for_length(system_name, length, seed, float(args.noise), float(args.dropout_p)))
        systems[system_name] = summarize_system(rows_by_length, lengths)
        systems[system_name]["config"] = system_configs()[system_name]
        g = systems[system_name]["global_summary"]
        ranking.append(
            {
                "system": system_name,
                "mean_closure_score": float(g["mean_closure_score"]),
                "mean_retention_score": float(g["mean_retention_score"]),
                "closure_relative_drop": float(g["closure_relative_drop"]),
                "max_length_score": float(systems[system_name]["per_length"][str(max_length)]["real_closure_score"]),
                "mean_gate_entropy": float(g["mean_gate_entropy"]),
                "mean_gate_peak": float(g["mean_gate_peak"]),
            }
        )

    ranking.sort(key=lambda row: row["mean_closure_score"], reverse=True)
    trace_ref = systems["trace_gated_local"]
    single_ref = systems["single_anchor_beta_086"]
    dual_ref = systems["dual_anchor_beta_050_086"]
    gated_dual_ref = systems["gated_dual_anchor_beta_050_086"]

    def gain_vs(ref_name: str, target_name: str) -> Dict[str, object]:
        out = {}
        area = 0.0
        for length in lengths:
            target_score = float(systems[target_name]["per_length"][str(length)]["real_closure_score"])
            ref_score = float(systems[ref_name]["per_length"][str(length)]["real_closure_score"])
            gain = target_score - ref_score
            area += gain
            out[str(length)] = {
                "closure_gain": float(gain),
                "retention_gain": float(
                    systems[target_name]["per_length"][str(length)]["summary"]["retention_after_phase2"]["mean"]
                    - systems[ref_name]["per_length"][str(length)]["summary"]["retention_after_phase2"]["mean"]
                ),
            }
        return {
            "per_length": out,
            "advantage_area": float(area),
            "final_length_gain": float(out[str(max_length)]["closure_gain"]),
        }

    gated_ranking = [row for row in ranking if row["system"].startswith("gated_")]

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "lengths": lengths,
            "runtime_sec": float(time.time() - t0),
        },
        "systems": systems,
        "ranking": ranking,
        "gains": {
            "gated_dual_vs_trace": gain_vs("trace_gated_local", "gated_dual_anchor_beta_050_086"),
            "gated_dual_vs_single": gain_vs("single_anchor_beta_086", "gated_dual_anchor_beta_050_086"),
            "gated_dual_vs_dual": gain_vs("dual_anchor_beta_050_086", "gated_dual_anchor_beta_050_086"),
            "gated_triple_vs_trace": gain_vs("trace_gated_local", "gated_triple_anchor_beta_050_080_092"),
            "gated_triple_vs_single": gain_vs("single_anchor_beta_086", "gated_triple_anchor_beta_050_080_092"),
        },
        "best_systems": {
            "best_mean_closure": max(ranking, key=lambda row: row["mean_closure_score"]),
            "best_max_length": max(ranking, key=lambda row: row["max_length_score"]),
            "best_mean_retention": max(ranking, key=lambda row: row["mean_retention_score"]),
            "best_gated_mean_closure": max(gated_ranking, key=lambda row: row["mean_closure_score"]),
            "best_gated_max_length": max(gated_ranking, key=lambda row: row["max_length_score"]),
            "best_gate_selectivity": max(gated_ranking, key=lambda row: row["mean_gate_peak"]),
        },
        "hypotheses": {
            "H1_gated_dual_beats_ungated_dual_on_average": bool(
                gated_dual_ref["global_summary"]["mean_closure_score"] > dual_ref["global_summary"]["mean_closure_score"]
            ),
            "H2_gated_dual_beats_single_anchor_on_average": bool(
                gated_dual_ref["global_summary"]["mean_closure_score"] > single_ref["global_summary"]["mean_closure_score"]
            ),
            "H3_gated_dual_beats_single_anchor_at_max_length": bool(
                gated_dual_ref["per_length"][str(max_length)]["real_closure_score"]
                > single_ref["per_length"][str(max_length)]["real_closure_score"]
            ),
            "H4_gating_is_nontrivial": bool(gated_dual_ref["global_summary"]["mean_gate_entropy"] < 0.95),
            "H5_gated_triple_beats_single_anchor_at_max_length": bool(
                systems["gated_triple_anchor_beta_050_080_092"]["per_length"][str(max_length)]["real_closure_score"]
                > single_ref["per_length"][str(max_length)]["real_closure_score"]
            ),
            "H6_gated_triple_flattens_decay_vs_single_anchor": bool(
                systems["gated_triple_anchor_beta_050_080_092"]["global_summary"]["closure_relative_drop"]
                < single_ref["global_summary"]["closure_relative_drop"]
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["best_systems"], ensure_ascii=False, indent=2))
    print(json.dumps(results["hypotheses"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
