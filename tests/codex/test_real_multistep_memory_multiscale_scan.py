#!/usr/bin/env python
"""
Compare single-timescale and multi-timescale slow-memory systems on long-horizon
real multi-step closure.

Systems:
- trace_gated_local: baseline trace + replay/stability
- single_anchor_beta_086: one slow memory bank with beta=0.86
- dual_anchor_beta_050_086: two slow memory banks with beta=(0.50, 0.86)
- triple_anchor_beta_050_080_092: three slow memory banks with beta=(0.50, 0.80, 0.92)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import test_real_multistep_agi_closure_benchmark as base
import test_real_multistep_agi_closure_memory_boost_scan as memory_scan


def system_configs() -> Dict[str, Dict[str, object]]:
    return {
        "trace_gated_local": {
            "betas": [],
            "stability": 0.12,
            "phase1_replay_stride": 2.0,
            "phase2_replay_stride": 5.0,
            "head_lr_scale": 1.0,
        },
        "single_anchor_beta_086": {
            "betas": [0.86],
            "stability": 0.16,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 3.0,
            "head_lr_scale": 1.0,
        },
        "dual_anchor_beta_050_086": {
            "betas": [0.50, 0.86],
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.92,
        },
        "triple_anchor_beta_050_080_092": {
            "betas": [0.50, 0.80, 0.92],
            "stability": 0.18,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.88,
        },
    }


@dataclass
class MultiScaleMemoryLearner:
    input_dim: int
    hidden_dim: int
    lr_head: float
    lr_enc: float
    lr_rec: float
    use_trace: bool
    memory_betas: List[float]
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

    def logits(self, head_name: str, h: np.ndarray, memories: List[np.ndarray]) -> np.ndarray:
        w_head, w_mem_list, bias = self._head_params(head_name)
        logits = h @ w_head + bias
        for memory, w_mem in zip(memories, w_mem_list):
            logits = logits + memory @ w_mem
        return logits.astype(np.float32)

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
            logits = self.logits(head_name, h, memories)
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
                    "err": err,
                    "w_head": w_head.copy(),
                    "w_mem_list": [w.copy() for w in w_mem_list],
                }
            )
            prev_h = h
            prev_memories = memories

        for trace in traces:
            head_name = trace["head"]
            w_head, w_mem_list, bias = self._head_params(head_name)
            h = trace["h"]
            memories = trace["memories"]
            err = trace["err"]
            w_head = w_head + self.lr_head * np.outer(h, err).astype(np.float32)
            updated_mem = []
            for memory, w_mem in zip(memories, w_mem_list):
                updated_mem.append(w_mem + self.lr_head * np.outer(memory, err).astype(np.float32))
            bias = bias + self.lr_head * err.astype(np.float32)
            self._set_head_params(head_name, w_head, updated_mem, bias)

            if self.use_trace:
                grad_h = (w_head @ err).astype(np.float32)
                for beta, w_mem in zip(self.memory_betas, updated_mem):
                    grad_h = grad_h + (1.0 - beta) * (w_mem @ err).astype(np.float32)
                grad_pre = (grad_h * (1.0 - np.square(h))).astype(np.float32)
                grad_enc = np.outer(trace["state"], grad_pre).astype(np.float32)
                grad_rec = np.outer(trace["prev_h"], grad_pre).astype(np.float32)
                self.w_enc = self._apply_regularized_update(self.w_enc, grad_enc, self.enc_ref, self.enc_importance, self.lr_enc)
                self.w_rec = self._apply_regularized_update(self.w_rec, grad_rec, self.rec_ref, self.rec_importance, self.lr_rec)

    def predict_episode(self, states: List[np.ndarray], head_targets: List[Tuple[str, int]]) -> List[Tuple[str, int, int]]:
        prev_h = np.zeros(self.hidden_dim, dtype=np.float32)
        prev_memories = [np.zeros(self.hidden_dim, dtype=np.float32) for _ in self.memory_betas]
        rows = []
        for state, (head_name, target_idx) in zip(states, head_targets):
            _pre, h = self.step_hidden(state, prev_h)
            memories = self.update_memories(prev_memories, h)
            probs = base.softmax(self.logits(head_name, h, memories))
            rows.append((head_name, int(np.argmax(probs)), int(target_idx)))
            prev_h = h
            prev_memories = memories
        return rows

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
                w_head, w_mem_list, _bias = self._head_params(head_name)
                probs = base.softmax(self.logits(head_name, h, memories))
                target_vec = base.one_hot(int(target_idx), probs.shape[0])
                err = (target_vec - probs).astype(np.float32)
                grad_h = (w_head @ err).astype(np.float32)
                for beta, w_mem in zip(self.memory_betas, w_mem_list):
                    grad_h = grad_h + (1.0 - beta) * (w_mem @ err).astype(np.float32)
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


def run_for_length(
    system_name: str,
    length: int,
    seed: int,
    noise: float,
    dropout_p: float,
) -> Dict[str, float]:
    cfg = system_configs()[system_name]
    rng = np.random.default_rng(seed)
    model = MultiScaleMemoryLearner(
        input_dim=13,
        hidden_dim=6,
        lr_head=0.13 * float(cfg["head_lr_scale"]),
        lr_enc=0.045,
        lr_rec=0.055,
        use_trace=True,
        memory_betas=[float(beta) for beta in cfg["betas"]],
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
    phase1_eval = memory_scan.evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18)

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

    phase2_eval = memory_scan.evaluate_system(model, phase2_families, length, rng, noise, dropout_p, repeats=18)
    retention_eval = memory_scan.evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18)
    overall_eval = memory_scan.evaluate_system(model, all_families, length, rng, noise, dropout_p, repeats=18)

    return {
        "phase1_episode_success": phase1_eval["episode_success"],
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


def summarize_system(rows_by_length: Dict[int, List[Dict[str, float]]], lengths: List[int]) -> Dict[str, object]:
    per_length = {}
    closure_curve = []
    retention_curve = []
    for length in lengths:
        summary = summarize_runs(rows_by_length[length])
        score = memory_scan.closure_score(summary)
        per_length[str(length)] = {
            "summary": summary,
            "real_closure_score": score,
        }
        closure_curve.append(score)
        retention_curve.append(float(summary["retention_after_phase2"]["mean"]))

    return {
        "per_length": per_length,
        "global_summary": {
            "lengths": lengths,
            "closure_curve": closure_curve,
            "retention_curve": retention_curve,
            "closure_decay_slope": memory_scan.fit_slope(lengths, closure_curve),
            "retention_decay_slope": memory_scan.fit_slope(lengths, retention_curve),
            "closure_relative_drop": memory_scan.relative_drop(closure_curve),
            "retention_relative_drop": memory_scan.relative_drop(retention_curve),
            "mean_closure_score": float(np.mean(closure_curve)),
            "mean_retention_score": float(np.mean(retention_curve)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-timescale memory scan for real multi-step AGI closure")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.34)
    ap.add_argument("--dropout-p", type=float, default=0.14)
    ap.add_argument("--lengths", type=int, nargs="+", default=[6, 8, 10, 12])
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/real_multistep_memory_multiscale_scan_20260308.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    lengths = [int(v) for v in args.lengths]
    systems: Dict[str, object] = {}
    ranking = []
    max_length = max(lengths)

    for system_name in system_configs().keys():
        rows_by_length: Dict[int, List[Dict[str, float]]] = {length: [] for length in lengths}
        for length in lengths:
            for offset in range(int(args.num_seeds)):
                seed = int(args.seed) + offset
                rows_by_length[length].append(run_for_length(system_name, length, seed, float(args.noise), float(args.dropout_p)))
        systems[system_name] = summarize_system(rows_by_length, lengths)
        systems[system_name]["config"] = system_configs()[system_name]
        global_summary = systems[system_name]["global_summary"]
        ranking.append(
            {
                "system": system_name,
                "mean_closure_score": float(global_summary["mean_closure_score"]),
                "mean_retention_score": float(global_summary["mean_retention_score"]),
                "closure_relative_drop": float(global_summary["closure_relative_drop"]),
                "max_length_score": float(systems[system_name]["per_length"][str(max_length)]["real_closure_score"]),
            }
        )

    trace_reference = systems["trace_gated_local"]
    single_reference = systems["single_anchor_beta_086"]
    gains_vs_trace = {}
    gains_vs_single = {}
    for system_name, row in systems.items():
        if system_name == "trace_gated_local":
            continue
        per_length_trace = {}
        per_length_single = {}
        area_trace = 0.0
        area_single = 0.0
        for length in lengths:
            this_score = float(row["per_length"][str(length)]["real_closure_score"])
            trace_score = float(trace_reference["per_length"][str(length)]["real_closure_score"])
            single_score = float(single_reference["per_length"][str(length)]["real_closure_score"])
            gain_trace = this_score - trace_score
            gain_single = this_score - single_score
            area_trace += gain_trace
            area_single += gain_single
            per_length_trace[str(length)] = {
                "closure_gain": float(gain_trace),
                "retention_gain": float(
                    row["per_length"][str(length)]["summary"]["retention_after_phase2"]["mean"]
                    - trace_reference["per_length"][str(length)]["summary"]["retention_after_phase2"]["mean"]
                ),
            }
            per_length_single[str(length)] = {
                "closure_gain": float(gain_single),
                "retention_gain": float(
                    row["per_length"][str(length)]["summary"]["retention_after_phase2"]["mean"]
                    - single_reference["per_length"][str(length)]["summary"]["retention_after_phase2"]["mean"]
                ),
            }
        gains_vs_trace[system_name] = {
            "per_length": per_length_trace,
            "advantage_area": float(area_trace),
            "final_length_gain": float(per_length_trace[str(max_length)]["closure_gain"]),
        }
        gains_vs_single[system_name] = {
            "per_length": per_length_single,
            "advantage_area": float(area_single),
            "final_length_gain": float(per_length_single[str(max_length)]["closure_gain"]),
        }

    ranking.sort(key=lambda row: row["mean_closure_score"], reverse=True)
    best_mean = max(ranking, key=lambda row: row["mean_closure_score"])
    best_max = max(ranking, key=lambda row: row["max_length_score"])
    best_retention = max(ranking, key=lambda row: row["mean_retention_score"])
    best_slowest_drop = min(ranking, key=lambda row: row["closure_relative_drop"])

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
        "gains_vs_trace": gains_vs_trace,
        "gains_vs_single_anchor": gains_vs_single,
        "best_systems": {
            "best_mean_closure": best_mean,
            "best_max_length": best_max,
            "best_mean_retention": best_retention,
            "slowest_decay": best_slowest_drop,
        },
        "hypotheses": {
            "H1_multiscale_beats_trace_on_average": bool(
                systems["dual_anchor_beta_050_086"]["global_summary"]["mean_closure_score"]
                > trace_reference["global_summary"]["mean_closure_score"]
                or systems["triple_anchor_beta_050_080_092"]["global_summary"]["mean_closure_score"]
                > trace_reference["global_summary"]["mean_closure_score"]
            ),
            "H2_multiscale_beats_single_anchor_on_average": bool(
                systems["dual_anchor_beta_050_086"]["global_summary"]["mean_closure_score"]
                > single_reference["global_summary"]["mean_closure_score"]
                or systems["triple_anchor_beta_050_080_092"]["global_summary"]["mean_closure_score"]
                > single_reference["global_summary"]["mean_closure_score"]
            ),
            "H3_multiscale_beats_single_anchor_at_max_length": bool(
                systems["dual_anchor_beta_050_086"]["per_length"][str(max_length)]["real_closure_score"]
                > single_reference["per_length"][str(max_length)]["real_closure_score"]
                or systems["triple_anchor_beta_050_080_092"]["per_length"][str(max_length)]["real_closure_score"]
                > single_reference["per_length"][str(max_length)]["real_closure_score"]
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["best_systems"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
