#!/usr/bin/env python
"""
Task block A upgrade:
learnable hierarchical state machine for ultra-long-horizon control.

Compared with the earlier explicit controller, this version learns how to use:
- phase code
- segment/global summary strengths
to modulate multi-timescale memory routing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import test_real_multistep_memory_gated_multiscale_scan as gated_scan
import test_real_multistep_memory_hierarchical_state_scan as hierarchical_scan


class LearnableStateMachineLearner(gated_scan.ContextGatedMemoryLearner):
    def __post_init__(self) -> None:
        super().__post_init__()
        rng = np.random.default_rng(self.seed + 17)
        gate_count = max(1, len(self.memory_betas))
        self.w_phase_tool = rng.normal(scale=0.06, size=(3, gate_count)).astype(np.float32)
        self.w_phase_route = rng.normal(scale=0.06, size=(3, gate_count)).astype(np.float32)
        self.w_phase_final = rng.normal(scale=0.06, size=(3, gate_count)).astype(np.float32)
        self.w_summary_tool = rng.normal(scale=0.05, size=(2, gate_count)).astype(np.float32)
        self.w_summary_route = rng.normal(scale=0.05, size=(2, gate_count)).astype(np.float32)
        self.w_summary_final = rng.normal(scale=0.05, size=(2, gate_count)).astype(np.float32)
        self.b_phase_tool = np.zeros(gate_count, dtype=np.float32)
        self.b_phase_route = np.zeros(gate_count, dtype=np.float32)
        self.b_phase_final = np.zeros(gate_count, dtype=np.float32)

    def _phase_params(self, head_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if head_name == "tool":
            return self.w_phase_tool, self.w_summary_tool, self.b_phase_tool
        if head_name == "route":
            return self.w_phase_route, self.w_summary_route, self.b_phase_route
        return self.w_phase_final, self.w_summary_final, self.b_phase_final

    def _set_phase_params(self, head_name: str, w_phase: np.ndarray, w_summary: np.ndarray, b_phase: np.ndarray) -> None:
        if head_name == "tool":
            self.w_phase_tool, self.w_summary_tool, self.b_phase_tool = w_phase, w_summary, b_phase
        elif head_name == "route":
            self.w_phase_route, self.w_summary_route, self.b_phase_route = w_phase, w_summary, b_phase
        else:
            self.w_phase_final, self.w_summary_final, self.b_phase_final = w_phase, w_summary, b_phase

    @staticmethod
    def state_features(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        phase = state[-3:].astype(np.float32)
        segment = state[13:26] if state.shape[0] >= 26 else state
        global_summary = state[26:39] if state.shape[0] >= 39 else state
        summary = np.array(
            [
                float(np.mean(np.abs(segment))),
                float(np.mean(np.abs(global_summary))),
            ],
            dtype=np.float32,
        )
        return phase, summary

    def gates_with_state(
        self,
        head_name: str,
        h: np.ndarray,
        memories: List[np.ndarray],
        state: np.ndarray,
        step_idx: int,
        total_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(memories) == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros(2, dtype=np.float32)
        if len(memories) == 1 or not self.use_gate:
            ones = np.ones(len(memories), dtype=np.float32)
            phase, summary = self.state_features(state)
            return ones, np.zeros(len(memories), dtype=np.float32), phase, summary

        w_gate, b_gate = self._gate_params(head_name)
        w_phase, w_summary, b_phase = self._phase_params(head_name)
        phase, summary = self.state_features(state)
        mem_strength = np.asarray([float(np.linalg.norm(memory)) for memory in memories], dtype=np.float32)
        remain_ratio = max(total_steps - step_idx - 1, 0) / max(total_steps - 1, 1)
        progress = step_idx / max(total_steps - 1, 1)
        dynamic_bias = np.array([0.18 * remain_ratio, 0.10 * (1.0 - abs(progress - 0.5)), -0.14 * progress], dtype=np.float32)
        gate_logits = (
            h @ w_gate
            + phase @ w_phase
            + summary @ w_summary
            + b_gate
            + b_phase
            + 0.06 * mem_strength
            + dynamic_bias
        ).astype(np.float32)
        gates = gated_scan.softmax(gate_logits)
        return gates, gate_logits, phase, summary

    def train_episode(self, states: List[np.ndarray], head_targets: List[Tuple[str, int]]) -> None:
        prev_h = np.zeros(self.hidden_dim, dtype=np.float32)
        prev_memories = [np.zeros(self.hidden_dim, dtype=np.float32) for _ in self.memory_betas]
        traces = []
        total_steps = len(states)

        for step_idx, (state, (head_name, target_idx)) in enumerate(zip(states, head_targets)):
            _pre, h = self.step_hidden(state, prev_h)
            memories = self.update_memories(prev_memories, h)
            gates, gate_logits, phase, summary = self.gates_with_state(head_name, h, memories, state, step_idx, total_steps)
            logits, contributions = self.logits(head_name, h, memories, gates)
            probs = gated_scan.base.softmax(logits)
            w_head, w_mem_list, _bias = self._head_params(head_name)
            target_vec = gated_scan.base.one_hot(int(target_idx), logits.shape[0])
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
                    "phase": phase.copy(),
                    "summary": summary.copy(),
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
                w_phase, w_summary, b_phase = self._phase_params(head_name)
                w_gate = w_gate + self.gate_lr * np.outer(h, gate_grad_logits).astype(np.float32)
                b_gate = b_gate + self.gate_lr * gate_grad_logits.astype(np.float32)
                w_phase = w_phase + self.gate_lr * np.outer(trace["phase"], gate_grad_logits).astype(np.float32)
                w_summary = w_summary + self.gate_lr * np.outer(trace["summary"], gate_grad_logits).astype(np.float32)
                b_phase = b_phase + 0.5 * self.gate_lr * gate_grad_logits.astype(np.float32)
                self._set_gate_params(head_name, w_gate, b_gate)
                self._set_phase_params(head_name, w_phase, w_summary, b_phase)

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
        total_steps = len(states)
        for step_idx, (state, (head_name, target_idx)) in enumerate(zip(states, head_targets)):
            _pre, h = self.step_hidden(state, prev_h)
            memories = self.update_memories(prev_memories, h)
            gates, _gate_logits, _phase, _summary = self.gates_with_state(head_name, h, memories, state, step_idx, total_steps)
            probs = gated_scan.base.softmax(self.logits(head_name, h, memories, gates)[0])
            rows.append((head_name, int(np.argmax(probs)), int(target_idx)))
            if len(gates) > 0:
                gate_rows.append(
                    {
                        "entropy": gated_scan.normalized_entropy(gates),
                        "peak": float(np.max(gates)),
                    }
                )
            prev_h = h
            prev_memories = memories
        return rows, gate_rows


def system_configs() -> Dict[str, Dict[str, object]]:
    return {
        "single_anchor_beta_086": {
            "learner": "base",
            "betas": [0.86],
            "use_gate": False,
            "hidden_dim": 6,
            "stability": 0.16,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 3.0,
            "head_lr_scale": 1.0,
            "gate_lr": 0.0,
        },
        "learnable_state_machine_h10": {
            "learner": "learnable",
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "hidden_dim": 10,
            "stability": 0.19,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 1.0,
            "head_lr_scale": 0.98,
            "gate_lr": 0.060,
        },
        "learnable_state_machine_h12": {
            "learner": "learnable",
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "hidden_dim": 12,
            "stability": 0.20,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 1.0,
            "head_lr_scale": 1.00,
            "gate_lr": 0.058,
        },
    }


def build_model(cfg: Dict[str, object], seed: int):
    common = dict(
        input_dim=42,
        hidden_dim=int(cfg["hidden_dim"]),
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
    if cfg["learner"] == "learnable":
        return LearnableStateMachineLearner(**common)
    return gated_scan.ContextGatedMemoryLearner(**common)


def run_for_length(system_name: str, length: int, seed: int, noise: float, dropout_p: float) -> Dict[str, float]:
    cfg = system_configs()[system_name]
    rng = np.random.default_rng(seed)
    model = build_model(cfg, seed)

    phase1_families = ["fruit", "animal"]
    phase2_families = ["abstract"]
    all_families = ["fruit", "animal", "abstract"]

    phase1_memory = hierarchical_scan.episode_pool(
        phase1_families, length, repeats=8, rng=rng, noise=noise, dropout_p=dropout_p, state_mode="hierarchical"
    )
    phase2_memory = hierarchical_scan.episode_pool(
        phase2_families, length, repeats=10, rng=rng, noise=noise, dropout_p=dropout_p, state_mode="hierarchical"
    )

    for states, targets in hierarchical_scan.episode_pool(
        phase1_families, length, repeats=44, rng=rng, noise=noise, dropout_p=dropout_p, state_mode="hierarchical"
    ):
        model.train_episode(states, targets)
    phase1_eval = hierarchical_scan.evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18, state_mode="hierarchical")

    model.consolidate(phase1_memory)

    phase2_rows = hierarchical_scan.episode_pool(
        phase2_families, length, repeats=62, rng=rng, noise=noise, dropout_p=dropout_p, state_mode="hierarchical"
    )
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

    phase2_eval = hierarchical_scan.evaluate_system(model, phase2_families, length, rng, noise, dropout_p, repeats=18, state_mode="hierarchical")
    retention_eval = hierarchical_scan.evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18, state_mode="hierarchical")
    overall_eval = hierarchical_scan.evaluate_system(model, all_families, length, rng, noise, dropout_p, repeats=18, state_mode="hierarchical")

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
    ap = argparse.ArgumentParser(description="Learnable state machine scan")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.34)
    ap.add_argument("--dropout-p", type=float, default=0.14)
    ap.add_argument("--lengths", type=int, nargs="+", default=[24, 28, 32])
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/real_multistep_memory_learnable_state_machine_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    lengths = [int(v) for v in args.lengths]
    max_length = max(lengths)
    systems = {}
    ranking = []
    for system_name, cfg in system_configs().items():
        rows_by_length = {length: [] for length in lengths}
        for length in lengths:
            for offset in range(int(args.num_seeds)):
                rows_by_length[length].append(
                    run_for_length(system_name, length, int(args.seed) + offset, float(args.noise), float(args.dropout_p))
                )
        systems[system_name] = gated_scan.summarize_system(rows_by_length, lengths)
        systems[system_name]["config"] = cfg
        g = systems[system_name]["global_summary"]
        ranking.append(
            {
                "system": system_name,
                "mean_closure_score": float(g["mean_closure_score"]),
                "mean_retention_score": float(g["mean_retention_score"]),
                "max_length_score": float(systems[system_name]["per_length"][str(max_length)]["real_closure_score"]),
                "closure_relative_drop": float(g["closure_relative_drop"]),
            }
        )

    ranking.sort(key=lambda row: row["mean_closure_score"], reverse=True)
    base = systems["single_anchor_beta_086"]["per_length"][str(max_length)]["real_closure_score"]
    best_machine = max([row for row in ranking if row["system"] != "single_anchor_beta_086"], key=lambda row: row["max_length_score"])

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
        "best_machine": best_machine,
        "gains": {
            "best_machine_vs_single_anchor_at_max_length": float(best_machine["max_length_score"] - base),
        },
        "hypotheses": {
            "H1_learnable_state_machine_beats_single_anchor_at_max_length": bool(best_machine["max_length_score"] > base),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["best_machine"], ensure_ascii=False, indent=2))
    print(json.dumps(results["gains"], ensure_ascii=False, indent=2))
    print(json.dumps(results["hypotheses"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
