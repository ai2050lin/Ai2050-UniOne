#!/usr/bin/env python
"""
Scan whether a slow-memory anchor reduces long-horizon decay.

Systems:
- plain_local: no trace, no slow memory
- trace_gated_local: local trace + replay/stability
- trace_anchor_local: trace_gated_local plus a slow memory anchor that feeds
  every decision head through an extra memory pathway
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


def system_configs() -> Dict[str, Dict[str, float]]:
    return {
        "plain_local": {
            "use_trace": 0.0,
            "use_slow_memory": 0.0,
            "memory_decay": 0.0,
            "stability": 0.0,
            "phase1_replay_stride": 0.0,
            "phase2_replay_stride": 0.0,
        },
        "trace_gated_local": {
            "use_trace": 1.0,
            "use_slow_memory": 0.0,
            "memory_decay": 0.0,
            "stability": 0.12,
            "phase1_replay_stride": 2.0,
            "phase2_replay_stride": 5.0,
        },
        "trace_anchor_local": {
            "use_trace": 1.0,
            "use_slow_memory": 1.0,
            "memory_decay": 0.86,
            "stability": 0.16,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 3.0,
        },
    }


@dataclass
class MemoryLearner:
    input_dim: int
    hidden_dim: int
    lr_head: float
    lr_enc: float
    lr_rec: float
    use_trace: bool
    use_slow_memory: bool
    memory_decay: float
    stability: float
    seed: int

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w_enc = rng.normal(scale=0.18, size=(self.input_dim, self.hidden_dim)).astype(np.float32)
        self.w_rec = rng.normal(scale=0.16, size=(self.hidden_dim, self.hidden_dim)).astype(np.float32)
        self.w_tool = rng.normal(scale=0.22, size=(self.hidden_dim, len(base.TOOLS))).astype(np.float32)
        self.w_route = rng.normal(scale=0.22, size=(self.hidden_dim, len(base.ROUTES))).astype(np.float32)
        self.w_final = rng.normal(scale=0.22, size=(self.hidden_dim, len(base.FINALS))).astype(np.float32)
        self.w_mem_tool = rng.normal(scale=0.12, size=(self.hidden_dim, len(base.TOOLS))).astype(np.float32)
        self.w_mem_route = rng.normal(scale=0.12, size=(self.hidden_dim, len(base.ROUTES))).astype(np.float32)
        self.w_mem_final = rng.normal(scale=0.12, size=(self.hidden_dim, len(base.FINALS))).astype(np.float32)
        self.b_tool = np.zeros(len(base.TOOLS), dtype=np.float32)
        self.b_route = np.zeros(len(base.ROUTES), dtype=np.float32)
        self.b_final = np.zeros(len(base.FINALS), dtype=np.float32)
        self.enc_ref = self.w_enc.copy()
        self.rec_ref = self.w_rec.copy()
        self.enc_importance = np.zeros_like(self.w_enc)
        self.rec_importance = np.zeros_like(self.w_rec)

    def _head_params(self, head_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if head_name == "tool":
            return self.w_tool, self.w_mem_tool, self.b_tool
        if head_name == "route":
            return self.w_route, self.w_mem_route, self.b_route
        return self.w_final, self.w_mem_final, self.b_final

    def _set_head_params(self, head_name: str, w_head: np.ndarray, w_mem: np.ndarray, bias: np.ndarray) -> None:
        if head_name == "tool":
            self.w_tool, self.w_mem_tool, self.b_tool = w_head, w_mem, bias
        elif head_name == "route":
            self.w_route, self.w_mem_route, self.b_route = w_head, w_mem, bias
        else:
            self.w_final, self.w_mem_final, self.b_final = w_head, w_mem, bias

    def step_hidden(self, x: np.ndarray, prev_h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pre = x @ self.w_enc + prev_h @ self.w_rec
        h = np.tanh(pre).astype(np.float32)
        return pre.astype(np.float32), h

    def update_memory(self, prev_memory: np.ndarray, h: np.ndarray) -> np.ndarray:
        if not self.use_slow_memory:
            return np.zeros_like(h)
        return (self.memory_decay * prev_memory + (1.0 - self.memory_decay) * h).astype(np.float32)

    def logits(self, head_name: str, h: np.ndarray, memory: np.ndarray) -> np.ndarray:
        w_head, w_mem, bias = self._head_params(head_name)
        if self.use_slow_memory:
            return (h @ w_head + memory @ w_mem + bias).astype(np.float32)
        return (h @ w_head + bias).astype(np.float32)

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
        prev_memory = np.zeros(self.hidden_dim, dtype=np.float32)
        traces = []

        for state, (head_name, target_idx) in zip(states, head_targets):
            pre, h = self.step_hidden(state, prev_h)
            memory = self.update_memory(prev_memory, h)
            logits = self.logits(head_name, h, memory)
            probs = base.softmax(logits)
            w_head, w_mem, _bias = self._head_params(head_name)
            target_vec = base.one_hot(int(target_idx), logits.shape[0])
            err = (target_vec - probs).astype(np.float32)
            traces.append(
                {
                    "head": head_name,
                    "state": state,
                    "prev_h": prev_h.copy(),
                    "h": h,
                    "memory": memory,
                    "err": err,
                    "w_head": w_head.copy(),
                    "w_mem": w_mem.copy(),
                }
            )
            prev_h = h
            prev_memory = memory

        for trace in traces:
            head_name = trace["head"]
            w_head, w_mem, bias = self._head_params(head_name)
            h = trace["h"]
            memory = trace["memory"]
            err = trace["err"]
            w_head = w_head + self.lr_head * np.outer(h, err).astype(np.float32)
            if self.use_slow_memory:
                w_mem = w_mem + self.lr_head * np.outer(memory, err).astype(np.float32)
            bias = bias + self.lr_head * err.astype(np.float32)
            self._set_head_params(head_name, w_head, w_mem, bias)

            if self.use_trace:
                grad_h = (w_head @ err).astype(np.float32)
                if self.use_slow_memory:
                    grad_h = grad_h + (1.0 - self.memory_decay) * (w_mem @ err).astype(np.float32)
                grad_pre = (grad_h * (1.0 - np.square(h))).astype(np.float32)
                grad_enc = np.outer(trace["state"], grad_pre).astype(np.float32)
                grad_rec = np.outer(trace["prev_h"], grad_pre).astype(np.float32)
                self.w_enc = self._apply_regularized_update(self.w_enc, grad_enc, self.enc_ref, self.enc_importance, self.lr_enc)
                self.w_rec = self._apply_regularized_update(self.w_rec, grad_rec, self.rec_ref, self.rec_importance, self.lr_rec)

    def predict_episode(self, states: List[np.ndarray], head_targets: List[Tuple[str, int]]) -> List[Tuple[str, int, int]]:
        prev_h = np.zeros(self.hidden_dim, dtype=np.float32)
        prev_memory = np.zeros(self.hidden_dim, dtype=np.float32)
        rows = []
        for state, (head_name, target_idx) in zip(states, head_targets):
            _pre, h = self.step_hidden(state, prev_h)
            memory = self.update_memory(prev_memory, h)
            probs = base.softmax(self.logits(head_name, h, memory))
            rows.append((head_name, int(np.argmax(probs)), int(target_idx)))
            prev_h = h
            prev_memory = memory
        return rows

    def consolidate(self, episodes: List[Tuple[List[np.ndarray], List[Tuple[str, int]]]]) -> None:
        enc_imp = np.zeros_like(self.w_enc)
        rec_imp = np.zeros_like(self.w_rec)
        count = 0
        for states, head_targets in episodes:
            prev_h = np.zeros(self.hidden_dim, dtype=np.float32)
            prev_memory = np.zeros(self.hidden_dim, dtype=np.float32)
            for state, (head_name, target_idx) in zip(states, head_targets):
                pre, h = self.step_hidden(state, prev_h)
                memory = self.update_memory(prev_memory, h)
                w_head, w_mem, _bias = self._head_params(head_name)
                probs = base.softmax(self.logits(head_name, h, memory))
                target_vec = base.one_hot(int(target_idx), probs.shape[0])
                err = (target_vec - probs).astype(np.float32)
                grad_h = (w_head @ err).astype(np.float32)
                if self.use_slow_memory:
                    grad_h = grad_h + (1.0 - self.memory_decay) * (w_mem @ err).astype(np.float32)
                grad_pre = (grad_h * (1.0 - np.square(h))).astype(np.float32)
                enc_imp += np.abs(np.outer(state, grad_pre)).astype(np.float32)
                rec_imp += np.abs(np.outer(prev_h, grad_pre)).astype(np.float32)
                prev_h = h
                prev_memory = memory
                count += 1
        if count > 0:
            enc_imp /= float(count)
            rec_imp /= float(count)
        self.enc_ref = self.w_enc.copy()
        self.rec_ref = self.w_rec.copy()
        self.enc_importance = enc_imp / (float(np.mean(enc_imp)) + 1e-6)
        self.rec_importance = rec_imp / (float(np.mean(rec_imp)) + 1e-6)


def build_episode(
    rng: np.random.Generator,
    concept: str,
    length: int,
    noise: float,
    dropout_p: float,
) -> Tuple[List[np.ndarray], List[Tuple[str, int]]]:
    family = base.concept_family(concept)
    actions = base.FAMILY_TO_ACTIONS[family]
    states = [base.sample_concept_state(rng, concept, noise, dropout_p)]
    targets: List[Tuple[str, int]] = [("tool", base.TOOL_INDEX[actions["tool"]])]
    for _step in range(1, max(1, length - 1)):
        states.append(base.sample_transition_state(rng, step=1, noise=noise))
        targets.append(("route", base.ROUTE_INDEX[actions["route"]]))
    states.append(base.sample_transition_state(rng, step=2, noise=noise))
    targets.append(("final", base.FINAL_INDEX[actions["final"]]))
    return states, targets


def episode_pool(
    families: List[str],
    length: int,
    repeats: int,
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
) -> List[Tuple[List[np.ndarray], List[Tuple[str, int]]]]:
    rows = []
    for _ in range(repeats):
        for family in families:
            for concept in base.CONCEPTS[family]:
                rows.append(build_episode(rng, concept, length, noise, dropout_p))
    rng.shuffle(rows)
    return rows


def evaluate_system(
    model: MemoryLearner,
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
    for _ in range(repeats):
        for family in families:
            for concept in base.CONCEPTS[family]:
                states, targets = build_episode(rng, concept, length, noise, dropout_p)
                preds = model.predict_episode(states, targets)
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
                episode_correct += int(all(step_ok))
                total += 1
    return {
        "tool_accuracy": float(tool_correct / max(1, total)),
        "route_accuracy": float(route_correct / max(1, route_total)),
        "final_accuracy": float(final_correct / max(1, total)),
        "episode_success": float(episode_correct / max(1, total)),
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


def closure_score(summary: Dict[str, Dict[str, float]]) -> float:
    def v(key: str) -> float:
        return float(summary[key]["mean"])
    return float(
        0.20 * v("overall_tool_accuracy")
        + 0.20 * v("overall_route_accuracy")
        + 0.20 * v("overall_final_accuracy")
        + 0.25 * v("overall_episode_success")
        + 0.15 * v("retention_after_phase2")
    )


def fit_slope(lengths: List[int], values: List[float]) -> float:
    if len(lengths) < 2:
        return 0.0
    x = np.asarray(lengths, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    a = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(a, y, rcond=None)[0]
    return float(slope)


def relative_drop(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(values[0] - values[-1])


def run_for_length(
    system_name: str,
    length: int,
    seed: int,
    noise: float,
    dropout_p: float,
) -> Dict[str, float]:
    cfg = system_configs()[system_name]
    rng = np.random.default_rng(seed)
    model = MemoryLearner(
        input_dim=13,
        hidden_dim=6,
        lr_head=0.13,
        lr_enc=0.045,
        lr_rec=0.055,
        use_trace=bool(cfg["use_trace"]),
        use_slow_memory=bool(cfg["use_slow_memory"]),
        memory_decay=float(cfg["memory_decay"]),
        stability=float(cfg["stability"]),
        seed=seed,
    )

    phase1_families = ["fruit", "animal"]
    phase2_families = ["abstract"]
    all_families = ["fruit", "animal", "abstract"]

    phase1_memory = episode_pool(phase1_families, length, repeats=8, rng=rng, noise=noise, dropout_p=dropout_p)
    phase2_memory = episode_pool(phase2_families, length, repeats=10, rng=rng, noise=noise, dropout_p=dropout_p)

    for states, targets in episode_pool(phase1_families, length, repeats=44, rng=rng, noise=noise, dropout_p=dropout_p):
        model.train_episode(states, targets)
    phase1_eval = evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18)

    if bool(cfg["use_trace"]):
        model.consolidate(phase1_memory)

    phase2_rows = episode_pool(phase2_families, length, repeats=62, rng=rng, noise=noise, dropout_p=dropout_p)
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
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Memory boost scan for real multi-step AGI closure")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.34)
    ap.add_argument("--dropout-p", type=float, default=0.14)
    ap.add_argument("--lengths", type=int, nargs="+", default=[3, 4, 5, 6, 8, 10, 12])
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/real_multistep_agi_closure_memory_boost_scan_20260308.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    systems: Dict[str, Dict[str, object]] = {}
    lengths = [int(x) for x in args.lengths]
    for system_name in system_configs().keys():
        per_length = {}
        closure_curve = []
        retention_curve = []
        for length in lengths:
            runs = []
            for offset in range(int(args.num_seeds)):
                seed = int(args.seed) + offset
                runs.append(run_for_length(system_name, length, seed, float(args.noise), float(args.dropout_p)))
            summary = summarize_runs(runs)
            score = closure_score(summary)
            per_length[str(length)] = {
                "summary": summary,
                "real_closure_score": score,
            }
            closure_curve.append(score)
            retention_curve.append(float(summary["retention_after_phase2"]["mean"]))

        systems[system_name] = {
            "per_length": per_length,
            "global_summary": {
                "lengths": lengths,
                "closure_curve": closure_curve,
                "retention_curve": retention_curve,
                "closure_decay_slope": fit_slope(lengths, closure_curve),
                "retention_decay_slope": fit_slope(lengths, retention_curve),
                "closure_relative_drop": relative_drop(closure_curve),
                "retention_relative_drop": relative_drop(retention_curve),
                "mean_closure_score": float(np.mean(closure_curve)),
            },
        }

    anchor_vs_trace = {}
    anchor_advantage_area = 0.0
    max_length = max(lengths)
    for length in lengths:
        anchor_row = systems["trace_anchor_local"]["per_length"][str(length)]
        trace_row = systems["trace_gated_local"]["per_length"][str(length)]
        plain_row = systems["plain_local"]["per_length"][str(length)]
        gain_vs_trace = float(anchor_row["real_closure_score"] - trace_row["real_closure_score"])
        anchor_advantage_area += gain_vs_trace
        anchor_vs_trace[str(length)] = {
            "closure_gain_vs_trace": gain_vs_trace,
            "closure_gain_vs_plain": float(anchor_row["real_closure_score"] - plain_row["real_closure_score"]),
            "retention_gain_vs_trace": float(
                anchor_row["summary"]["retention_after_phase2"]["mean"] - trace_row["summary"]["retention_after_phase2"]["mean"]
            ),
        }

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
        "anchor_vs_trace_by_length": anchor_vs_trace,
        "global_comparison": {
            "trace_advantage_area_over_plain": float(
                sum(
                    systems["trace_gated_local"]["per_length"][str(length)]["real_closure_score"]
                    - systems["plain_local"]["per_length"][str(length)]["real_closure_score"]
                    for length in lengths
                )
            ),
            "anchor_advantage_area_over_trace": float(anchor_advantage_area),
            "anchor_final_length_gain_vs_trace": float(anchor_vs_trace[str(max_length)]["closure_gain_vs_trace"]),
            "trace_final_length_gain_vs_plain": float(
                systems["trace_gated_local"]["per_length"][str(max_length)]["real_closure_score"]
                - systems["plain_local"]["per_length"][str(max_length)]["real_closure_score"]
            ),
        },
        "hypotheses": {
            "H1_anchor_beats_trace_on_average": bool(
                systems["trace_anchor_local"]["global_summary"]["mean_closure_score"]
                > systems["trace_gated_local"]["global_summary"]["mean_closure_score"]
            ),
            "H2_anchor_beats_trace_at_max_length": bool(anchor_vs_trace[str(max_length)]["closure_gain_vs_trace"] > 0.03),
            "H3_anchor_retention_beats_trace_at_max_length": bool(anchor_vs_trace[str(max_length)]["retention_gain_vs_trace"] > 0.05),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["global_comparison"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
