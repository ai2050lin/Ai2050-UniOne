#!/usr/bin/env python
"""
Length-scan over the real-ish multi-step AGI closure benchmark.

This extends the fixed 3-step benchmark into a family of task graphs with
length L in {3, 4, 5, 6}. The goal is to measure how closure quality decays
with horizon length, and whether trace/stability/replay slow that decay.
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


@dataclass
class LengthScanLearner:
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

    def step_hidden(self, x: np.ndarray, prev_h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pre = x @ self.w_enc + prev_h @ self.w_rec
        h = np.tanh(pre).astype(np.float32)
        return pre.astype(np.float32), h

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

    def _head_params(self, head_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if head_name == "tool":
            return self.w_tool, self.b_tool
        if head_name == "route":
            return self.w_route, self.b_route
        return self.w_final, self.b_final

    def _set_head_params(self, head_name: str, w: np.ndarray, b: np.ndarray) -> None:
        if head_name == "tool":
            self.w_tool, self.b_tool = w, b
        elif head_name == "route":
            self.w_route, self.b_route = w, b
        else:
            self.w_final, self.b_final = w, b

    def train_episode(self, states: List[np.ndarray], head_targets: List[Tuple[str, int]]) -> None:
        prev_h = np.zeros(self.hidden_dim, dtype=np.float32)
        traces = []
        for state, (head_name, target_idx) in zip(states, head_targets):
            head_w, head_b = self._head_params(head_name)
            head_dim = head_w.shape[1]
            pre, h = self.step_hidden(state, prev_h)
            probs = base.softmax(h @ head_w + head_b)
            target_vec = base.one_hot(int(target_idx), head_dim)
            err = (target_vec - probs).astype(np.float32)
            traces.append(
                {
                    "head": head_name,
                    "state": state,
                    "prev_h": prev_h.copy(),
                    "h": h,
                    "err": err,
                }
            )
            prev_h = h

        for trace in traces:
            head_name = trace["head"]
            head_w, head_b = self._head_params(head_name)
            h = trace["h"]
            err = trace["err"]
            grad_head = np.outer(h, err).astype(np.float32)
            grad_b = err.astype(np.float32)
            head_w = head_w + self.lr_head * grad_head
            head_b = head_b + self.lr_head * grad_b
            self._set_head_params(head_name, head_w, head_b)

            if self.use_trace:
                grad_h = (head_w @ err).astype(np.float32)
                grad_pre = (grad_h * (1.0 - np.square(h))).astype(np.float32)
                grad_enc = np.outer(trace["state"], grad_pre).astype(np.float32)
                grad_rec = np.outer(trace["prev_h"], grad_pre).astype(np.float32)
                self.w_enc = self._apply_regularized_update(self.w_enc, grad_enc, self.enc_ref, self.enc_importance, self.lr_enc)
                self.w_rec = self._apply_regularized_update(self.w_rec, grad_rec, self.rec_ref, self.rec_importance, self.lr_rec)

    def predict_episode(self, states: List[np.ndarray], head_targets: List[Tuple[str, int]]) -> List[Tuple[str, int, int]]:
        prev_h = np.zeros(self.hidden_dim, dtype=np.float32)
        rows = []
        for state, (head_name, target_idx) in zip(states, head_targets):
            head_w, head_b = self._head_params(head_name)
            _pre, h = self.step_hidden(state, prev_h)
            probs = base.softmax(h @ head_w + head_b)
            rows.append((head_name, int(np.argmax(probs)), int(target_idx)))
            prev_h = h
        return rows

    def consolidate(self, episodes: List[Tuple[List[np.ndarray], List[Tuple[str, int]]]]) -> None:
        enc_imp = np.zeros_like(self.w_enc)
        rec_imp = np.zeros_like(self.w_rec)
        count = 0
        for states, head_targets in episodes:
            prev_h = np.zeros(self.hidden_dim, dtype=np.float32)
            for state, (head_name, target_idx) in zip(states, head_targets):
                head_w, head_b = self._head_params(head_name)
                pre, h = self.step_hidden(state, prev_h)
                probs = base.softmax(h @ head_w + head_b)
                target_vec = base.one_hot(int(target_idx), head_w.shape[1])
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


def build_length_episode(
    rng: np.random.Generator,
    concept: str,
    length: int,
    noise: float,
    dropout_p: float,
) -> Tuple[List[np.ndarray], List[Tuple[str, int]]]:
    family = base.concept_family(concept)
    actions = base.FAMILY_TO_ACTIONS[family]
    states = [base.sample_concept_state(rng, concept, noise, dropout_p)]
    head_targets: List[Tuple[str, int]] = [("tool", base.TOOL_INDEX[actions["tool"]])]

    for step in range(1, max(1, length - 1)):
        states.append(base.sample_transition_state(rng, step=1, noise=noise))
        head_targets.append(("route", base.ROUTE_INDEX[actions["route"]]))

    states.append(base.sample_transition_state(rng, step=2, noise=noise))
    head_targets.append(("final", base.FINAL_INDEX[actions["final"]]))
    return states, head_targets


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
                rows.append(build_length_episode(rng, concept, length, noise, dropout_p))
    rng.shuffle(rows)
    return rows


def evaluate_system(
    model: LengthScanLearner,
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
                states, head_targets = build_length_episode(rng, concept, length, noise, dropout_p)
                preds = model.predict_episode(states, head_targets)
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


def summarize_runs(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = rows[0].keys()
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
        }
        for key in keys
    }


def run_for_length(length: int, use_trace: bool, seed: int, noise: float, dropout_p: float) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    model = LengthScanLearner(
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
    phase1_memory = episode_pool(phase1_families, length, repeats=8, rng=rng, noise=noise, dropout_p=dropout_p)
    phase2_memory = episode_pool(phase2_families, length, repeats=10, rng=rng, noise=noise, dropout_p=dropout_p)

    for states, head_targets in episode_pool(phase1_families, length, repeats=44, rng=rng, noise=noise, dropout_p=dropout_p):
        model.train_episode(states, head_targets)
    phase1_eval = evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18)

    if use_trace:
        model.consolidate(phase1_memory)

    phase2_rows = episode_pool(phase2_families, length, repeats=62, rng=rng, noise=noise, dropout_p=dropout_p)
    for idx, (states, head_targets) in enumerate(phase2_rows):
        model.train_episode(states, head_targets)
        if use_trace and idx % 2 == 0:
            mem_states, mem_targets = phase1_memory[idx % len(phase1_memory)]
            model.train_episode(mem_states, mem_targets)
        if use_trace and idx % 5 == 0:
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


def fit_decay(lengths: List[int], values: List[float]) -> float:
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
    start = float(values[0])
    end = float(values[-1])
    return float(start - end)


def main() -> None:
    ap = argparse.ArgumentParser(description="Length scan for real multi-step AGI closure benchmark")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.34)
    ap.add_argument("--dropout-p", type=float, default=0.14)
    ap.add_argument("--lengths", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/real_multistep_agi_closure_length_scan_20260308.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    systems: Dict[str, Dict[str, object]] = {}
    trace_curve = []
    plain_curve = []

    for system_name, use_trace in [("plain_local", False), ("trace_gated_local", True)]:
        per_length = {}
        closure_curve = []
        retention_curve = []
        for length in [int(x) for x in args.lengths]:
            runs = []
            for offset in range(int(args.num_seeds)):
                seed = int(args.seed) + offset
                runs.append(run_for_length(length, use_trace, seed, float(args.noise), float(args.dropout_p)))
            summary = summarize_runs(runs)
            score = closure_score(summary)
            per_length[str(length)] = {
                "summary": summary,
                "real_closure_score": score,
            }
            closure_curve.append(score)
            retention_curve.append(float(summary["retention_after_phase2"]["mean"]))
            if use_trace:
                trace_curve.append((length, score))
            else:
                plain_curve.append((length, score))

        lengths = [int(x) for x in args.lengths]
        systems[system_name] = {
            "per_length": per_length,
            "global_summary": {
                "lengths": lengths,
                "closure_curve": closure_curve,
                "retention_curve": retention_curve,
                "closure_decay_slope": fit_decay(lengths, closure_curve),
                "retention_decay_slope": fit_decay(lengths, retention_curve),
                "closure_relative_drop": relative_drop(closure_curve),
                "retention_relative_drop": relative_drop(retention_curve),
                "mean_closure_score": float(np.mean(closure_curve)),
            },
        }

    improvements_by_length = {}
    advantage_area = 0.0
    for length in [int(x) for x in args.lengths]:
        plain_row = systems["plain_local"]["per_length"][str(length)]
        trace_row = systems["trace_gated_local"]["per_length"][str(length)]
        gain = float(trace_row["real_closure_score"] - plain_row["real_closure_score"])
        advantage_area += gain
        improvements_by_length[str(length)] = {
            "real_closure_gain": gain,
            "episode_success_gain": float(
                trace_row["summary"]["overall_episode_success"]["mean"] - plain_row["summary"]["overall_episode_success"]["mean"]
            ),
            "retention_gain": float(
                trace_row["summary"]["retention_after_phase2"]["mean"] - plain_row["summary"]["retention_after_phase2"]["mean"]
            ),
        }

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "lengths": [int(x) for x in args.lengths],
            "runtime_sec": float(time.time() - t0),
        },
        "systems": systems,
        "improvements_by_length": improvements_by_length,
        "global_comparison": {
            "plain_closure_decay_slope": float(systems["plain_local"]["global_summary"]["closure_decay_slope"]),
            "trace_closure_decay_slope": float(systems["trace_gated_local"]["global_summary"]["closure_decay_slope"]),
            "plain_retention_decay_slope": float(systems["plain_local"]["global_summary"]["retention_decay_slope"]),
            "trace_retention_decay_slope": float(systems["trace_gated_local"]["global_summary"]["retention_decay_slope"]),
            "plain_closure_relative_drop": float(systems["plain_local"]["global_summary"]["closure_relative_drop"]),
            "trace_closure_relative_drop": float(systems["trace_gated_local"]["global_summary"]["closure_relative_drop"]),
            "plain_retention_relative_drop": float(systems["plain_local"]["global_summary"]["retention_relative_drop"]),
            "trace_retention_relative_drop": float(systems["trace_gated_local"]["global_summary"]["retention_relative_drop"]),
            "trace_advantage_area": float(advantage_area),
            "final_length_gain": float(improvements_by_length[str(max(int(x) for x in args.lengths))]["real_closure_gain"]),
        },
        "hypotheses": {
            "H1_trace_positive_gain_all_lengths": bool(
                all(float(row["real_closure_gain"]) > 0 for row in improvements_by_length.values())
            ),
            "H2_trace_positive_gain_at_max_length": bool(
                float(improvements_by_length[str(max(int(x) for x in args.lengths))]["real_closure_gain"]) > 0.15
            ),
            "H3_trace_retention_above_floor_at_max_length": bool(
                float(
                    systems["trace_gated_local"]["per_length"][str(max(int(x) for x in args.lengths))]["summary"]["retention_after_phase2"]["mean"]
                )
                > 0.10
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["global_comparison"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
