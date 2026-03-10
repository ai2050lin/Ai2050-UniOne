#!/usr/bin/env python
"""
Open-world grounding + action loop benchmark.

This extends the continuous grounding stream benchmark with:
- family-conditioned action selection
- confidence-triggered self-correction
- post-stream old-concept pollution check

The goal is to see whether the improved stream update law starts to transfer
from pure perception closure into a minimal perception-action-correction loop.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import test_open_world_continuous_grounding_stream as stream_bench


ACTION_BY_FAMILY = {
    "fruit": "collect_bin_a",
    "animal": "redirect_bin_b",
    "abstract": "archive_bin_c",
}


def action_for_family(family: str) -> str:
    return ACTION_BY_FAMILY[family]


def concept_family(concept: str) -> str:
    return stream_bench.concept_family(concept)


class DirectActionAgent(stream_bench.DirectStreamPrototype):
    def family_candidates(self, x: np.ndarray) -> List[Tuple[str, float]]:
        rows = [(name, stream_bench.sq_dist(x, proto)) for name, proto in self.family_proto.items()]
        rows.sort(key=lambda item: item[1])
        return rows

    def concept_candidates(self, x: np.ndarray, family: str) -> List[Tuple[str, float]]:
        rows = [(name, stream_bench.sq_dist(x, proto)) for name, proto in self.concept_proto.items()]
        rows.sort(key=lambda item: item[1])
        return rows

    def predict_with_margin(self, x: np.ndarray) -> Tuple[str, str, float]:
        fam_rows = self.family_candidates(x)
        family = fam_rows[0][0]
        concept_rows = self.concept_candidates(x, family)
        concept = concept_rows[0][0]
        if len(fam_rows) < 2:
            margin = 0.0
        else:
            margin = float(fam_rows[1][1] - fam_rows[0][1])
        return family, concept, margin


class SharedActionAgent(stream_bench.SharedOffsetStreamGrounder):
    def __init__(self, family_alpha: float, offset_alpha: float) -> None:
        super().__init__()
        self.family_alpha_fixed = float(family_alpha)
        self.offset_alpha_fixed = float(offset_alpha)

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        family_prev = self.family_basis.get(family)
        family_mean = self._mean(
            family_prev,
            self.family_count.get(family, 0),
            x,
            alpha_override=self.family_alpha_fixed if family_prev is not None else None,
        )
        self.family_basis[family] = family_mean
        self.family_count[family] = self.family_count.get(family, 0) + 1

        offset = (x - family_mean).astype(np.float32)
        offset_prev = self.concept_offset.get(concept)
        self.concept_offset[concept] = self._mean(
            offset_prev,
            self.concept_count.get(concept, 0),
            offset,
            alpha_override=self.offset_alpha_fixed if offset_prev is not None else None,
        )
        self.concept_count[concept] = self.concept_count.get(concept, 0) + 1
        self.concept_family[concept] = family

    def family_candidates(self, x: np.ndarray) -> List[Tuple[str, float]]:
        rows = [(name, stream_bench.sq_dist(x, proto)) for name, proto in self.family_basis.items()]
        rows.sort(key=lambda item: item[1])
        return rows

    def concept_candidates(self, x: np.ndarray, family: str) -> List[Tuple[str, float]]:
        rows = []
        for concept, fam in self.concept_family.items():
            if fam != family:
                continue
            proto = self.family_basis[family] + self.concept_offset[concept]
            rows.append((concept, stream_bench.sq_dist(x, proto)))
        rows.sort(key=lambda item: item[1])
        return rows

    def predict_with_margin(self, x: np.ndarray) -> Tuple[str, str, float]:
        fam_rows = self.family_candidates(x)
        family = fam_rows[0][0]
        concept_rows = self.concept_candidates(x, family)
        concept = concept_rows[0][0]
        if len(fam_rows) < 2:
            margin = 0.0
        else:
            margin = float(fam_rows[1][1] - fam_rows[0][1])
        return family, concept, margin


def build_agent(system_name: str):
    if system_name == "direct_action":
        return DirectActionAgent()
    if system_name == "shared_action_base":
        return SharedActionAgent(family_alpha=0.05, offset_alpha=0.12)
    if system_name == "shared_action_tuned":
        return SharedActionAgent(family_alpha=0.05, offset_alpha=0.32)
    raise KeyError(system_name)


def confidence_corrected_action(
    agent,
    rng: np.random.Generator,
    concept: str,
    family: str,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    margin_threshold: float,
) -> Tuple[bool, bool]:
    x = stream_bench.sample_stream_input(rng, concept, noise, dropout_p, missing_modality_p, drift_scale)
    pred_family, _pred_concept, margin = agent.predict_with_margin(x)
    first_ok = action_for_family(pred_family) == action_for_family(family)

    if margin >= margin_threshold:
        return first_ok, first_ok

    x_retry = stream_bench.sample_stream_input(rng, concept, noise * 0.75, dropout_p * 0.8, missing_modality_p * 0.8, drift_scale * 0.6)
    merged = 0.5 * (x + x_retry)
    retry_family, _retry_concept, _retry_margin = agent.predict_with_margin(merged.astype(np.float32))
    corrected_ok = action_for_family(retry_family) == action_for_family(family)
    return first_ok, corrected_ok


def evaluate_old_concepts(
    agent,
    rng: np.random.Generator,
    repeats: int,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
) -> float:
    old_concepts = [concept for family in stream_bench.FAMILIES for concept in stream_bench.PHASE1[family]]
    ok = 0
    total = 0
    for _ in range(repeats):
        for concept in old_concepts:
            family = concept_family(concept)
            x = stream_bench.sample_stream_input(rng, concept, noise, dropout_p, missing_modality_p, drift_scale)
            pred_family, pred_concept, _ = agent.predict_with_margin(x)
            ok += int(pred_family == family and pred_concept == concept)
            total += 1
    return float(ok / max(1, total))


def run_system(
    system_name: str,
    seed: int,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    margin_threshold: float,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    agent = build_agent(system_name)
    dim = 24
    stream = stream_bench.build_stream(seed)

    action_ok = 0
    corrected_action_ok = 0
    action_total = 0
    update_count = 0

    for item in stream:
        if item["kind"] == "concept":
            x = stream_bench.sample_stream_input(
                rng,
                item["concept"],
                noise=noise,
                dropout_p=dropout_p,
                missing_modality_p=missing_modality_p,
                drift_scale=drift_scale,
            )
            agent.train(x, item["family"], item["concept"])
            update_count += 1

            first_ok, corrected_ok = confidence_corrected_action(
                agent=agent,
                rng=rng,
                concept=item["concept"],
                family=item["family"],
                noise=noise,
                dropout_p=dropout_p,
                missing_modality_p=missing_modality_p,
                drift_scale=drift_scale,
                margin_threshold=margin_threshold,
            )
            action_ok += int(first_ok)
            corrected_action_ok += int(corrected_ok)
            action_total += 1
        else:
            _ = stream_bench.sample_noise_chunk(rng, dim=dim, noise_scale=noise * 1.2)

    old_concept_retention = evaluate_old_concepts(
        agent=agent,
        rng=rng,
        repeats=18,
        noise=noise,
        dropout_p=dropout_p,
        missing_modality_p=missing_modality_p,
        drift_scale=drift_scale,
    )
    action_accuracy = float(action_ok / max(1, action_total))
    corrected_action_accuracy = float(corrected_action_ok / max(1, action_total))
    loop_score = float((1.3 * corrected_action_accuracy + 1.0 * action_accuracy + 1.1 * old_concept_retention) / 3.4)

    return {
        "action_accuracy": action_accuracy,
        "corrected_action_accuracy": corrected_action_accuracy,
        "old_concept_retention": old_concept_retention,
        "loop_score": loop_score,
        "update_count": float(update_count),
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(rows[0].keys())
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Open-world grounding + action loop benchmark")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.25)
    ap.add_argument("--drift-scale", type=float, default=0.06)
    ap.add_argument("--margin-threshold", type=float, default=0.035)
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/open_world_grounding_action_loop_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    systems = {}
    for system_name in ["direct_action", "shared_action_base", "shared_action_tuned"]:
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
                    margin_threshold=float(args.margin_threshold),
                )
            )
        systems[system_name] = summarize(rows)

    direct = systems["direct_action"]
    base = systems["shared_action_base"]
    tuned = systems["shared_action_tuned"]
    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "missing_modality_p": float(args.missing_modality_p),
            "drift_scale": float(args.drift_scale),
            "margin_threshold": float(args.margin_threshold),
            "runtime_sec": float(time.time() - t0),
        },
        "systems": systems,
        "gains_vs_direct": {
            "base_loop_score_gain": float(base["loop_score"] - direct["loop_score"]),
            "tuned_loop_score_gain": float(tuned["loop_score"] - direct["loop_score"]),
            "base_corrected_action_gain": float(base["corrected_action_accuracy"] - direct["corrected_action_accuracy"]),
            "tuned_corrected_action_gain": float(tuned["corrected_action_accuracy"] - direct["corrected_action_accuracy"]),
            "tuned_old_concept_retention_gain": float(tuned["old_concept_retention"] - direct["old_concept_retention"]),
        },
        "hypotheses": {
            "H1_tuned_beats_direct_on_loop_score": bool(tuned["loop_score"] > direct["loop_score"]),
            "H2_tuned_beats_direct_on_corrected_action": bool(tuned["corrected_action_accuracy"] > direct["corrected_action_accuracy"]),
            "H3_tuned_beats_base_on_loop_score": bool(tuned["loop_score"] > base["loop_score"]),
        },
        "project_readout": {
            "summary": "这一版把连续流接地接上最小动作回路和自纠错环，开始测试正向更新律增益能否真正传到感知-动作-修正闭环。",
            "next_question": "如果 tuned 流式更新律能在最小动作回路上保持正增益，下一步就该接长期状态和多步目标。"
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["gains_vs_direct"], ensure_ascii=False, indent=2))
    print(json.dumps(results["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
