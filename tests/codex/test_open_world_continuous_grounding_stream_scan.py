#!/usr/bin/env python
"""
Sweep update rates for the open-world continuous grounding stream benchmark.

Goal:
- identify whether the negative closure result is a structural failure
  or just a bad update-rate regime
- search lightweight family/offset learning-rate regions
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import test_open_world_continuous_grounding_stream as stream_bench


class TunableSharedOffsetStreamGrounder:
    def __init__(self, family_alpha: float, offset_alpha: float) -> None:
        self.family_alpha = float(family_alpha)
        self.offset_alpha = float(offset_alpha)
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
        family_mean = self._mean(family_prev, self.family_count.get(family, 0), x, alpha_override=self.family_alpha if family_prev is not None else None)
        self.family_basis[family] = family_mean
        self.family_count[family] = self.family_count.get(family, 0) + 1

        offset = (x - family_mean).astype(np.float32)
        offset_prev = self.concept_offset.get(concept)
        self.concept_offset[concept] = self._mean(offset_prev, self.concept_count.get(concept, 0), offset, alpha_override=self.offset_alpha if offset_prev is not None else None)
        self.concept_count[concept] = self.concept_count.get(concept, 0) + 1
        self.concept_family[concept] = family

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        family = min(self.family_basis, key=lambda name: stream_bench.sq_dist(x, self.family_basis[name]))
        candidates = [concept for concept, fam in self.concept_family.items() if fam == family]
        if not candidates:
            candidates = list(self.concept_family.keys())
        best_concept = None
        best_dist = float("inf")
        for concept in candidates:
            proto = self.family_basis[family] + self.concept_offset[concept]
            dist = stream_bench.sq_dist(x, proto)
            if dist < best_dist:
                best_dist = dist
                best_concept = concept
        assert best_concept is not None
        return family, best_concept


def run_tunable_system(
    seed: int,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    family_alpha: float,
    offset_alpha: float,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    model = TunableSharedOffsetStreamGrounder(family_alpha=family_alpha, offset_alpha=offset_alpha)
    dim = 24

    stream = stream_bench.build_stream(seed)
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
            model.train(x, item["family"], item["concept"])
            update_count += 1
        else:
            _ = stream_bench.sample_noise_chunk(rng, dim=dim, noise_scale=noise * 1.2)

    phase1_concepts = [concept for family in stream_bench.FAMILIES for concept in stream_bench.PHASE1[family]]
    phase2_concepts = [concept for family in stream_bench.FAMILIES for concept in stream_bench.PHASE2[family]]
    all_concepts = phase1_concepts + phase2_concepts

    stable_old = stream_bench.evaluate_concepts(model, phase1_concepts, 24, rng, noise, dropout_p, missing_modality_p, drift_scale)
    novel = stream_bench.evaluate_concepts(model, phase2_concepts, 28, rng, noise, dropout_p, missing_modality_p, drift_scale)
    drifted = stream_bench.evaluate_concepts(model, all_concepts, 24, rng, noise * 1.1, dropout_p, missing_modality_p + 0.05, drift_scale * 1.8)

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
    ap = argparse.ArgumentParser(description="Sweep update rates in the open-world continuous grounding stream benchmark")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.25)
    ap.add_argument("--drift-scale", type=float, default=0.06)
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/open_world_continuous_grounding_stream_scan_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    baseline_payload = json.loads((Path("tests/codex_temp/open_world_continuous_grounding_stream_20260310.json")).read_text(encoding="utf-8"))
    direct = baseline_payload["systems"]["direct_stream"]
    base_shared = baseline_payload["systems"]["shared_offset_stream"]

    rows = []
    best = None
    for family_alpha in [0.03, 0.05, 0.08, 0.12]:
        for offset_alpha in [0.08, 0.12, 0.18, 0.24, 0.32]:
            seeds = []
            for offset in range(int(args.num_seeds)):
                seeds.append(
                    run_tunable_system(
                        seed=int(args.seed) + offset,
                        noise=float(args.noise),
                        dropout_p=float(args.dropout_p),
                        missing_modality_p=float(args.missing_modality_p),
                        drift_scale=float(args.drift_scale),
                        family_alpha=float(family_alpha),
                        offset_alpha=float(offset_alpha),
                    )
                )
            summary = summarize(seeds)
            row = {
                "family_alpha": float(family_alpha),
                "offset_alpha": float(offset_alpha),
                **summary,
                "closure_score_gain_vs_direct": float(summary["closure_score"] - direct["closure_score"]),
                "closure_score_gain_vs_base_shared": float(summary["closure_score"] - base_shared["closure_score"]),
                "novel_concept_gain_vs_direct": float(summary["novel_concept_accuracy"] - direct["novel_concept_accuracy"]),
                "stable_old_concept_gain_vs_direct": float(summary["stable_old_concept_accuracy"] - direct["stable_old_concept_accuracy"]),
                "drifted_concept_gain_vs_direct": float(summary["drifted_concept_accuracy"] - direct["drifted_concept_accuracy"]),
            }
            rows.append(row)
            objective = -(row["closure_score_gain_vs_direct"]) - 0.5 * row["novel_concept_gain_vs_direct"] - 0.2 * row["drifted_concept_gain_vs_direct"]
            if best is None or objective < best[0]:
                best = (objective, row)

    assert best is not None
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "num_seeds": int(args.num_seeds),
            "config_count": len(rows),
            "source_files": [
                "open_world_continuous_grounding_stream_20260310.json",
            ],
        },
        "baseline_direct_stream": direct,
        "baseline_shared_offset_stream": base_shared,
        "best_config": best[1],
        "rows": rows,
        "project_readout": {
            "summary": "这一版不是继续换结构，而是扫描流式更新律，测试开放环境中的负闭环到底是结构问题，还是更新区间选错了。",
            "next_question": "如果更新律扫描能翻正 closure_score_gain，下一步就该把动作和长期状态接进来。"
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["best_config"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
