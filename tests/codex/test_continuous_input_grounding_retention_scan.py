#!/usr/bin/env python
"""
Focused scan for task block D.

Try replay- and dual-rate variants on the continuous-input grounding task to
see whether we can simultaneously improve novel grounding and retention.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import test_continuous_input_grounding_proto as proto


class SharedOffsetReplayGrounder(proto.SharedOffsetGrounder):
    def __init__(self, dim: int, family_lr_scale: float, offset_lr_scale: float) -> None:
        super().__init__(dim)
        self.family_lr_scale = family_lr_scale
        self.offset_lr_scale = offset_lr_scale

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        prev_family = self.family_basis.get(family)
        prev_family_count = self.family_count.get(family, 0)
        if prev_family is None:
            family_mean = x.astype(np.float32).copy()
        else:
            alpha = self.family_lr_scale / float(prev_family_count + 1)
            family_mean = ((1.0 - alpha) * prev_family + alpha * x).astype(np.float32)
        self.family_basis[family] = family_mean
        self.family_count[family] = prev_family_count + 1

        offset = (x - family_mean).astype(np.float32)
        prev_offset = self.concept_offset.get(concept)
        prev_count = self.concept_count.get(concept, 0)
        if prev_offset is None:
            next_offset = offset
        else:
            alpha = self.offset_lr_scale / float(prev_count + 1)
            next_offset = ((1.0 - alpha) * prev_offset + alpha * offset).astype(np.float32)
        self.concept_offset[concept] = next_offset
        self.concept_count[concept] = prev_count + 1
        self.concept_family[concept] = family


class FamilyGatedPrototypeLearner(proto.DirectPrototypeLearner):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.concept_family: Dict[str, str] = {}

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        super().train(x, family, concept)
        self.concept_family[concept] = family

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        family = min(self.family_proto, key=lambda name: proto.sq_dist(x, self.family_proto[name]))
        candidates = [concept for concept, fam in self.concept_family.items() if fam == family]
        if not candidates:
            candidates = list(self.concept_proto.keys())
        concept = min(candidates, key=lambda name: proto.sq_dist(x, self.concept_proto[name]))
        return family, concept


def replay_memory(rng: np.random.Generator, repeats: int, noise: float, dropout_p: float) -> List[Tuple[np.ndarray, str, str]]:
    rows = []
    for _ in range(repeats):
        for family, concepts in proto.PHASE1.items():
            for concept in concepts:
                rows.append((proto.sample_continuous_input(rng, concept, noise, dropout_p), family, concept))
    rng.shuffle(rows)
    return rows


def run_system(system_name: str, seed: int, noise: float, dropout_p: float) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    if system_name == "direct_prototype":
        model = proto.DirectPrototypeLearner(dim=16)
        replay = None
    elif system_name == "family_gated_prototype":
        model = FamilyGatedPrototypeLearner(dim=16)
        replay = None
    elif system_name == "shared_offset_grounder":
        model = proto.SharedOffsetGrounder(dim=16)
        replay = None
    elif system_name == "shared_offset_replay":
        model = SharedOffsetReplayGrounder(dim=16, family_lr_scale=0.60, offset_lr_scale=1.00)
        replay = replay_memory(rng, repeats=6, noise=noise, dropout_p=dropout_p)
    else:
        model = SharedOffsetReplayGrounder(dim=16, family_lr_scale=0.45, offset_lr_scale=0.90)
        replay = replay_memory(rng, repeats=8, noise=noise, dropout_p=dropout_p)

    for _ in range(42):
        for family, concepts in proto.PHASE1.items():
            for concept in concepts:
                model.train(proto.sample_continuous_input(rng, concept, noise, dropout_p), family, concept)

    phase1_eval = proto.evaluate_model(model, proto.PHASE1, repeats=22, rng=rng, noise=noise, dropout_p=dropout_p)

    replay_rows = replay or []
    replay_idx = 0
    for _ in range(3):
        for family, concepts in proto.PHASE2.items():
            for concept in concepts:
                model.train(proto.sample_continuous_input(rng, concept, noise, dropout_p), family, concept)
                if replay_rows:
                    x_old, fam_old, concept_old = replay_rows[replay_idx % len(replay_rows)]
                    model.train(x_old, fam_old, concept_old)
                    replay_idx += 1

    novel_eval = proto.evaluate_model(model, proto.PHASE2, repeats=24, rng=rng, noise=noise, dropout_p=dropout_p)

    for _ in range(18):
        for family, concepts in proto.PHASE2.items():
            for concept in concepts:
                model.train(proto.sample_continuous_input(rng, concept, noise, dropout_p), family, concept)
                if replay_rows:
                    x_old, fam_old, concept_old = replay_rows[replay_idx % len(replay_rows)]
                    model.train(x_old, fam_old, concept_old)
                    replay_idx += 1

    retention_eval = proto.evaluate_model(model, proto.PHASE1, repeats=22, rng=rng, noise=noise, dropout_p=dropout_p)
    overall_eval = proto.evaluate_model(
        model,
        {family: proto.PHASE1[family] + proto.PHASE2[family] for family in proto.FAMILIES},
        repeats=22,
        rng=rng,
        noise=noise,
        dropout_p=dropout_p,
    )

    grounding_score = float(
        (
            phase1_eval["family_accuracy"]
            + novel_eval["family_accuracy"]
            + overall_eval["family_accuracy"]
            + 2.0 * novel_eval["concept_accuracy"]
            + 1.5 * overall_eval["concept_accuracy"]
            + 1.0 * retention_eval["concept_accuracy"]
        )
        / 7.5
    )
    return {
        "phase1_family_accuracy": phase1_eval["family_accuracy"],
        "phase1_concept_accuracy": phase1_eval["concept_accuracy"],
        "novel_family_accuracy": novel_eval["family_accuracy"],
        "novel_concept_accuracy": novel_eval["concept_accuracy"],
        "retention_family_accuracy": retention_eval["family_accuracy"],
        "retention_concept_accuracy": retention_eval["concept_accuracy"],
        "overall_family_accuracy": overall_eval["family_accuracy"],
        "overall_concept_accuracy": overall_eval["concept_accuracy"],
        "grounding_score": grounding_score,
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(rows[0].keys())
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Continuous-input grounding retention scan")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/continuous_input_grounding_retention_scan_20260309.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    systems = {}
    ranking = []
    names = [
        "direct_prototype",
        "family_gated_prototype",
        "shared_offset_grounder",
        "shared_offset_replay",
        "shared_offset_replay_dualrate",
    ]
    for system_name in names:
        rows = []
        for offset in range(int(args.num_seeds)):
            rows.append(run_system(system_name, int(args.seed) + offset, float(args.noise), float(args.dropout_p)))
        systems[system_name] = summarize(rows)
        ranking.append({"system": system_name, **systems[system_name]})
        print(
            f"[summary] {system_name} grounding={systems[system_name]['grounding_score']:.4f} "
            f"novel={systems[system_name]['novel_concept_accuracy']:.4f} "
            f"retention={systems[system_name]['retention_concept_accuracy']:.4f}"
        )

    ranking.sort(key=lambda row: row["grounding_score"], reverse=True)
    baseline = systems["direct_prototype"]
    best = ranking[0]
    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "runtime_sec": float(time.time() - t0),
        },
        "systems": systems,
        "ranking": ranking,
        "best": best,
        "gains_vs_direct": {
            key: {
                "grounding_score_gain": float(row["grounding_score"] - baseline["grounding_score"]),
                "novel_concept_gain": float(row["novel_concept_accuracy"] - baseline["novel_concept_accuracy"]),
                "retention_concept_gain": float(row["retention_concept_accuracy"] - baseline["retention_concept_accuracy"]),
                "overall_concept_gain": float(row["overall_concept_accuracy"] - baseline["overall_concept_accuracy"]),
            }
            for key, row in systems.items()
            if key != "direct_prototype"
        },
        "hypotheses": {
            "H1_some_grounder_beats_direct_on_grounding_score": bool(best["grounding_score"] > baseline["grounding_score"]),
            "H2_some_grounder_beats_direct_on_novel_and_retention": bool(
                any(
                    row["novel_concept_accuracy"] > baseline["novel_concept_accuracy"]
                    and row["retention_concept_accuracy"] > baseline["retention_concept_accuracy"]
                    for row in ranking
                    if row["system"] != "direct_prototype"
                )
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["best"], ensure_ascii=False, indent=2))
    print(json.dumps(results["hypotheses"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
