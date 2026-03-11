from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import test_continuous_multimodal_grounding_proto as cmg
import test_stage_c7_consensus_discriminator_temporal_binding_search as c7


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"
MODALITY_SLICES = {
    "visual": slice(0, 8),
    "tactile": slice(8, 16),
    "language": slice(16, 24),
}


class PersistentSlotBinderGrounder(c7.ConsensusDiscriminatorTemporalGrounder):
    def __init__(
        self,
        direct_weight: float,
        modality_weight: float,
        canonical_weight: float,
        trace_weight: float,
        family_weight: float,
        trace_pull: float,
        slot_weight: float,
        binder_weight: float,
        arbitration_weight: float,
        slot_pull: float,
    ) -> None:
        super().__init__(
            direct_weight=direct_weight,
            modality_weight=modality_weight,
            canonical_weight=canonical_weight,
            trace_weight=trace_weight,
            family_weight=family_weight,
            trace_pull=trace_pull,
        )
        self.slot_weight = slot_weight
        self.binder_weight = binder_weight
        self.arbitration_weight = arbitration_weight
        self.slot_pull = slot_pull
        self.object_slot: Dict[str, np.ndarray] = {}
        self.identity_binder: Dict[str, np.ndarray] = {}
        self.arbitration_memory: Dict[str, np.ndarray] = {}

    def _refresh_high_level_memories(self, concept: str) -> None:
        full_proto = self.full_concept_proto.get(concept)
        canonical = self.canonical_concept.get(concept)
        if full_proto is None or canonical is None:
            return
        prev_slot = self.object_slot.get(concept)
        if prev_slot is None:
            self.object_slot[concept] = full_proto.copy()
        else:
            self.object_slot[concept] = (
                (1.0 - self.slot_pull) * prev_slot + self.slot_pull * full_proto
            ).astype(np.float32)

        binder_target = np.concatenate(
            [
                full_proto,
                np.tile(canonical, 3),
            ],
            axis=0,
        ).astype(np.float32)
        prev_binder = self.identity_binder.get(concept)
        if prev_binder is None:
            self.identity_binder[concept] = binder_target
        else:
            self.identity_binder[concept] = (
                (1.0 - self.slot_pull) * prev_binder + self.slot_pull * binder_target
            ).astype(np.float32)

        modality_parts = []
        for modality in MODALITY_SLICES:
            proto = self.modality_concept_proto[modality].get(concept)
            if proto is not None:
                modality_parts.append(proto)
            else:
                modality_parts.append(canonical)
        arbitration_target = np.concatenate(modality_parts, axis=0).astype(np.float32)
        prev_arb = self.arbitration_memory.get(concept)
        if prev_arb is None:
            self.arbitration_memory[concept] = arbitration_target
        else:
            self.arbitration_memory[concept] = (
                (1.0 - self.slot_pull) * prev_arb + self.slot_pull * arbitration_target
            ).astype(np.float32)

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        super().train(x, family, concept)
        self._refresh_high_level_memories(concept)

    def _arbitration_observation(self, observed: List[Tuple[str, np.ndarray]], canonical: np.ndarray) -> np.ndarray:
        parts: List[np.ndarray] = []
        for modality in MODALITY_SLICES:
            hit = None
            for name, raw_part in observed:
                if name == modality:
                    hit = self.norms[modality].normalize(raw_part)
                    break
            if hit is None:
                hit = canonical
            parts.append(hit.astype(np.float32))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _concept_score(self, x: np.ndarray, observed: List[Tuple[str, np.ndarray]], family: str, concept: str, family_score: float) -> float:
        score = super()._concept_score(x, observed, family, concept, family_score)
        slot = self.object_slot.get(concept)
        if slot is not None:
            score += self.slot_weight * c7.sq_dist(x, slot)
        canonical = self.canonical_concept.get(concept)
        binder = self.identity_binder.get(concept)
        arbitration = self.arbitration_memory.get(concept)
        if canonical is not None and binder is not None:
            obs_binder = np.concatenate([x, np.tile(canonical, 3)], axis=0).astype(np.float32)
            score += self.binder_weight * c7.sq_dist(obs_binder, binder)
        if canonical is not None and arbitration is not None:
            obs_arbitration = self._arbitration_observation(observed, canonical)
            score += self.arbitration_weight * c7.sq_dist(obs_arbitration, arbitration)
        return float(score)


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(rows[0].keys())
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def objective(row: Dict[str, float]) -> float:
    return float(
        3.0 * row["crossmodal_consistency"]
        + 1.4 * row["retention_concept_accuracy"]
        + 1.0 * row["overall_concept_accuracy"]
        + 1.0 * row["temporal_binding_score"]
        + 0.4 * row["grounding_score"]
    )


def evaluate_model(
    model,
    concept_groups: Dict[str, List[str]],
    repeats: int,
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
) -> Dict[str, float]:
    family_ok = 0
    concept_ok = 0
    total = 0
    for _ in range(repeats):
        for family, concepts in concept_groups.items():
            for concept in concepts:
                x = cmg.sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p)
                pred_family, pred_concept = model.predict(x)
                family_ok += int(pred_family == family)
                concept_ok += int(pred_concept == concept)
                total += 1
    return {
        "family_accuracy": float(family_ok / max(1, total)),
        "concept_accuracy": float(concept_ok / max(1, total)),
    }


def run_system(
    seed: int,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    direct_weight: float,
    modality_weight: float,
    canonical_weight: float,
    trace_weight: float,
    family_weight: float,
    trace_pull: float,
    replay_steps: int,
    replay_boost: float,
    replay_after_novel_steps: int,
    slot_weight: float,
    binder_weight: float,
    arbitration_weight: float,
    slot_pull: float,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    model = PersistentSlotBinderGrounder(
        direct_weight=direct_weight,
        modality_weight=modality_weight,
        canonical_weight=canonical_weight,
        trace_weight=trace_weight,
        family_weight=family_weight,
        trace_pull=trace_pull,
        slot_weight=slot_weight,
        binder_weight=binder_weight,
        arbitration_weight=arbitration_weight,
        slot_pull=slot_pull,
    )

    phase1_memory: List[Tuple[np.ndarray, str, str]] = []

    for _ in range(42):
        for family, concepts in cmg.PHASE1.items():
            for concept in concepts:
                x = cmg.sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p)
                model.train(x, family, concept)
                if len(phase1_memory) < 256:
                    phase1_memory.append((x.copy(), family, concept))

    phase1_eval = evaluate_model(model, cmg.PHASE1, repeats=22, rng=rng, noise=noise, dropout_p=dropout_p, missing_modality_p=missing_modality_p)

    for _ in range(3):
        for family, concepts in cmg.PHASE2.items():
            for concept in concepts:
                x = cmg.sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p)
                model.train(x, family, concept)
                for _ in range(replay_after_novel_steps):
                    old_x, old_family, old_concept = py_rng.choice(phase1_memory)
                    replay_x = ((1.0 - replay_boost) * old_x + replay_boost * cmg.sample_multimodal_input(rng, old_concept, noise, dropout_p, missing_modality_p)).astype(np.float32)
                    model.train(replay_x, old_family, old_concept)

    novel_eval = evaluate_model(model, cmg.PHASE2, repeats=24, rng=rng, noise=noise, dropout_p=dropout_p, missing_modality_p=missing_modality_p)

    for _ in range(18):
        for family, concepts in cmg.PHASE2.items():
            for concept in concepts:
                x = cmg.sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p)
                model.train(x, family, concept)
                for _ in range(replay_steps):
                    old_x, old_family, old_concept = py_rng.choice(phase1_memory)
                    replay_x = ((1.0 - replay_boost) * old_x + replay_boost * cmg.sample_multimodal_input(rng, old_concept, noise, dropout_p, missing_modality_p)).astype(np.float32)
                    model.train(replay_x, old_family, old_concept)

    retention_eval = evaluate_model(model, cmg.PHASE1, repeats=22, rng=rng, noise=noise, dropout_p=dropout_p, missing_modality_p=missing_modality_p)
    overall_eval = evaluate_model(
        model,
        {family: cmg.PHASE1[family] + cmg.PHASE2[family] for family in cmg.FAMILIES},
        repeats=22,
        rng=rng,
        noise=noise,
        dropout_p=dropout_p,
        missing_modality_p=missing_modality_p,
    )
    concepts = [concept for family in cmg.FAMILIES for concept in cmg.PHASE1[family] + cmg.PHASE2[family]]
    consistency = c7.crossmodal_consistency(model, concepts, repeats=10, rng=rng, noise=noise, dropout_p=dropout_p)
    temporal_score = c7.temporal_binding_score(model, concepts, repeats=10, rng=rng, noise=noise, dropout_p=dropout_p)

    grounding_score = float(
        (
            phase1_eval["family_accuracy"]
            + novel_eval["family_accuracy"]
            + overall_eval["family_accuracy"]
            + 1.8 * novel_eval["concept_accuracy"]
            + 1.3 * retention_eval["concept_accuracy"]
            + 1.5 * overall_eval["concept_accuracy"]
            + 1.0 * consistency
        )
        / 8.6
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
        "crossmodal_consistency": consistency,
        "temporal_binding_score": temporal_score,
        "grounding_score": grounding_score,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage C11 persistent slot binder arbitration search")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.22)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_c11_persistent_slot_binder_arbitration_search_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage_c10 = json.loads((TEMP_DIR / "stage_c10_object_persistence_sequence_consensus_search_20260311.json").read_text(encoding="utf-8"))
    direct = json.loads((TEMP_DIR / "stage_c8_retention_compatible_direct_consensus_search_20260311.json").read_text(encoding="utf-8"))["baselines"]["direct_multimodal"]
    base = stage_c10["best_retention_compatible_candidate"]["config"]
    strong_multimodal_target = float(stage_c10["headline_metrics"]["strong_multimodal_target"])
    baseline_consistency_ceiling = float(stage_c10["headline_metrics"]["baseline_consistency_ceiling"])

    best_objective = None
    best_consistency = None
    best_retention_compatible = None

    for slot_weight in [0.0, 0.05, 0.1]:
        for binder_weight in [0.0, 0.02, 0.05]:
            for arbitration_weight in [0.0, 0.02, 0.05]:
                for slot_pull in [0.0, 0.1, 0.2]:
                    rows = []
                    for offset in range(int(args.num_seeds)):
                        rows.append(
                            run_system(
                                seed=int(args.seed) + offset,
                                noise=float(args.noise),
                                dropout_p=float(args.dropout_p),
                                missing_modality_p=float(args.missing_modality_p),
                                direct_weight=float(base["direct_weight"]),
                                modality_weight=float(base["modality_weight"]),
                                canonical_weight=float(base["canonical_weight"]),
                                trace_weight=float(base["trace_weight"]),
                                family_weight=float(base["family_weight"]),
                                trace_pull=float(base["trace_pull"]),
                                replay_steps=int(base["replay_steps"]),
                                replay_boost=float(base["replay_boost"]),
                                replay_after_novel_steps=int(base["replay_after_novel_steps"]),
                                slot_weight=slot_weight,
                                binder_weight=binder_weight,
                                arbitration_weight=arbitration_weight,
                                slot_pull=slot_pull,
                            )
                        )
                    summary = summarize(rows)
                    candidate = {
                        "config": {
                            **base,
                            "slot_weight": slot_weight,
                            "binder_weight": binder_weight,
                            "arbitration_weight": arbitration_weight,
                            "slot_pull": slot_pull,
                        },
                        "summary": summary,
                        "objective": objective(summary),
                    }
                    if best_objective is None or candidate["objective"] > best_objective["objective"]:
                        best_objective = candidate
                    if best_consistency is None or summary["crossmodal_consistency"] > best_consistency["summary"]["crossmodal_consistency"]:
                        best_consistency = candidate
                    if (
                        summary["crossmodal_consistency"] >= baseline_consistency_ceiling
                        and summary["retention_concept_accuracy"] >= float(direct["retention_concept_accuracy"])
                        and summary["overall_concept_accuracy"] >= float(direct["overall_concept_accuracy"])
                    ):
                        if best_retention_compatible is None or candidate["objective"] > best_retention_compatible["objective"]:
                            best_retention_compatible = candidate

    assert best_objective is not None
    assert best_consistency is not None

    if best_retention_compatible is None:
        status = "stage_c11_new_block_not_enough"
        core_answer = (
            "Persistent object slot, identity binder, and arbitration memory are still not enough to create a stronger retention-compatible direct closure."
        )
        main_open_gap = "needs_more_than_slot_binder_arbitration"
        best_compatible_consistency = 0.0
    else:
        best_compatible_consistency = float(best_retention_compatible["summary"]["crossmodal_consistency"])
        status = (
            "stage_c11_strong_direct_closure_found"
            if best_compatible_consistency >= strong_multimodal_target
            else "stage_c11_retention_compatible_family_strengthened"
        )
        core_answer = (
            "Persistent object slot, identity binder, and arbitration memory strengthen the retention-compatible direct family, but still do not reach the strong target."
            if best_compatible_consistency < strong_multimodal_target
            else "A persistent-slot family reaches strong direct closure."
        )
        main_open_gap = (
            "strong_target_gap_remains_after_c11"
            if best_compatible_consistency < strong_multimodal_target
            else "strong_target_reached"
        )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "missing_modality_p": float(args.missing_modality_p),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageC11_persistent_slot_binder_arbitration_search",
        },
        "headline_metrics": {
            "strong_multimodal_target": strong_multimodal_target,
            "baseline_consistency_ceiling": baseline_consistency_ceiling,
            "best_objective_consistency": float(best_objective["summary"]["crossmodal_consistency"]),
            "best_consistency_value": float(best_consistency["summary"]["crossmodal_consistency"]),
            "best_compatible_consistency": best_compatible_consistency,
            "strong_target_gap_after_best_compatible": float(max(0.0, strong_multimodal_target - best_compatible_consistency)),
        },
        "best_objective_candidate": best_objective,
        "best_consistency_candidate": best_consistency,
        "best_retention_compatible_candidate": best_retention_compatible,
        "hypotheses": {
            "H1_c11_improves_retention_compatible_consistency": bool(
                best_retention_compatible is not None
                and best_compatible_consistency > float(stage_c10["best_retention_compatible_candidate"]["summary"]["crossmodal_consistency"])
            ),
            "H2_c11_reaches_strong_target": bool(best_compatible_consistency >= strong_multimodal_target),
        },
        "verdict": {
            "status": status,
            "core_answer": core_answer,
            "main_open_gap": main_open_gap,
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
