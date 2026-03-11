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
import test_stage_c15_stronger_long_horizon_binder_search as c15
import test_stage_c30_three_path_architecture_search as c30


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values / max(1e-6, temperature)
    shifted = scaled - np.max(scaled)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


class ThreePathControllerGrounder(c30.ThreePathArchitectureGrounder):
    def __init__(
        self,
        direct_weight: float,
        modality_weight: float,
        canonical_weight: float,
        trace_weight: float,
        family_weight: float,
        trace_pull: float,
        slot_weight: float,
        write_gate: float,
        read_pull: float,
        persistence_pull: float,
        binder_pull: float,
        binder_weight: float,
        graph_pull: float,
        graph_weight: float,
        family_graph_pull: float,
        family_graph_weight: float,
        anchor_pull: float,
        anchor_weight: float,
        negative_margin_weight: float,
        family_center_weight: float,
        manifold_pull: float,
        manifold_weight: float,
        sequence_vote_weight: float,
        contrastive_pull: float,
        same_object_weight: float,
        negative_push_weight: float,
        pair_pull: float,
        pair_same_weight: float,
        pair_negative_weight: float,
        retention_pull: float,
        retention_weight: float,
        identity_pull: float,
        identity_mix_weight: float,
        concept_pull: float,
        concept_mix_weight: float,
        fusion_identity_weight: float,
        fusion_memory_weight: float,
        fusion_novelty_weight: float,
        controller_temperature: float,
        identity_gain: float,
        memory_guard: float,
        novelty_gain: float,
        routing_sharpness: float,
    ) -> None:
        super().__init__(
            direct_weight=direct_weight,
            modality_weight=modality_weight,
            canonical_weight=canonical_weight,
            trace_weight=trace_weight,
            family_weight=family_weight,
            trace_pull=trace_pull,
            slot_weight=slot_weight,
            write_gate=write_gate,
            read_pull=read_pull,
            persistence_pull=persistence_pull,
            binder_pull=binder_pull,
            binder_weight=binder_weight,
            graph_pull=graph_pull,
            graph_weight=graph_weight,
            family_graph_pull=family_graph_pull,
            family_graph_weight=family_graph_weight,
            anchor_pull=anchor_pull,
            anchor_weight=anchor_weight,
            negative_margin_weight=negative_margin_weight,
            family_center_weight=family_center_weight,
            manifold_pull=manifold_pull,
            manifold_weight=manifold_weight,
            sequence_vote_weight=sequence_vote_weight,
            contrastive_pull=contrastive_pull,
            same_object_weight=same_object_weight,
            negative_push_weight=negative_push_weight,
            pair_pull=pair_pull,
            pair_same_weight=pair_same_weight,
            pair_negative_weight=pair_negative_weight,
            retention_pull=retention_pull,
            retention_weight=retention_weight,
            identity_pull=identity_pull,
            identity_mix_weight=identity_mix_weight,
            concept_pull=concept_pull,
            concept_mix_weight=concept_mix_weight,
            fusion_identity_weight=fusion_identity_weight,
            fusion_memory_weight=fusion_memory_weight,
            fusion_novelty_weight=fusion_novelty_weight,
            controller_temperature=controller_temperature,
        )
        self.identity_gain = identity_gain
        self.memory_guard = memory_guard
        self.novelty_gain = novelty_gain
        self.routing_sharpness = routing_sharpness

    def _controller_logits(self, x: np.ndarray, concept: str) -> np.ndarray:
        id_e, mem_e, nov_e = self._path_energies(x, concept)
        pair = self.pair_state.get(concept)
        retention = self.retention_state.get(concept)
        manifold = self.novelty_path.get(concept)

        identity_conf = 0.0 if pair is None else 1.0 / max(1e-4, c7.sq_dist(x, pair))
        memory_stability = 0.0 if retention is None else 1.0 / max(1e-4, c7.sq_dist(x, retention))
        novelty_pressure = 0.0 if manifold is None else 1.0 / max(1e-4, c7.sq_dist(x, manifold))

        logits = np.array(
            [
                self.identity_gain * identity_conf - 0.25 * id_e,
                self.memory_guard * memory_stability - 0.25 * mem_e,
                self.novelty_gain * novelty_pressure - 0.25 * nov_e,
            ],
            dtype=np.float64,
        )
        return logits

    def _fusion_penalty(self, x: np.ndarray, concept: str) -> float:
        id_e, mem_e, nov_e = self._path_energies(x, concept)
        logits = self._controller_logits(x, concept)
        gates = softmax(self.routing_sharpness * logits, self.controller_temperature)
        explicit = gates[0] * id_e + gates[1] * mem_e + gates[2] * nov_e
        base = (
            self.fusion_identity_weight * id_e
            + self.fusion_memory_weight * mem_e
            + self.fusion_novelty_weight * nov_e
        )
        return float(base + explicit)


def run_system(seed: int, noise: float, dropout_p: float, missing_modality_p: float, config: Dict[str, float]) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    model = ThreePathControllerGrounder(
        direct_weight=float(config["direct_weight"]),
        modality_weight=float(config["modality_weight"]),
        canonical_weight=float(config["canonical_weight"]),
        trace_weight=float(config["trace_weight"]),
        family_weight=float(config["family_weight"]),
        trace_pull=float(config["trace_pull"]),
        slot_weight=float(config.get("slot_weight", config["object_token_weight"])),
        write_gate=float(config["write_gate"]),
        read_pull=float(config["read_pull"]),
        persistence_pull=float(config["persistence_pull"]),
        binder_pull=float(config["binder_pull"]),
        binder_weight=float(config["binder_weight"]),
        graph_pull=float(config["graph_pull"]),
        graph_weight=float(config["graph_weight"]),
        family_graph_pull=float(config["family_graph_pull"]),
        family_graph_weight=float(config["family_graph_weight"]),
        anchor_pull=float(config["anchor_pull"]),
        anchor_weight=float(config["anchor_weight"]),
        negative_margin_weight=float(config["negative_margin_weight"]),
        family_center_weight=float(config["family_center_weight"]),
        manifold_pull=float(config["manifold_pull"]),
        manifold_weight=float(config["manifold_weight"]),
        sequence_vote_weight=float(config["sequence_vote_weight"]),
        contrastive_pull=float(config["contrastive_pull"]),
        same_object_weight=float(config["same_object_weight"]),
        negative_push_weight=float(config["negative_push_weight"]),
        pair_pull=float(config["pair_pull"]),
        pair_same_weight=float(config["pair_same_weight"]),
        pair_negative_weight=float(config["pair_negative_weight"]),
        retention_pull=float(config["retention_pull"]),
        retention_weight=float(config["retention_weight"]),
        identity_pull=float(config["identity_pull"]),
        identity_mix_weight=float(config["identity_mix_weight"]),
        concept_pull=float(config["concept_pull"]),
        concept_mix_weight=float(config["concept_mix_weight"]),
        fusion_identity_weight=float(config["fusion_identity_weight"]),
        fusion_memory_weight=float(config["fusion_memory_weight"]),
        fusion_novelty_weight=float(config["fusion_novelty_weight"]),
        controller_temperature=float(config["controller_temperature"]),
        identity_gain=float(config["identity_gain"]),
        memory_guard=float(config["memory_guard"]),
        novelty_gain=float(config["novelty_gain"]),
        routing_sharpness=float(config["routing_sharpness"]),
    )

    phase1_memory: List[Tuple[np.ndarray, str, str]] = []

    for _ in range(42):
        for family, concepts in cmg.PHASE1.items():
            for concept in concepts:
                x = cmg.sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p)
                model.train(x, family, concept)
                if len(phase1_memory) < 256:
                    phase1_memory.append((x.copy(), family, concept))

    phase1_eval = c15.evaluate_model(model, cmg.PHASE1, 22, rng, noise, dropout_p, missing_modality_p)

    for _ in range(3):
        for family, concepts in cmg.PHASE2.items():
            for concept in concepts:
                x = cmg.sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p)
                model.train(x, family, concept)
                for _ in range(int(config["replay_after_novel_steps"])):
                    old_x, old_family, old_concept = py_rng.choice(phase1_memory)
                    replay_x = (
                        (1.0 - float(config["replay_boost"])) * old_x
                        + float(config["replay_boost"]) * cmg.sample_multimodal_input(
                            rng, old_concept, noise, dropout_p, missing_modality_p
                        )
                    ).astype(np.float32)
                    model.train(replay_x, old_family, old_concept)

    novel_eval = c15.evaluate_model(model, cmg.PHASE2, 24, rng, noise, dropout_p, missing_modality_p)

    for _ in range(18):
        for family, concepts in cmg.PHASE2.items():
            for concept in concepts:
                x = cmg.sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p)
                model.train(x, family, concept)
                for _ in range(int(config["replay_steps"])):
                    old_x, old_family, old_concept = py_rng.choice(phase1_memory)
                    replay_x = (
                        (1.0 - float(config["replay_boost"])) * old_x
                        + float(config["replay_boost"]) * cmg.sample_multimodal_input(
                            rng, old_concept, noise, dropout_p, missing_modality_p
                        )
                    ).astype(np.float32)
                    model.train(replay_x, old_family, old_concept)

    retention_eval = c15.evaluate_model(model, cmg.PHASE1, 22, rng, noise, dropout_p, missing_modality_p)
    overall_eval = c15.evaluate_model(
        model,
        {family: cmg.PHASE1[family] + cmg.PHASE2[family] for family in cmg.FAMILIES},
        22,
        rng,
        noise,
        dropout_p,
        missing_modality_p,
    )
    concepts = [concept for family in cmg.FAMILIES for concept in cmg.PHASE1[family] + cmg.PHASE2[family]]
    consistency = c7.crossmodal_consistency(model, concepts, 10, rng, noise, dropout_p)
    temporal_score = c7.temporal_binding_score(model, concepts, 10, rng, noise, dropout_p)

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
    ap = argparse.ArgumentParser(description="Stage C32 three-path controller system search")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.22)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_c32_three_path_controller_system_search_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage_c30 = json.loads((TEMP_DIR / "stage_c30_three_path_architecture_search_20260311.json").read_text(encoding="utf-8"))
    stage_c15 = json.loads((TEMP_DIR / "stage_c15_stronger_long_horizon_binder_search_20260311.json").read_text(encoding="utf-8"))
    base = dict(stage_c30["best_retention_compatible_candidate"]["config"])
    incumbent = stage_c15["best_retention_compatible_candidate"]["summary"]
    strong_multimodal_target = float(stage_c15["headline_metrics"]["strong_multimodal_target"])
    baseline_consistency_ceiling = float(stage_c15["headline_metrics"]["baseline_consistency_ceiling"])

    best_objective = None
    best_consistency = None
    best_retention_compatible = None

    for identity_gain in [0.5, 1.0]:
        for memory_guard in [0.8, 1.2]:
            for novelty_gain in [0.2, 0.5]:
                for routing_sharpness in [0.8, 1.5]:
                    config = {
                        **base,
                        "identity_gain": identity_gain,
                        "memory_guard": memory_guard,
                        "novelty_gain": novelty_gain,
                        "routing_sharpness": routing_sharpness,
                    }
                    rows = []
                    for offset in range(int(args.num_seeds)):
                        rows.append(
                            run_system(
                                seed=int(args.seed) + offset,
                                noise=float(args.noise),
                                dropout_p=float(args.dropout_p),
                                missing_modality_p=float(args.missing_modality_p),
                                config=config,
                            )
                        )
                    summary = c15.summarize(rows)
                    candidate = {
                        "config": config,
                        "summary": summary,
                        "objective": c15.objective(summary),
                    }
                    if best_objective is None or candidate["objective"] > best_objective["objective"]:
                        best_objective = candidate
                    if best_consistency is None or summary["crossmodal_consistency"] > best_consistency["summary"]["crossmodal_consistency"]:
                        best_consistency = candidate
                    if (
                        summary["crossmodal_consistency"] >= baseline_consistency_ceiling
                        and summary["retention_concept_accuracy"] >= float(incumbent["retention_concept_accuracy"])
                        and summary["overall_concept_accuracy"] >= float(incumbent["overall_concept_accuracy"])
                    ):
                        if best_retention_compatible is None or candidate["objective"] > best_retention_compatible["objective"]:
                            best_retention_compatible = candidate

    assert best_objective is not None
    assert best_consistency is not None

    if best_retention_compatible is None:
        status = "stage_c32_controller_not_enough"
        core_answer = "The three-path controller system changes routing behavior, but still does not create a better retention-compatible family."
        main_open_gap = "three_path_controller_still_not_enough"
        best_compatible_consistency = 0.0
    else:
        best_compatible_consistency = float(best_retention_compatible["summary"]["crossmodal_consistency"])
        status = (
            "stage_c32_strong_direct_closure_found"
            if best_compatible_consistency >= strong_multimodal_target
            else "stage_c32_retention_compatible_controller_found"
        )
        core_answer = (
            "The three-path controller system creates a new retention-compatible family, but still does not reach the strong direct closure target."
            if best_compatible_consistency < strong_multimodal_target
            else "The three-path controller system reaches strong direct closure."
        )
        main_open_gap = (
            "strong_target_gap_remains_after_c32"
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
            "task_block": "StageC32_three_path_controller_system_search",
        },
        "headline_metrics": {
            "strong_multimodal_target": strong_multimodal_target,
            "baseline_consistency_ceiling": baseline_consistency_ceiling,
            "c15_retention_compatible_consistency": float(incumbent["crossmodal_consistency"]),
            "best_objective_consistency": float(best_objective["summary"]["crossmodal_consistency"]),
            "best_consistency_value": float(best_consistency["summary"]["crossmodal_consistency"]),
            "best_compatible_consistency": best_compatible_consistency,
            "strong_target_gap_after_best_compatible": float(max(0.0, strong_multimodal_target - best_compatible_consistency)),
        },
        "best_objective_candidate": best_objective,
        "best_consistency_candidate": best_consistency,
        "best_retention_compatible_candidate": best_retention_compatible,
        "hypotheses": {
            "H1_c32_beats_c15_retention_compatible": bool(
                best_retention_compatible is not None
                and best_compatible_consistency > float(incumbent["crossmodal_consistency"])
            ),
            "H2_c32_reaches_strong_target": bool(best_compatible_consistency >= strong_multimodal_target),
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
