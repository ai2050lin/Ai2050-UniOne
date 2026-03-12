from __future__ import annotations

import argparse
import inspect
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import test_continuous_multimodal_grounding_proto as cmg
import test_stage_c7_consensus_discriminator_temporal_binding_search as c7
import test_stage_c15_stronger_long_horizon_binder_search as c15
import test_stage_c42_strong_cross_law_manifold_lift_search as c42
import test_stage_c51_dual_geometry_compatibility_projection_search as c51
import test_stage_c52_trilevel_compatibility_law_search as c52
import test_stage_c53_shared_manifold_discriminative_transport_search as c53
import test_stage_c54_global_transport_dual_manifold_alignment_search as c54


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


class PhaseConditionedManifoldGrounder(c54.GlobalTransportDualManifoldGrounder):
    def __init__(
        self,
        *args,
        phase_pull: float,
        phase_object_weight: float,
        phase_discriminative_weight: float,
        phase_memory_weight: float,
        phase_identity_weight: float,
        phase_penalty_weight: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.phase_pull = phase_pull
        self.phase_object_weight = phase_object_weight
        self.phase_discriminative_weight = phase_discriminative_weight
        self.phase_memory_weight = phase_memory_weight
        self.phase_identity_weight = phase_identity_weight
        self.phase_penalty_weight = phase_penalty_weight
        self.phase_object_state: Dict[str, np.ndarray] = {}
        self.phase_discriminative_state: Dict[str, np.ndarray] = {}

    def _phase_key(self, concept: str) -> str:
        risk = self.risk_accumulator.get(concept, self._write_risk(concept))
        if risk > self.veto_threshold:
            return "memory"
        if risk > self.admissibility_threshold:
            return "identity"
        return "stabilize"

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        super().train(x, family, concept)

        phase = self._phase_key(concept)
        object_state = self.object_manifold_state.get(concept)
        discrim_state = self.discriminative_manifold_state.get(concept)
        protected = self._protected_center(concept)
        identity = self.identity_preserve_map.get(concept)
        transport = self.transport_state.get(concept)

        object_rows: List[np.ndarray] = []
        discrim_rows: List[np.ndarray] = []
        if object_state is not None:
            object_rows.append(self.phase_object_weight * object_state.astype(np.float32))
        if discrim_state is not None:
            discrim_rows.append(self.phase_discriminative_weight * discrim_state.astype(np.float32))
        if protected is not None and phase == "memory":
            object_rows.append(self.phase_memory_weight * protected.astype(np.float32))
            discrim_rows.append(0.35 * protected.astype(np.float32))
        if identity is not None and phase == "identity":
            object_rows.append(0.35 * identity.astype(np.float32))
            discrim_rows.append(self.phase_identity_weight * identity.astype(np.float32))
        if transport is not None and phase == "stabilize":
            object_rows.append(0.45 * transport.astype(np.float32))
            discrim_rows.append(0.45 * transport.astype(np.float32))

        if object_rows:
            target = np.mean(np.stack(object_rows, axis=0), axis=0).astype(np.float32)
            prev = self.phase_object_state.get(concept)
            self.phase_object_state[concept] = (
                target
                if prev is None
                else ((1.0 - self.phase_pull) * prev + self.phase_pull * target).astype(np.float32)
            )
        if discrim_rows:
            target = np.mean(np.stack(discrim_rows, axis=0), axis=0).astype(np.float32)
            prev = self.phase_discriminative_state.get(concept)
            self.phase_discriminative_state[concept] = (
                target
                if prev is None
                else ((1.0 - self.phase_pull) * prev + self.phase_pull * target).astype(np.float32)
            )

    def _fusion_penalty(self, x: np.ndarray, concept: str) -> float:
        penalty = super()._fusion_penalty(x, concept)
        object_state = self.phase_object_state.get(concept)
        discrim_state = self.phase_discriminative_state.get(concept)
        if object_state is None or discrim_state is None:
            return penalty
        phase = self._phase_key(concept)
        align_cost = c7.sq_dist(object_state, discrim_state)
        query_cost = c7.sq_dist(x, discrim_state)
        if phase == "memory":
            weight = 1.25
        elif phase == "identity":
            weight = 1.0
        else:
            weight = 0.85
        return float(penalty + weight * self.phase_penalty_weight * (0.4 * align_cost + 0.6 * query_cost))


def run_system(seed: int, noise: float, dropout_p: float, missing_modality_p: float, config: Dict[str, float]) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    allowed = set(inspect.signature(c51.DualGeometryCompatibilityGrounder.__init__).parameters)
    allowed.update(inspect.signature(c52.TriLevelCompatibilityGrounder.__init__).parameters)
    allowed.update(inspect.signature(c53.SharedManifoldTransportGrounder.__init__).parameters)
    allowed.update(inspect.signature(c54.GlobalTransportDualManifoldGrounder.__init__).parameters)
    allowed.update(inspect.signature(PhaseConditionedManifoldGrounder.__init__).parameters)
    allowed.discard("self")
    allowed.discard("args")
    allowed.discard("kwargs")
    model_config = {key: float(value) for key, value in config.items() if key in allowed}
    if "slot_weight" not in model_config and "object_token_weight" in model_config:
        model_config["slot_weight"] = float(model_config["object_token_weight"])
    model_config.pop("object_token_weight", None)
    model = PhaseConditionedManifoldGrounder(**model_config)

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
                        + float(config["replay_boost"])
                        * cmg.sample_multimodal_input(rng, old_concept, noise, dropout_p, missing_modality_p)
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
                        + float(config["replay_boost"])
                        * cmg.sample_multimodal_input(rng, old_concept, noise, dropout_p, missing_modality_p)
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
    ap = argparse.ArgumentParser(description="Stage C55 phase-conditioned manifold geometry search")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.22)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_c55_phase_conditioned_manifold_geometry_search_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage_c15 = json.loads(
        (TEMP_DIR / "stage_c15_stronger_long_horizon_binder_search_20260311.json").read_text(encoding="utf-8")
    )
    stage_c42 = json.loads(
        (TEMP_DIR / "stage_c42_strong_cross_law_manifold_lift_search_20260312.json").read_text(encoding="utf-8")
    )
    stage_c54 = json.loads(
        (TEMP_DIR / "stage_c54_global_transport_dual_manifold_alignment_search_20260312.json").read_text(encoding="utf-8")
    )

    base = dict(stage_c54["best_objective_candidate"]["config"])
    incumbent = stage_c15["best_retention_compatible_candidate"]["summary"]
    c42_retention_compatible = stage_c42["best_retention_compatible_candidate"]["summary"]
    strong_multimodal_target = float(stage_c15["headline_metrics"]["strong_multimodal_target"])
    baseline_consistency_ceiling = float(stage_c15["headline_metrics"]["baseline_consistency_ceiling"])

    best_objective = None
    best_consistency = None
    best_retention_compatible = None

    for phase_pull in [0.08, 0.18]:
        for phase_object_weight in [0.25, 0.45]:
            for phase_discriminative_weight in [0.25, 0.45]:
                for phase_memory_weight in [0.20, 0.40]:
                    for phase_identity_weight in [0.20, 0.40]:
                        for phase_penalty_weight in [0.02, 0.05]:
                            config = {
                                **base,
                                "phase_pull": phase_pull,
                                "phase_object_weight": phase_object_weight,
                                "phase_discriminative_weight": phase_discriminative_weight,
                                "phase_memory_weight": phase_memory_weight,
                                "phase_identity_weight": phase_identity_weight,
                                "phase_penalty_weight": phase_penalty_weight,
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
                            if (
                                best_consistency is None
                                or summary["crossmodal_consistency"] > best_consistency["summary"]["crossmodal_consistency"]
                            ):
                                best_consistency = candidate
                            if (
                                summary["crossmodal_consistency"] >= baseline_consistency_ceiling
                                and summary["retention_concept_accuracy"] >= float(incumbent["retention_concept_accuracy"])
                                and summary["overall_concept_accuracy"] >= float(incumbent["overall_concept_accuracy"])
                            ):
                                if (
                                    best_retention_compatible is None
                                    or candidate["objective"] > best_retention_compatible["objective"]
                                ):
                                    best_retention_compatible = candidate

    assert best_objective is not None
    assert best_consistency is not None

    if best_retention_compatible is None:
        best_compatible_consistency = 0.0
        status = "stage_c55_phase_conditioned_geometry_not_enough"
        core_answer = "A phase-conditioned manifold geometry improves phase separation, but still does not produce a better retention-compatible family."
        main_open_gap = "phase_conditioned_compatibility_still_not_enough"
    else:
        best_compatible_consistency = float(best_retention_compatible["summary"]["crossmodal_consistency"])
        status = (
            "stage_c55_strong_direct_closure_found"
            if best_compatible_consistency >= strong_multimodal_target
            else "stage_c55_retention_compatible_phase_geometry_found"
        )
        core_answer = (
            "A phase-conditioned manifold geometry creates a stronger retention-compatible family, but still does not reach the strong target."
            if best_compatible_consistency < strong_multimodal_target
            else "A phase-conditioned manifold geometry reaches strong direct closure."
        )
        main_open_gap = (
            "strong_target_gap_remains_after_c55"
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
            "task_block": "StageC55_phase_conditioned_manifold_geometry_search",
        },
        "headline_metrics": {
            "strong_multimodal_target": strong_multimodal_target,
            "baseline_consistency_ceiling": baseline_consistency_ceiling,
            "c15_retention_compatible_consistency": float(incumbent["crossmodal_consistency"]),
            "c42_retention_compatible_consistency": float(c42_retention_compatible["crossmodal_consistency"]),
            "best_objective_consistency": float(best_objective["summary"]["crossmodal_consistency"]),
            "best_consistency_value": float(best_consistency["summary"]["crossmodal_consistency"]),
            "best_compatible_consistency": best_compatible_consistency,
            "strong_target_gap_after_best_compatible": float(max(0.0, strong_multimodal_target - best_compatible_consistency)),
        },
        "best_objective_candidate": best_objective,
        "best_consistency_candidate": best_consistency,
        "best_retention_compatible_candidate": best_retention_compatible,
        "hypotheses": {
            "H1_c55_beats_c42_retention_compatible": bool(
                best_retention_compatible is not None
                and best_compatible_consistency > float(c42_retention_compatible["crossmodal_consistency"])
            ),
            "H2_c55_reaches_strong_target": bool(best_compatible_consistency >= strong_multimodal_target),
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
