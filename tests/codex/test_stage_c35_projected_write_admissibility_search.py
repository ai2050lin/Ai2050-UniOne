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
import test_stage_c34_hybrid_controller_write_veto_search as c34


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def gram_schmidt(vectors: List[np.ndarray]) -> List[np.ndarray]:
    basis: List[np.ndarray] = []
    for vec in vectors:
        work = vec.astype(np.float64)
        for base in basis:
            work = work - float(np.dot(work, base)) * base
        norm = float(np.linalg.norm(work))
        if norm > 1e-8:
            basis.append((work / norm).astype(np.float64))
    return [b.astype(np.float32) for b in basis]


class ProjectedWriteAdmissibilityGrounder(c34.WriteVetoProtectedMemoryGrounder):
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
        mode_identity_bias: float,
        mode_memory_bias: float,
        mode_novelty_bias: float,
        mode_margin: float,
        veto_threshold: float,
        veto_strength: float,
        memory_shield_weight: float,
        shield_pull: float,
        admissibility_threshold: float,
        admissibility_strength: float,
        dangerous_shrink: float,
        tangent_keep: float,
        projector_weight: float,
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
            mode_identity_bias=mode_identity_bias,
            mode_memory_bias=mode_memory_bias,
            mode_novelty_bias=mode_novelty_bias,
            mode_margin=mode_margin,
            veto_threshold=veto_threshold,
            veto_strength=veto_strength,
            memory_shield_weight=memory_shield_weight,
            shield_pull=shield_pull,
        )
        self.admissibility_threshold = admissibility_threshold
        self.admissibility_strength = admissibility_strength
        self.dangerous_shrink = dangerous_shrink
        self.tangent_keep = tangent_keep
        self.projector_weight = projector_weight

    def _protected_center(self, concept: str) -> np.ndarray | None:
        pieces = []
        for state in (
            self.protected_memory.get(concept),
            self.memory_path.get(concept),
            self.retention_state.get(concept),
            self.read_slot.get(concept),
            self.persistence_state.get(concept),
        ):
            if state is not None:
                pieces.append(state.astype(np.float32))
        if not pieces:
            return None
        return np.mean(np.stack(pieces, axis=0), axis=0).astype(np.float32)

    def _dangerous_basis(self, concept: str, center: np.ndarray) -> List[np.ndarray]:
        candidates = []
        for state in (
            self.protected_memory.get(concept),
            self.memory_path.get(concept),
            self.retention_state.get(concept),
            self.read_slot.get(concept),
            self.persistence_state.get(concept),
        ):
            if state is not None:
                candidates.append((state - center).astype(np.float32))
        return gram_schmidt(candidates)

    def _project_update(self, state: np.ndarray, concept: str, admissibility: float) -> np.ndarray:
        center = self._protected_center(concept)
        if center is None:
            return state
        delta = (state - center).astype(np.float32)
        basis = self._dangerous_basis(concept, center)
        dangerous = np.zeros_like(delta, dtype=np.float32)
        residual = delta.astype(np.float32)
        for base in basis:
            coeff = float(np.dot(residual, base))
            dangerous = dangerous + coeff * base
            residual = residual - coeff * base
        safe_delta = (
            (1.0 - admissibility * self.dangerous_shrink) * dangerous
            + (1.0 - admissibility * (1.0 - self.tangent_keep)) * residual
        ).astype(np.float32)
        return (center + safe_delta).astype(np.float32)

    def _projection_cost(self, concept: str) -> float:
        center = self._protected_center(concept)
        if center is None:
            return 0.0
        total = 0.0
        count = 0
        basis = self._dangerous_basis(concept, center)
        for state in (self.identity_path.get(concept), self.novelty_path.get(concept)):
            if state is None:
                continue
            delta = (state - center).astype(np.float32)
            dangerous_norm = 0.0
            for base in basis:
                coeff = float(np.dot(delta, base))
                dangerous_norm += coeff * coeff
            total += dangerous_norm
            count += 1
        if count == 0:
            return 0.0
        return float(total / count)

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        super().train(x, family, concept)
        risk = self._write_risk(concept)
        admissibility = sigmoid(self.admissibility_strength * (risk - self.admissibility_threshold))

        pair = self.identity_path.get(concept)
        nov = self.novelty_path.get(concept)
        if pair is not None:
            self.identity_path[concept] = self._project_update(pair, concept, admissibility)
        if nov is not None:
            self.novelty_path[concept] = self._project_update(nov, concept, admissibility)

    def _fusion_penalty(self, x: np.ndarray, concept: str) -> float:
        base = super()._fusion_penalty(x, concept)
        risk = self._write_risk(concept)
        admissibility = sigmoid(self.admissibility_strength * (risk - self.admissibility_threshold))
        return float(base + self.projector_weight * admissibility * self._projection_cost(concept))


def run_system(seed: int, noise: float, dropout_p: float, missing_modality_p: float, config: Dict[str, float]) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    model = ProjectedWriteAdmissibilityGrounder(
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
        mode_identity_bias=float(config["mode_identity_bias"]),
        mode_memory_bias=float(config["mode_memory_bias"]),
        mode_novelty_bias=float(config["mode_novelty_bias"]),
        mode_margin=float(config["mode_margin"]),
        veto_threshold=float(config["veto_threshold"]),
        veto_strength=float(config["veto_strength"]),
        memory_shield_weight=float(config["memory_shield_weight"]),
        shield_pull=float(config["shield_pull"]),
        admissibility_threshold=float(config["admissibility_threshold"]),
        admissibility_strength=float(config["admissibility_strength"]),
        dangerous_shrink=float(config["dangerous_shrink"]),
        tangent_keep=float(config["tangent_keep"]),
        projector_weight=float(config["projector_weight"]),
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
    ap = argparse.ArgumentParser(description="Stage C35 projected write admissibility search")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.22)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_c35_projected_write_admissibility_search_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage_c34 = json.loads(
        (TEMP_DIR / "stage_c34_hybrid_controller_write_veto_search_20260311.json").read_text(encoding="utf-8")
    )
    stage_c15 = json.loads(
        (TEMP_DIR / "stage_c15_stronger_long_horizon_binder_search_20260311.json").read_text(encoding="utf-8")
    )
    base = dict(stage_c34["best_objective_candidate"]["config"])
    incumbent = stage_c15["best_retention_compatible_candidate"]["summary"]
    strong_multimodal_target = float(stage_c15["headline_metrics"]["strong_multimodal_target"])
    baseline_consistency_ceiling = float(stage_c15["headline_metrics"]["baseline_consistency_ceiling"])

    best_objective = None
    best_consistency = None
    best_retention_compatible = None

    for admissibility_threshold in [8.0, 16.0]:
        for dangerous_shrink in [0.55, 0.85]:
            for tangent_keep in [0.60, 0.85]:
                for projector_weight in [0.02, 0.06]:
                    config = {
                        **base,
                        "admissibility_threshold": admissibility_threshold,
                        "admissibility_strength": 0.35,
                        "dangerous_shrink": dangerous_shrink,
                        "tangent_keep": tangent_keep,
                        "projector_weight": projector_weight,
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
                        if best_retention_compatible is None or candidate["objective"] > best_retention_compatible["objective"]:
                            best_retention_compatible = candidate

    assert best_objective is not None
    assert best_consistency is not None

    if best_retention_compatible is None:
        best_compatible_consistency = 0.0
        status = "stage_c35_projected_admissibility_not_enough"
        core_answer = "Projected admissibility protects memory directions better than scalar veto, but still does not produce a new retention-compatible family."
        main_open_gap = "projected_update_still_not_enough"
    else:
        best_compatible_consistency = float(best_retention_compatible["summary"]["crossmodal_consistency"])
        status = (
            "stage_c35_strong_direct_closure_found"
            if best_compatible_consistency >= strong_multimodal_target
            else "stage_c35_retention_compatible_projected_update_found"
        )
        core_answer = (
            "Projected admissibility creates a new retention-compatible family, but still does not reach the strong target."
            if best_compatible_consistency < strong_multimodal_target
            else "Projected admissibility reaches strong direct closure."
        )
        main_open_gap = (
            "strong_target_gap_remains_after_c35"
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
            "task_block": "StageC35_projected_write_admissibility_search",
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
            "H1_c35_beats_c15_retention_compatible": bool(
                best_retention_compatible is not None
                and best_compatible_consistency > float(incumbent["crossmodal_consistency"])
            ),
            "H2_c35_reaches_strong_target": bool(best_compatible_consistency >= strong_multimodal_target),
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
