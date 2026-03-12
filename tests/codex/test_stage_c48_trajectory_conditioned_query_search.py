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
import test_stage_c42_strong_cross_law_manifold_lift_search as c42
import test_stage_c44_manifold_conditioned_consistency_head_search as c44


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


class TrajectoryConditionedQueryGrounder(c44.ManifoldConditionedConsistencyHeadGrounder):
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
        geometry_anchor_pull: float,
        risk_accumulator_pull: float,
        direction_memory_pull: float,
        anchor_guard_weight: float,
        law_temperature: float,
        law_memory_gain: float,
        law_identity_gain: float,
        law_novelty_gain: float,
        external_map_pull: float,
        identity_preserve_pull: float,
        identity_anchor_mix: float,
        memory_rule_relax: float,
        identity_penalty_weight: float,
        transition_pull: float,
        switch_penalty_weight: float,
        transition_smooth_weight: float,
        transition_identity_bonus: float,
        latent_pull: float,
        latent_identity_mix: float,
        latent_transition_mix: float,
        latent_penalty_weight: float,
        manifold_coordinate_pull: float,
        manifold_identity_weight: float,
        manifold_persistence_weight: float,
        manifold_penalty_weight: float,
        family_manifold_pull: float,
        family_manifold_weight: float,
        relation_pull: float,
        temporal_vote_pull: float,
        readout_pull: float,
        readout_anchor_weight: float,
        readout_family_weight: float,
        readout_negative_weight: float,
        query_temperature: float,
        trajectory_pull: float,
        trajectory_weight: float,
        selective_penalty_weight: float,
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
            admissibility_threshold=admissibility_threshold,
            admissibility_strength=admissibility_strength,
            dangerous_shrink=dangerous_shrink,
            tangent_keep=tangent_keep,
            projector_weight=projector_weight,
            geometry_anchor_pull=geometry_anchor_pull,
            risk_accumulator_pull=risk_accumulator_pull,
            direction_memory_pull=direction_memory_pull,
            anchor_guard_weight=anchor_guard_weight,
            law_temperature=law_temperature,
            law_memory_gain=law_memory_gain,
            law_identity_gain=law_identity_gain,
            law_novelty_gain=law_novelty_gain,
            external_map_pull=external_map_pull,
            identity_preserve_pull=identity_preserve_pull,
            identity_anchor_mix=identity_anchor_mix,
            memory_rule_relax=memory_rule_relax,
            identity_penalty_weight=identity_penalty_weight,
            transition_pull=transition_pull,
            switch_penalty_weight=switch_penalty_weight,
            transition_smooth_weight=transition_smooth_weight,
            transition_identity_bonus=transition_identity_bonus,
            latent_pull=latent_pull,
            latent_identity_mix=latent_identity_mix,
            latent_transition_mix=latent_transition_mix,
            latent_penalty_weight=latent_penalty_weight,
            manifold_coordinate_pull=manifold_coordinate_pull,
            manifold_identity_weight=manifold_identity_weight,
            manifold_persistence_weight=manifold_persistence_weight,
            manifold_penalty_weight=manifold_penalty_weight,
            family_manifold_pull=family_manifold_pull,
            family_manifold_weight=family_manifold_weight,
            relation_pull=relation_pull,
            temporal_vote_pull=temporal_vote_pull,
            readout_pull=readout_pull,
            readout_anchor_weight=readout_anchor_weight,
            readout_family_weight=readout_family_weight,
            readout_negative_weight=readout_negative_weight,
        )
        self.query_temperature = query_temperature
        self.trajectory_pull = trajectory_pull
        self.trajectory_weight = trajectory_weight
        self.selective_penalty_weight = selective_penalty_weight
        self.trajectory_query_state: Dict[str, np.ndarray] = {}

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        super().train(x, family, concept)
        rows = []
        for state in (
            self.shared_manifold.get(concept),
            self.temporal_vote_state.get(concept),
            self.identity_preserve_map.get(concept),
            self.persistence_state.get(concept),
        ):
            if state is not None:
                rows.append(state.astype(np.float32))
        if not rows:
            return
        target = np.mean(np.stack(rows, axis=0), axis=0).astype(np.float32)
        prev = self.trajectory_query_state.get(concept)
        self.trajectory_query_state[concept] = (
            target
            if prev is None
            else ((1.0 - self.trajectory_pull) * prev + self.trajectory_pull * target).astype(np.float32)
        )

    def _query_gate(self, x: np.ndarray, family: str, concept: str) -> Tuple[float, np.ndarray | None]:
        manifold = self.shared_manifold.get(concept)
        readout = self.consistency_readout.get(concept)
        traj = self.trajectory_query_state.get(concept)
        if manifold is None or readout is None or traj is None:
            return 0.0, manifold
        manifold_energy = c7.sq_dist(x, manifold)
        readout_energy = c7.sq_dist(x, readout)
        traj_energy = c7.sq_dist(x, traj)
        delta = (manifold_energy - readout_energy - self.trajectory_weight * traj_energy) / max(1e-4, self.query_temperature)
        gate = 1.0 / (1.0 + np.exp(-delta))
        target = ((1.0 - gate) * traj + gate * readout).astype(np.float32)
        return float(gate), target

    def _fusion_penalty(self, x: np.ndarray, concept: str) -> float:
        penalty = super()._fusion_penalty(x, concept)
        family = self.concept_family.get(concept)
        if family is None:
            return penalty
        gate, target = self._query_gate(x, family, concept)
        if target is None:
            return penalty
        penalty += self.selective_penalty_weight * gate * c7.sq_dist(x, target)
        return float(penalty)


def run_system(seed: int, noise: float, dropout_p: float, missing_modality_p: float, config: Dict[str, float]) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    model = TrajectoryConditionedQueryGrounder(
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
        geometry_anchor_pull=float(config["geometry_anchor_pull"]),
        risk_accumulator_pull=float(config["risk_accumulator_pull"]),
        direction_memory_pull=float(config["direction_memory_pull"]),
        anchor_guard_weight=float(config["anchor_guard_weight"]),
        law_temperature=float(config["law_temperature"]),
        law_memory_gain=float(config["law_memory_gain"]),
        law_identity_gain=float(config["law_identity_gain"]),
        law_novelty_gain=float(config["law_novelty_gain"]),
        external_map_pull=float(config["external_map_pull"]),
        identity_preserve_pull=float(config["identity_preserve_pull"]),
        identity_anchor_mix=float(config["identity_anchor_mix"]),
        memory_rule_relax=float(config["memory_rule_relax"]),
        identity_penalty_weight=float(config["identity_penalty_weight"]),
        transition_pull=float(config["transition_pull"]),
        switch_penalty_weight=float(config["switch_penalty_weight"]),
        transition_smooth_weight=float(config["transition_smooth_weight"]),
        transition_identity_bonus=float(config["transition_identity_bonus"]),
        latent_pull=float(config["latent_pull"]),
        latent_identity_mix=float(config["latent_identity_mix"]),
        latent_transition_mix=float(config["latent_transition_mix"]),
        latent_penalty_weight=float(config["latent_penalty_weight"]),
        manifold_coordinate_pull=float(config["manifold_coordinate_pull"]),
        manifold_identity_weight=float(config["manifold_identity_weight"]),
        manifold_persistence_weight=float(config["manifold_persistence_weight"]),
        manifold_penalty_weight=float(config["manifold_penalty_weight"]),
        family_manifold_pull=float(config["family_manifold_pull"]),
        family_manifold_weight=float(config["family_manifold_weight"]),
        relation_pull=float(config["relation_pull"]),
        temporal_vote_pull=float(config["temporal_vote_pull"]),
        readout_pull=float(config["readout_pull"]),
        readout_anchor_weight=float(config["readout_anchor_weight"]),
        readout_family_weight=float(config["readout_family_weight"]),
        readout_negative_weight=float(config["readout_negative_weight"]),
        query_temperature=float(config["query_temperature"]),
        trajectory_pull=float(config["trajectory_pull"]),
        trajectory_weight=float(config["trajectory_weight"]),
        selective_penalty_weight=float(config["selective_penalty_weight"]),
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
    ap = argparse.ArgumentParser(description="Stage C48 trajectory-conditioned query search")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.22)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_c48_trajectory_conditioned_query_search_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage_c42 = json.loads(
        (TEMP_DIR / "stage_c42_strong_cross_law_manifold_lift_search_20260312.json").read_text(encoding="utf-8")
    )
    stage_c44 = json.loads(
        (TEMP_DIR / "stage_c44_manifold_conditioned_consistency_head_search_20260312.json").read_text(encoding="utf-8")
    )
    stage_c15 = json.loads(
        (TEMP_DIR / "stage_c15_stronger_long_horizon_binder_search_20260311.json").read_text(encoding="utf-8")
    )
    base = {
        **dict(stage_c42["best_objective_candidate"]["config"]),
        "readout_pull": float(stage_c44["best_objective_candidate"]["config"]["readout_pull"]),
        "readout_anchor_weight": float(stage_c44["best_objective_candidate"]["config"]["readout_anchor_weight"]),
        "readout_family_weight": float(stage_c44["best_objective_candidate"]["config"]["readout_family_weight"]),
        "readout_negative_weight": float(stage_c44["best_objective_candidate"]["config"]["readout_negative_weight"]),
    }
    incumbent = stage_c15["best_retention_compatible_candidate"]["summary"]
    c42_retention_compatible = stage_c42["best_retention_compatible_candidate"]["summary"]
    strong_multimodal_target = float(stage_c15["headline_metrics"]["strong_multimodal_target"])
    baseline_consistency_ceiling = float(stage_c15["headline_metrics"]["baseline_consistency_ceiling"])

    best_objective = None
    best_consistency = None
    best_retention_compatible = None

    for query_temperature in [0.14, 0.24]:
        for trajectory_pull in [0.10, 0.18]:
            for trajectory_weight in [0.6, 0.9]:
                for selective_penalty_weight in [0.02, 0.05]:
                    config = {
                        **base,
                        "query_temperature": query_temperature,
                        "trajectory_pull": trajectory_pull,
                        "trajectory_weight": trajectory_weight,
                        "selective_penalty_weight": selective_penalty_weight,
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
        status = "stage_c48_trajectory_conditioned_query_not_enough"
        core_answer = "A trajectory-conditioned query improves history-aware readout, but still does not produce a better retention-compatible family."
        main_open_gap = "trajectory_conditioned_query_still_not_enough"
    else:
        best_compatible_consistency = float(best_retention_compatible["summary"]["crossmodal_consistency"])
        status = (
            "stage_c48_strong_direct_closure_found"
            if best_compatible_consistency >= strong_multimodal_target
            else "stage_c48_retention_compatible_trajectory_query_found"
        )
        core_answer = (
            "A trajectory-conditioned query creates a stronger retention-compatible family, but still does not reach the strong target."
            if best_compatible_consistency < strong_multimodal_target
            else "A trajectory-conditioned query reaches strong direct closure."
        )
        main_open_gap = (
            "strong_target_gap_remains_after_c48"
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
            "task_block": "StageC48_trajectory_conditioned_query_search",
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
            "H1_c48_beats_c42_retention_compatible": bool(
                best_retention_compatible is not None
                and best_compatible_consistency > float(c42_retention_compatible["crossmodal_consistency"])
            ),
            "H2_c48_reaches_strong_target": bool(best_compatible_consistency >= strong_multimodal_target),
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
