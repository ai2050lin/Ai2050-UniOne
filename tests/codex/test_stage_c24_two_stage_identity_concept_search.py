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
import test_stage_c23_decoupled_identity_concept_memory_search as c23


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


class TwoStageIdentityConceptGrounder(c23.DecoupledIdentityConceptGrounder):
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
        stage1_identity_weight: float,
        stage2_concept_weight: float,
        stage1_topk: int,
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
        )
        self.stage1_identity_weight = stage1_identity_weight
        self.stage2_concept_weight = stage2_concept_weight
        self.stage1_topk = stage1_topk

    def _identity_cluster_score(self, x: np.ndarray, concept: str) -> float:
        score = 0.0
        fast = self.identity_fast_state.get(concept)
        pair = self.pair_state.get(concept)
        margin = self.margin_state.get(concept)
        if fast is not None:
            score += 0.45 * c7.sq_dist(x, fast)
        if pair is not None:
            score += 0.35 * c7.sq_dist(x, pair)
        if margin is not None:
            score += 0.20 * c7.sq_dist(x, margin)
        return float(score)

    def _concept_memory_score(self, x: np.ndarray, concept: str) -> float:
        score = 0.0
        slow = self.concept_slow_state.get(concept)
        retention = self.retention_state.get(concept)
        manifold = self.object_manifold.get(concept)
        if slow is not None:
            score += 0.45 * c7.sq_dist(x, slow)
        if retention is not None:
            score += 0.30 * c7.sq_dist(x, retention)
        if manifold is not None:
            score += 0.25 * c7.sq_dist(x, manifold)
        return float(score)

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        concepts = list(self.concept_family.keys())
        stage1_scores = {concept: self._identity_cluster_score(x, concept) for concept in concepts}
        ranked = sorted(concepts, key=lambda name: stage1_scores[name])
        topk = ranked[: max(1, self.stage1_topk)]

        best_concept = None
        best_score = None
        for concept in topk:
            family = self.concept_family[concept]
            observed = self._observed_modalities(x)
            family_score = self._family_score(x, observed, family)
            full_score = super()._concept_score(x, observed, family, concept, family_score)
            fused_score = (
                self.stage1_identity_weight * stage1_scores[concept]
                + self.stage2_concept_weight * self._concept_memory_score(x, concept)
                + full_score
            )
            if best_score is None or fused_score < best_score:
                best_score = fused_score
                best_concept = concept

        assert best_concept is not None
        return self.concept_family[best_concept], best_concept


def run_system(seed: int, noise: float, dropout_p: float, missing_modality_p: float, config: Dict[str, float]) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    model = TwoStageIdentityConceptGrounder(
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
        stage1_identity_weight=float(config["stage1_identity_weight"]),
        stage2_concept_weight=float(config["stage2_concept_weight"]),
        stage1_topk=int(config["stage1_topk"]),
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
                replay_count = int(config["replay_after_novel_steps"]) + int(config["extra_retention_replay"])
                for _ in range(replay_count):
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
                replay_count = int(config["replay_steps"]) + int(config["extra_retention_replay"])
                for _ in range(replay_count):
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
    ap = argparse.ArgumentParser(description="Stage C24 two-stage identity-first / concept-second search")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.22)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_c24_two_stage_identity_concept_search_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage_c23 = json.loads((TEMP_DIR / "stage_c23_decoupled_identity_concept_memory_search_20260311.json").read_text(encoding="utf-8"))
    stage_c15 = json.loads((TEMP_DIR / "stage_c15_stronger_long_horizon_binder_search_20260311.json").read_text(encoding="utf-8"))
    base = dict(stage_c23["best_objective_candidate"]["config"])
    incumbent = stage_c15["best_retention_compatible_candidate"]["summary"]
    strong_multimodal_target = float(stage_c15["headline_metrics"]["strong_multimodal_target"])
    baseline_consistency_ceiling = float(stage_c15["headline_metrics"]["baseline_consistency_ceiling"])

    best_objective = None
    best_consistency = None
    best_retention_compatible = None

    for stage1_identity_weight in [0.4, 0.8]:
        for stage2_concept_weight in [0.2, 0.5]:
            for stage1_topk in [2, 3]:
                config = {
                    **base,
                    "stage1_identity_weight": stage1_identity_weight,
                    "stage2_concept_weight": stage2_concept_weight,
                    "stage1_topk": stage1_topk,
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
        status = "stage_c24_two_stage_not_enough"
        core_answer = "Two-stage identity-first / concept-second softens decision coupling, but still does not create a better retention-compatible family."
        main_open_gap = "two_stage_decision_still_not_enough"
        best_compatible_consistency = 0.0
    else:
        best_compatible_consistency = float(best_retention_compatible["summary"]["crossmodal_consistency"])
        status = (
            "stage_c24_strong_direct_closure_found"
            if best_compatible_consistency >= strong_multimodal_target
            else "stage_c24_retention_compatible_two_stage_found"
        )
        core_answer = (
            "Two-stage identity-first / concept-second creates a new retention-compatible family, but still does not reach the strong direct closure target."
            if best_compatible_consistency < strong_multimodal_target
            else "Two-stage identity-first / concept-second reaches strong direct closure."
        )
        main_open_gap = (
            "strong_target_gap_remains_after_c24"
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
            "task_block": "StageC24_two_stage_identity_concept_search",
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
            "H1_c24_improves_over_c15_retention_compatible": bool(
                best_retention_compatible is not None
                and best_compatible_consistency > float(incumbent["crossmodal_consistency"])
            ),
            "H2_c24_reaches_strong_target": bool(best_compatible_consistency >= strong_multimodal_target),
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
