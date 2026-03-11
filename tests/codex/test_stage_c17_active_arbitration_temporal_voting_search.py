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
import test_stage_c16_identity_graph_consensus_search as c16


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"
MODALITY_NAMES = ("visual", "tactile", "language")
MODALITY_SLICES = c7.MODALITY_SLICES


class ActiveArbitrationGrounder(c16.IdentityGraphConsensusGrounder):
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
        arbitration_weight: float,
        disagreement_penalty: float,
        temporal_vote_weight: float,
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
        )
        self.arbitration_weight = arbitration_weight
        self.disagreement_penalty = disagreement_penalty
        self.temporal_vote_weight = temporal_vote_weight
        self.modality_reliability: Dict[str, Dict[str, float]] = {name: {} for name in MODALITY_NAMES}

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        super().train(x, family, concept)
        canonical = self.canonical_concept.get(concept)
        if canonical is None:
            return
        for modality, sl in MODALITY_SLICES.items():
            part = x[sl].astype(np.float32)
            if float(np.linalg.norm(part)) <= 1e-5:
                continue
            normed = self.norms[modality].normalize(part)
            rel = 1.0 / (1e-4 + c7.sq_dist(normed, canonical))
            prev = self.modality_reliability[modality].get(concept)
            if prev is None:
                self.modality_reliability[modality][concept] = float(rel)
            else:
                self.modality_reliability[modality][concept] = float(0.85 * prev + 0.15 * rel)

    def _single_modality_view(self, x: np.ndarray, modality: str) -> np.ndarray:
        y = np.zeros_like(x)
        sl = MODALITY_SLICES[modality]
        y[sl] = x[sl]
        return y

    def _concept_vote(self, x: np.ndarray, family: str, concept: str) -> Tuple[float, float]:
        observed = self._observed_modalities(x)
        base_family_score = self._family_score(x, observed, family)
        base_score = self._concept_score(x, observed, family, concept, base_family_score)

        modality_scores = []
        modality_weights = []
        for modality, _ in observed:
            view = self._single_modality_view(x, modality)
            view_observed = self._observed_modalities(view)
            family_score = self._family_score(view, view_observed, family)
            concept_score = self._concept_score(view, view_observed, family, concept, family_score)
            modality_scores.append(concept_score)
            modality_weights.append(self.modality_reliability[modality].get(concept, 1.0))

        if modality_scores:
            weighted_modality = float(np.average(np.array(modality_scores, dtype=np.float64), weights=np.array(modality_weights, dtype=np.float64)))
            disagreement = float(np.std(np.array(modality_scores, dtype=np.float64)))
        else:
            weighted_modality = base_score
            disagreement = 0.0

        temporal_vote = 0.0
        persist = self.persistence_state.get(concept)
        node = self.identity_node.get(concept)
        if persist is not None:
            temporal_vote += c7.sq_dist(x, persist)
        if node is not None:
            temporal_vote += c7.sq_dist(x, node)

        total = (
            base_score
            + self.arbitration_weight * weighted_modality
            + self.temporal_vote_weight * temporal_vote
            + self.disagreement_penalty * disagreement
        )
        return float(total), float(disagreement)

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        families = list({fam for fam in self.concept_family.values()})
        family_scores = {}
        for family in families:
            concept_votes = []
            concepts = [name for name, fam in self.concept_family.items() if fam == family]
            for concept in concepts:
                vote, _ = self._concept_vote(x, family, concept)
                concept_votes.append(vote)
            family_scores[family] = float(min(concept_votes)) if concept_votes else 1e9

        family = min(family_scores, key=family_scores.get)
        concepts = [name for name, fam in self.concept_family.items() if fam == family]
        concept = min(concepts, key=lambda name: self._concept_vote(x, family, name)[0])
        return family, concept


def run_system(seed: int, noise: float, dropout_p: float, missing_modality_p: float, config: Dict[str, float]) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    model = ActiveArbitrationGrounder(
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
        arbitration_weight=float(config["arbitration_weight"]),
        disagreement_penalty=float(config["disagreement_penalty"]),
        temporal_vote_weight=float(config["temporal_vote_weight"]),
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
    ap = argparse.ArgumentParser(description="Stage C17 active arbitration temporal voting search")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.22)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_c17_active_arbitration_temporal_voting_search_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage_c16 = json.loads((TEMP_DIR / "stage_c16_identity_graph_consensus_search_20260311.json").read_text(encoding="utf-8"))
    stage_c15 = json.loads((TEMP_DIR / "stage_c15_stronger_long_horizon_binder_search_20260311.json").read_text(encoding="utf-8"))
    base = dict(stage_c16["best_objective_candidate"]["config"])
    incumbent = stage_c15["best_retention_compatible_candidate"]["summary"]
    strong_multimodal_target = float(stage_c15["headline_metrics"]["strong_multimodal_target"])
    baseline_consistency_ceiling = float(stage_c15["headline_metrics"]["baseline_consistency_ceiling"])

    best_objective = None
    best_consistency = None
    best_retention_compatible = None

    for arbitration_weight in [0.1, 0.35]:
        for disagreement_penalty in [0.02, 0.08]:
            for temporal_vote_weight in [0.02, 0.08]:
                config = {
                    **base,
                    "arbitration_weight": arbitration_weight,
                    "disagreement_penalty": disagreement_penalty,
                    "temporal_vote_weight": temporal_vote_weight,
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
        status = "stage_c17_active_arbitration_not_retention_compatible"
        core_answer = "Active arbitration raises selective direct consistency, but still cannot beat the current retention-compatible family."
        main_open_gap = "active_arbitration_still_not_enough"
        best_compatible_consistency = 0.0
    else:
        best_compatible_consistency = float(best_retention_compatible["summary"]["crossmodal_consistency"])
        status = (
            "stage_c17_strong_direct_closure_found"
            if best_compatible_consistency >= strong_multimodal_target
            else "stage_c17_active_arbitration_family_strengthened"
        )
        core_answer = (
            "Active arbitration strengthens the retention-compatible family, but still does not reach the strong direct closure target."
            if best_compatible_consistency < strong_multimodal_target
            else "Active arbitration reaches strong direct closure."
        )
        main_open_gap = (
            "strong_target_gap_remains_after_c17"
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
            "task_block": "StageC17_active_arbitration_temporal_voting_search",
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
            "H1_c17_improves_over_c15_retention_compatible": bool(
                best_retention_compatible is not None
                and best_compatible_consistency > float(incumbent["crossmodal_consistency"])
            ),
            "H2_c17_reaches_strong_target": bool(best_compatible_consistency >= strong_multimodal_target),
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
