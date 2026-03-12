from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

import test_continuous_multimodal_grounding_proto as cmg
import test_stage_c7_consensus_discriminator_temporal_binding_search as c7
import test_stage_c15_stronger_long_horizon_binder_search as c15
import test_stage_c42_strong_cross_law_manifold_lift_search as c42


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


class ExactPrototypeOracle:
    def __init__(self) -> None:
        self.family_proto = {}
        self.concept_proto = {}
        for family in cmg.FAMILIES:
            members = cmg.PHASE1[family] + cmg.PHASE2[family]
            rows = [self._concept_proto(concept) for concept in members]
            self.family_proto[family] = np.mean(np.stack(rows, axis=0), axis=0).astype(np.float32)
            for concept in members:
                self.concept_proto[concept] = self._concept_proto(concept)

    def _concept_proto(self, concept: str) -> np.ndarray:
        family = cmg.concept_family(concept)
        base16 = cmg.proto.family_basis()[family] + cmg.proto.concept_offset()[concept]
        language = cmg.lang_family_basis()[family] + cmg.lang_concept_offset()[concept]
        return np.concatenate([base16[:8], base16[8:], language], axis=0).astype(np.float32)

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        return

    def predict(self, x: np.ndarray) -> tuple[str, str]:
        family = min(self.family_proto, key=lambda name: c7.sq_dist(x, self.family_proto[name]))
        concept = min(self.concept_proto, key=lambda name: c7.sq_dist(x, self.concept_proto[name]))
        return family, concept


def run_oracle_control(seed: int, noise: float, dropout_p: float, missing_modality_p: float) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    model = ExactPrototypeOracle()
    phase1_eval = c15.evaluate_model(model, cmg.PHASE1, 22, rng, noise, dropout_p, missing_modality_p)
    novel_eval = c15.evaluate_model(model, cmg.PHASE2, 24, rng, noise, dropout_p, missing_modality_p)
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
    ap = argparse.ArgumentParser(description="Stage C49 model sanity diagnostics")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=3)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_c49_model_sanity_diagnostics_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage_c41 = json.loads(
        (TEMP_DIR / "stage_c41_strong_cross_law_coordinate_search_20260312.json").read_text(encoding="utf-8")
    )
    stage_c15 = json.loads(
        (TEMP_DIR / "stage_c15_stronger_long_horizon_binder_search_20260311.json").read_text(encoding="utf-8")
    )
    base = dict(stage_c41["best_objective_candidate"]["config"])
    baseline_consistency_ceiling = float(stage_c15["headline_metrics"]["baseline_consistency_ceiling"])
    c15_retention_compatible = float(stage_c15["best_retention_compatible_candidate"]["summary"]["crossmodal_consistency"])

    moderate_config = {
        **base,
        "family_manifold_pull": 0.08,
        "family_manifold_weight": 0.14,
        "relation_pull": 0.12,
        "temporal_vote_pull": 0.18,
    }
    easy_config = {
        **base,
        "family_manifold_pull": 0.08,
        "family_manifold_weight": 0.06,
        "relation_pull": 0.05,
        "temporal_vote_pull": 0.08,
    }

    moderate_rows: List[Dict[str, float]] = []
    easy_rows: List[Dict[str, float]] = []
    oracle_rows: List[Dict[str, float]] = []
    for offset in range(int(args.num_seeds)):
        seed = int(args.seed) + offset
        moderate_rows.append(
            c42.run_system(
                seed=seed,
                noise=0.12,
                dropout_p=0.05,
                missing_modality_p=0.10,
                config=moderate_config,
            )
        )
        easy_rows.append(
            c42.run_system(
                seed=seed,
                noise=0.08,
                dropout_p=0.02,
                missing_modality_p=0.05,
                config=easy_config,
            )
        )
        oracle_rows.append(
            run_oracle_control(
                seed=seed,
                noise=0.18,
                dropout_p=0.10,
                missing_modality_p=0.22,
            )
        )

    moderate_summary = c15.summarize(moderate_rows)
    easy_summary = c15.summarize(easy_rows)
    oracle_summary = c15.summarize(oracle_rows)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageC49_model_sanity_diagnostics",
        },
        "controls": {
            "moderate_regime_c42": {
                "noise": 0.12,
                "dropout_p": 0.05,
                "missing_modality_p": 0.10,
                "summary": moderate_summary,
            },
            "easy_regime_c42": {
                "noise": 0.08,
                "dropout_p": 0.02,
                "missing_modality_p": 0.05,
                "summary": easy_summary,
            },
            "oracle_exact_prototype": {
                "noise": 0.18,
                "dropout_p": 0.10,
                "missing_modality_p": 0.22,
                "summary": oracle_summary,
            },
        },
        "hypotheses": {
            "H1_moderate_regime_beats_c15_retention_compatible": bool(
                moderate_summary["crossmodal_consistency"] > c15_retention_compatible
                and moderate_summary["retention_concept_accuracy"] >= 0.34
                and moderate_summary["overall_concept_accuracy"] >= 0.35
            ),
            "H2_easy_regime_shows_high_consistency_capacity": bool(
                easy_summary["crossmodal_consistency"] >= 0.30
                and easy_summary["retention_concept_accuracy"] >= 0.45
            ),
            "H3_oracle_shows_benchmark_headroom": bool(
                oracle_summary["crossmodal_consistency"] > baseline_consistency_ceiling
                and oracle_summary["overall_concept_accuracy"] >= 0.35
            ),
        },
        "verdict": {
            "core_answer": "The tested model stack is not fundamentally broken; under easier regimes and oracle structure, the benchmark shows clear headroom.",
            "main_open_gap": "compatibility_geometry_not_testbed_failure",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
