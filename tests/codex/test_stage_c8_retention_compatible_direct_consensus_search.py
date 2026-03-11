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


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(rows[0].keys())
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def objective(row: Dict[str, float]) -> float:
    return float(
        2.8 * row["crossmodal_consistency"]
        + 1.3 * row["retention_concept_accuracy"]
        + 1.0 * row["overall_concept_accuracy"]
        + 1.0 * row["temporal_binding_score"]
        + 0.5 * row["grounding_score"]
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
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    model = c7.ConsensusDiscriminatorTemporalGrounder(
        direct_weight=direct_weight,
        modality_weight=modality_weight,
        canonical_weight=canonical_weight,
        trace_weight=trace_weight,
        family_weight=family_weight,
        trace_pull=trace_pull,
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
    ap = argparse.ArgumentParser(description="Stage C8 retention-compatible direct consensus search")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.22)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_c8_retention_compatible_direct_consensus_search_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    direct_rows = []
    shared_rows = []
    for offset in range(int(args.num_seeds)):
        seed = int(args.seed) + offset
        direct_rows.append(
            cmg.run_system(
                "direct_multimodal",
                seed=seed,
                noise=float(args.noise),
                dropout_p=float(args.dropout_p),
                missing_modality_p=float(args.missing_modality_p),
            )
        )
        shared_rows.append(
            cmg.run_system(
                "shared_offset_multimodal",
                seed=seed,
                noise=float(args.noise),
                dropout_p=float(args.dropout_p),
                missing_modality_p=float(args.missing_modality_p),
            )
        )

    direct = summarize(direct_rows)
    shared = summarize(shared_rows)
    baseline_consistency_ceiling = max(float(direct["crossmodal_consistency"]), float(shared["crossmodal_consistency"]))

    stage_c7 = json.loads((TEMP_DIR / "stage_c7_consensus_discriminator_temporal_binding_search_20260311.json").read_text(encoding="utf-8"))
    best_c7 = stage_c7["best_consistency_candidate"]["config"]
    strong_multimodal_target = float(stage_c7["headline_metrics"]["strong_multimodal_target"])

    best_objective = None
    best_consistency = None
    best_retention_compatible = None

    for replay_steps in [0, 1, 2, 3]:
        for replay_boost in [0.0, 0.25, 0.5]:
            for replay_after_novel_steps in [0, 1, 2]:
                rows = []
                for offset in range(int(args.num_seeds)):
                    rows.append(
                        run_system(
                            seed=int(args.seed) + offset,
                            noise=float(args.noise),
                            dropout_p=float(args.dropout_p),
                            missing_modality_p=float(args.missing_modality_p),
                            direct_weight=float(best_c7["direct_weight"]),
                            modality_weight=float(best_c7["modality_weight"]),
                            canonical_weight=float(best_c7["canonical_weight"]),
                            trace_weight=float(best_c7["trace_weight"]),
                            family_weight=float(best_c7["family_weight"]),
                            trace_pull=float(best_c7["trace_pull"]),
                            replay_steps=replay_steps,
                            replay_boost=replay_boost,
                            replay_after_novel_steps=replay_after_novel_steps,
                        )
                    )
                summary = summarize(rows)
                candidate = {
                    "config": {
                        **best_c7,
                        "replay_steps": replay_steps,
                        "replay_boost": replay_boost,
                        "replay_after_novel_steps": replay_after_novel_steps,
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
                    and summary["retention_concept_accuracy"] >= direct["retention_concept_accuracy"]
                    and summary["overall_concept_accuracy"] >= direct["overall_concept_accuracy"]
                ):
                    if best_retention_compatible is None or candidate["objective"] > best_retention_compatible["objective"]:
                        best_retention_compatible = candidate

    assert best_objective is not None
    assert best_consistency is not None

    if best_retention_compatible is not None:
        status = "stage_c8_retention_compatible_direct_consensus_found"
        core_answer = (
            "Replay-stabilized direct consensus finds a family that keeps direct crossmodal consistency above baseline while restoring retention and overall concept accuracy."
        )
        main_open_gap = (
            "retention_compatible_family_still_below_strong_target"
            if best_retention_compatible["summary"]["crossmodal_consistency"] < strong_multimodal_target
            else "strong_target_reached"
        )
    else:
        status = "stage_c8_replay_improves_retention_but_not_full_anti_tradeoff"
        core_answer = (
            "Replay helps retention pressure, but no searched family yet preserves baseline direct consistency while also restoring retention and overall concept accuracy."
        )
        main_open_gap = "retention_compatible_direct_consensus_not_found_yet"

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "missing_modality_p": float(args.missing_modality_p),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageC8_retention_compatible_direct_consensus_search",
        },
        "baselines": {
            "direct_multimodal": direct,
            "shared_offset_multimodal": shared,
        },
        "headline_metrics": {
            "baseline_consistency_ceiling": baseline_consistency_ceiling,
            "strong_multimodal_target": strong_multimodal_target,
            "best_objective_consistency": float(best_objective["summary"]["crossmodal_consistency"]),
            "best_consistency_value": float(best_consistency["summary"]["crossmodal_consistency"]),
            "best_consistency_retention": float(best_consistency["summary"]["retention_concept_accuracy"]),
            "best_consistency_overall_concept": float(best_consistency["summary"]["overall_concept_accuracy"]),
        },
        "best_objective_candidate": best_objective,
        "best_consistency_candidate": best_consistency,
        "best_retention_compatible_candidate": best_retention_compatible,
        "hypotheses": {
            "H1_any_candidate_keeps_consistency_above_baseline": bool(best_consistency["summary"]["crossmodal_consistency"] >= baseline_consistency_ceiling),
            "H2_any_candidate_reaches_strong_target": bool(best_consistency["summary"]["crossmodal_consistency"] >= strong_multimodal_target),
            "H3_retention_compatible_candidate_exists": bool(best_retention_compatible is not None),
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
