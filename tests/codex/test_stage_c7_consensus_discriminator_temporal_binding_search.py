from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import test_continuous_multimodal_grounding_proto as cmg


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"
MODALITY_SLICES = {
    "visual": slice(0, 8),
    "tactile": slice(8, 16),
    "language": slice(16, 24),
}


def sq_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.square(a - b)))


class RunningNorm:
    def __init__(self, dim: int) -> None:
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float32)
        self.m2 = np.zeros(dim, dtype=np.float32)

    def update(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean = self.mean + delta / float(self.count)
        delta2 = x - self.mean
        self.m2 = self.m2 + delta * delta2

    def std(self) -> np.ndarray:
        if self.count < 2:
            return np.ones_like(self.mean)
        var = self.m2 / float(max(1, self.count - 1))
        return np.sqrt(np.maximum(var, 1e-4)).astype(np.float32)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std()).astype(np.float32)


class ConsensusDiscriminatorTemporalGrounder:
    def __init__(
        self,
        direct_weight: float,
        modality_weight: float,
        canonical_weight: float,
        trace_weight: float,
        family_weight: float,
        trace_pull: float,
    ) -> None:
        self.direct_weight = direct_weight
        self.modality_weight = modality_weight
        self.canonical_weight = canonical_weight
        self.trace_weight = trace_weight
        self.family_weight = family_weight
        self.trace_pull = trace_pull

        self.norms = {name: RunningNorm(8) for name in MODALITY_SLICES}
        self.full_family_proto: Dict[str, np.ndarray] = {}
        self.full_family_count: Dict[str, int] = {}
        self.full_concept_proto: Dict[str, np.ndarray] = {}
        self.full_concept_count: Dict[str, int] = {}
        self.modality_family_proto: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in MODALITY_SLICES}
        self.modality_family_count: Dict[str, Dict[str, int]] = {name: {} for name in MODALITY_SLICES}
        self.modality_concept_proto: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in MODALITY_SLICES}
        self.modality_concept_count: Dict[str, Dict[str, int]] = {name: {} for name in MODALITY_SLICES}
        self.canonical_concept: Dict[str, np.ndarray] = {}
        self.temporal_trace: Dict[str, np.ndarray] = {}
        self.separation_penalty: Dict[str, float] = {}
        self.concept_family: Dict[str, str] = {}

    @staticmethod
    def _mean(prev: np.ndarray | None, count: int, x: np.ndarray) -> np.ndarray:
        if prev is None:
            return x.astype(np.float32).copy()
        alpha = 1.0 / float(count + 1)
        return ((1.0 - alpha) * prev + alpha * x).astype(np.float32)

    def _observed_modalities(self, x: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        rows = []
        for name, sl in MODALITY_SLICES.items():
            part = x[sl].astype(np.float32)
            if float(np.linalg.norm(part)) > 1e-5:
                rows.append((name, part))
        return rows

    def _refresh_canonical(self, concept: str) -> None:
        rows = []
        weights = []
        for modality in MODALITY_SLICES:
            proto = self.modality_concept_proto[modality].get(concept)
            count = self.modality_concept_count[modality].get(concept, 0)
            if proto is None or count <= 0:
                continue
            rows.append(proto)
            weights.append(float(count))
        if not rows:
            return
        canonical = np.average(
            np.stack(rows, axis=0),
            axis=0,
            weights=np.array(weights, dtype=np.float64),
        ).astype(np.float32)
        self.canonical_concept[concept] = canonical
        prev_trace = self.temporal_trace.get(concept)
        if prev_trace is None:
            self.temporal_trace[concept] = canonical.copy()
        else:
            self.temporal_trace[concept] = (
                (1.0 - self.trace_pull) * prev_trace + self.trace_pull * canonical
            ).astype(np.float32)

        family = self.concept_family.get(concept)
        if family is None:
            return
        negatives = []
        for other, other_family in self.concept_family.items():
            if other == concept or other_family != family or other not in self.canonical_concept:
                continue
            negatives.append(sq_dist(canonical, self.canonical_concept[other]))
        if negatives:
            self.separation_penalty[concept] = float(1.0 / max(1e-4, min(negatives)))
        else:
            self.separation_penalty[concept] = 0.0

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        self.full_family_proto[family] = self._mean(self.full_family_proto.get(family), self.full_family_count.get(family, 0), x)
        self.full_family_count[family] = self.full_family_count.get(family, 0) + 1
        self.full_concept_proto[concept] = self._mean(self.full_concept_proto.get(concept), self.full_concept_count.get(concept, 0), x)
        self.full_concept_count[concept] = self.full_concept_count.get(concept, 0) + 1

        observed = self._observed_modalities(x)
        for modality, raw_part in observed:
            self.norms[modality].update(raw_part)
            normed = self.norms[modality].normalize(raw_part)
            self.modality_family_proto[modality][family] = self._mean(
                self.modality_family_proto[modality].get(family),
                self.modality_family_count[modality].get(family, 0),
                normed,
            )
            self.modality_family_count[modality][family] = self.modality_family_count[modality].get(family, 0) + 1
            self.modality_concept_proto[modality][concept] = self._mean(
                self.modality_concept_proto[modality].get(concept),
                self.modality_concept_count[modality].get(concept, 0),
                normed,
            )
            self.modality_concept_count[modality][concept] = self.modality_concept_count[modality].get(concept, 0) + 1

        self.concept_family[concept] = family
        self._refresh_canonical(concept)

    def _family_score(self, x: np.ndarray, observed: List[Tuple[str, np.ndarray]], family: str) -> float:
        direct_score = sq_dist(x, self.full_family_proto[family]) if family in self.full_family_proto else 1e9
        modality_scores = []
        for modality, raw_part in observed:
            proto = self.modality_family_proto[modality].get(family)
            if proto is None:
                continue
            normed = self.norms[modality].normalize(raw_part)
            modality_scores.append(sq_dist(normed, proto))
        modality_score = float(np.mean(modality_scores)) if modality_scores else 1e9
        return self.direct_weight * direct_score + self.family_weight * modality_score

    def _concept_score(self, x: np.ndarray, observed: List[Tuple[str, np.ndarray]], family: str, concept: str, family_score: float) -> float:
        if self.concept_family.get(concept) != family:
            return 1e9
        score = 0.0
        if concept in self.full_concept_proto:
            score += self.direct_weight * sq_dist(x, self.full_concept_proto[concept])
        else:
            score += 1e9

        modality_scores = []
        canonical_scores = []
        trace_scores = []
        canonical = self.canonical_concept.get(concept)
        trace = self.temporal_trace.get(concept)
        for modality, raw_part in observed:
            normed = self.norms[modality].normalize(raw_part)
            proto = self.modality_concept_proto[modality].get(concept)
            if proto is not None:
                modality_scores.append(sq_dist(normed, proto))
            if canonical is not None:
                canonical_scores.append(sq_dist(normed, canonical))
            if trace is not None:
                trace_scores.append(sq_dist(normed, trace))
        if modality_scores:
            score += self.modality_weight * float(np.mean(modality_scores))
        if canonical_scores:
            score += self.canonical_weight * float(np.mean(canonical_scores))
        if trace_scores:
            score += self.trace_weight * float(np.mean(trace_scores))
        score += 0.2 * family_score
        score += self.separation_penalty.get(concept, 0.0)
        return float(score)

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        observed = self._observed_modalities(x)
        families = list({fam for fam in self.concept_family.values()})
        family_scores = {family: self._family_score(x, observed, family) for family in families}
        family = min(family_scores, key=family_scores.get)
        concepts = [name for name, fam in self.concept_family.items() if fam == family]
        concept = min(concepts, key=lambda name: self._concept_score(x, observed, family, name, family_scores[family]))
        return family, concept


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


def crossmodal_consistency(
    model,
    concepts: List[str],
    repeats: int,
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
) -> float:
    ok = 0
    total = 0
    for _ in range(repeats):
        for concept in concepts:
            family = cmg.concept_family(concept)
            full = cmg.sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p=0.0)
            visual_only = full.copy()
            visual_only[8:] = 0.0
            tactile_only = full.copy()
            tactile_only[:8] = 0.0
            tactile_only[16:] = 0.0
            lang_only = full.copy()
            lang_only[:16] = 0.0
            fam_v, con_v = model.predict(visual_only)
            fam_t, con_t = model.predict(tactile_only)
            fam_l, con_l = model.predict(lang_only)
            ok += int(
                fam_v == family
                and fam_t == family
                and fam_l == family
                and con_v == con_t == con_l == concept
            )
            total += 1
    return float(ok / max(1, total))


def temporal_binding_score(
    model,
    concepts: List[str],
    repeats: int,
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
) -> float:
    ok = 0
    total = 0
    for _ in range(repeats):
        for concept in concepts:
            family = cmg.concept_family(concept)
            x1 = cmg.sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p=0.45)
            x2 = cmg.sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p=0.45)
            fam1, con1 = model.predict(x1)
            fam2, con2 = model.predict(x2)
            ok += int(fam1 == family and fam2 == family and con1 == con2 == concept)
            total += 1
    return float(ok / max(1, total))


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
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    model = ConsensusDiscriminatorTemporalGrounder(
        direct_weight=direct_weight,
        modality_weight=modality_weight,
        canonical_weight=canonical_weight,
        trace_weight=trace_weight,
        family_weight=family_weight,
        trace_pull=trace_pull,
    )

    for _ in range(42):
        for family, concepts in cmg.PHASE1.items():
            for concept in concepts:
                model.train(cmg.sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p), family, concept)

    phase1_eval = evaluate_model(model, cmg.PHASE1, repeats=22, rng=rng, noise=noise, dropout_p=dropout_p, missing_modality_p=missing_modality_p)

    for _ in range(3):
        for family, concepts in cmg.PHASE2.items():
            for concept in concepts:
                model.train(cmg.sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p), family, concept)

    novel_eval = evaluate_model(model, cmg.PHASE2, repeats=24, rng=rng, noise=noise, dropout_p=dropout_p, missing_modality_p=missing_modality_p)

    for _ in range(18):
        for family, concepts in cmg.PHASE2.items():
            for concept in concepts:
                model.train(cmg.sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p), family, concept)

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
    consistency = crossmodal_consistency(model, concepts, repeats=10, rng=rng, noise=noise, dropout_p=dropout_p)
    temporal_score = temporal_binding_score(model, concepts, repeats=10, rng=rng, noise=noise, dropout_p=dropout_p)

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


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(rows[0].keys())
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def objective(row: Dict[str, float]) -> float:
    return float(
        3.0 * row["crossmodal_consistency"]
        + 1.5 * row["temporal_binding_score"]
        + 1.0 * row["overall_concept_accuracy"]
        + 0.9 * row["retention_concept_accuracy"]
        + 0.6 * row["grounding_score"]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage C7 consensus discriminator temporal binding search")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.22)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_c7_consensus_discriminator_temporal_binding_search_20260311.json",
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

    best_objective = None
    best_consistency = None
    best_anti_tradeoff = None

    for direct_weight in [0.5, 1.0]:
        for modality_weight in [0.5, 1.0]:
            for canonical_weight in [0.0, 0.25]:
                for trace_weight in [0.0, 0.25]:
                    for family_weight in [0.0, 0.1]:
                        for trace_pull in [0.0, 0.2]:
                            rows = []
                            for offset in range(int(args.num_seeds)):
                                rows.append(
                                    run_system(
                                        seed=int(args.seed) + offset,
                                        noise=float(args.noise),
                                        dropout_p=float(args.dropout_p),
                                        missing_modality_p=float(args.missing_modality_p),
                                        direct_weight=direct_weight,
                                        modality_weight=modality_weight,
                                        canonical_weight=canonical_weight,
                                        trace_weight=trace_weight,
                                        family_weight=family_weight,
                                        trace_pull=trace_pull,
                                    )
                                )
                            summary = summarize(rows)
                            score = objective(summary)
                            candidate = {
                                "config": {
                                    "direct_weight": direct_weight,
                                    "modality_weight": modality_weight,
                                    "canonical_weight": canonical_weight,
                                    "trace_weight": trace_weight,
                                    "family_weight": family_weight,
                                    "trace_pull": trace_pull,
                                },
                                "summary": summary,
                                "objective": score,
                            }
                            if best_objective is None or score > best_objective["objective"]:
                                best_objective = candidate
                            if best_consistency is None or summary["crossmodal_consistency"] > best_consistency["summary"]["crossmodal_consistency"]:
                                best_consistency = candidate
                            if (
                                summary["crossmodal_consistency"] >= baseline_consistency_ceiling
                                and summary["overall_concept_accuracy"] >= direct["overall_concept_accuracy"]
                                and summary["retention_concept_accuracy"] >= direct["retention_concept_accuracy"]
                            ):
                                if best_anti_tradeoff is None or score > best_anti_tradeoff["objective"]:
                                    best_anti_tradeoff = candidate

    assert best_objective is not None
    assert best_consistency is not None

    stage_c6 = json.loads((TEMP_DIR / "stage_c6_modality_consensus_cycle_consistency_search_20260311.json").read_text(encoding="utf-8"))
    strong_multimodal_target = float(stage_c6["headline_metrics"]["strong_multimodal_target"])

    if best_anti_tradeoff is not None:
        status = "stage_c7_partial_direct_anti_tradeoff_found"
        core_answer = (
            "An explicit consensus discriminator with temporal binding finds a family that preserves baseline direct consistency while lifting concept stability."
        )
        main_open_gap = (
            "anti_tradeoff_found_but_still_below_strong_target"
            if best_anti_tradeoff["summary"]["crossmodal_consistency"] < strong_multimodal_target
            else "strong_target_reached"
        )
    else:
        status = "stage_c7_consensus_discriminator_still_not_enough"
        core_answer = (
            "Even with an explicit consensus discriminator and temporal binding, no searched family preserved baseline direct consistency while lifting retention and overall concept accuracy."
        )
        main_open_gap = "direct_consistency_requires_stronger_object_persistence_mechanism"

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "missing_modality_p": float(args.missing_modality_p),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageC7_consensus_discriminator_temporal_binding_search",
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
            "best_consistency_temporal_binding": float(best_consistency["summary"]["temporal_binding_score"]),
            "best_consistency_overall_concept": float(best_consistency["summary"]["overall_concept_accuracy"]),
            "best_consistency_retention_concept": float(best_consistency["summary"]["retention_concept_accuracy"]),
        },
        "best_objective_candidate": best_objective,
        "best_consistency_candidate": best_consistency,
        "best_anti_tradeoff_candidate": best_anti_tradeoff,
        "hypotheses": {
            "H1_any_candidate_beats_baseline_consistency": bool(best_consistency["summary"]["crossmodal_consistency"] >= baseline_consistency_ceiling),
            "H2_any_candidate_reaches_strong_target": bool(best_consistency["summary"]["crossmodal_consistency"] >= strong_multimodal_target),
            "H3_any_candidate_solves_anti_tradeoff": bool(best_anti_tradeoff is not None),
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
