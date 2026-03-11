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


class AntiTradeoffConsensusGrounder:
    def __init__(
        self,
        anchor_pull: float,
        anchor_blend: float,
        family_blend: float,
        replay_pull: float,
    ) -> None:
        self.anchor_pull = anchor_pull
        self.anchor_blend = anchor_blend
        self.family_blend = family_blend
        self.replay_pull = replay_pull
        self.norms = {name: RunningNorm(8) for name in MODALITY_SLICES}
        self.family_proto: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in MODALITY_SLICES}
        self.family_count: Dict[str, Dict[str, int]] = {name: {} for name in MODALITY_SLICES}
        self.concept_proto: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in MODALITY_SLICES}
        self.concept_count: Dict[str, Dict[str, int]] = {name: {} for name in MODALITY_SLICES}
        self.canonical_concept: Dict[str, np.ndarray] = {}
        self.concept_family: Dict[str, str] = {}

    @staticmethod
    def _mean(prev: np.ndarray | None, count: int, x: np.ndarray) -> np.ndarray:
        if prev is None:
            return x.astype(np.float32).copy()
        alpha = 1.0 / float(count + 1)
        return ((1.0 - alpha) * prev + alpha * x).astype(np.float32)

    def _observed_modalities(self, x: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        observed = []
        for name, sl in MODALITY_SLICES.items():
            part = x[sl].astype(np.float32)
            if float(np.linalg.norm(part)) > 1e-5:
                observed.append((name, part))
        return observed

    def _refresh_canonical_concept(self, concept: str) -> None:
        rows = []
        weights = []
        for modality in MODALITY_SLICES:
            proto = self.concept_proto[modality].get(concept)
            count = self.concept_count[modality].get(concept, 0)
            if proto is None or count <= 0:
                continue
            rows.append(proto)
            weights.append(float(count))
        if rows:
            self.canonical_concept[concept] = np.average(
                np.stack(rows, axis=0),
                axis=0,
                weights=np.array(weights, dtype=np.float64),
            ).astype(np.float32)

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        observed = self._observed_modalities(x)
        for modality, raw_part in observed:
            self.norms[modality].update(raw_part)
            normed = self.norms[modality].normalize(raw_part)

            prev_family = self.family_proto[modality].get(family)
            prev_family_count = self.family_count[modality].get(family, 0)
            self.family_proto[modality][family] = self._mean(prev_family, prev_family_count, normed)
            self.family_count[modality][family] = prev_family_count + 1

            target = normed
            if concept in self.canonical_concept:
                target = ((1.0 - self.anchor_pull) * normed + self.anchor_pull * self.canonical_concept[concept]).astype(np.float32)
            prev_concept = self.concept_proto[modality].get(concept)
            prev_concept_count = self.concept_count[modality].get(concept, 0)
            self.concept_proto[modality][concept] = self._mean(prev_concept, prev_concept_count, target)
            self.concept_count[modality][concept] = prev_concept_count + 1

        self.concept_family[concept] = family
        self._refresh_canonical_concept(concept)

        if self.replay_pull > 0.0 and concept in self.canonical_concept:
            canonical = self.canonical_concept[concept]
            for modality, _ in observed:
                proto = self.concept_proto[modality].get(concept)
                if proto is None:
                    continue
                self.concept_proto[modality][concept] = (
                    (1.0 - self.replay_pull) * proto + self.replay_pull * canonical
                ).astype(np.float32)

    def _family_score(self, observed: List[Tuple[str, np.ndarray]], family: str) -> float:
        dists = []
        for modality, raw_part in observed:
            proto = self.family_proto[modality].get(family)
            if proto is None:
                continue
            normed = self.norms[modality].normalize(raw_part)
            dists.append(sq_dist(normed, proto))
        return float(np.mean(dists)) if dists else 1e9

    def _concept_score(self, observed: List[Tuple[str, np.ndarray]], family: str, concept: str) -> float:
        if self.concept_family.get(concept) != family:
            return 1e9
        canonical = self.canonical_concept.get(concept)
        dists = []
        for modality, raw_part in observed:
            proto = self.concept_proto[modality].get(concept)
            if proto is None:
                continue
            normed = self.norms[modality].normalize(raw_part)
            score = sq_dist(normed, proto)
            if canonical is not None:
                score += self.anchor_blend * sq_dist(normed, canonical)
            dists.append(score)
        if not dists:
            return 1e9
        family_score = self._family_score(observed, family)
        return float(np.mean(dists) + self.family_blend * family_score)

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        observed = self._observed_modalities(x)
        families = list({fam for fam in self.concept_family.values()})
        family = min(families, key=lambda name: self._family_score(observed, name))
        concepts = [name for name, fam in self.concept_family.items() if fam == family]
        concept = min(concepts, key=lambda name: self._concept_score(observed, family, name))
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


def crossmodal_consistency(model, concepts: List[str], repeats: int, rng: np.random.Generator, noise: float, dropout_p: float) -> float:
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


def run_system(
    seed: int,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    anchor_pull: float,
    anchor_blend: float,
    family_blend: float,
    replay_pull: float,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    model = AntiTradeoffConsensusGrounder(
        anchor_pull=anchor_pull,
        anchor_blend=anchor_blend,
        family_blend=family_blend,
        replay_pull=replay_pull,
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
    consistency = crossmodal_consistency(
        model,
        [concept for family in cmg.FAMILIES for concept in cmg.PHASE1[family] + cmg.PHASE2[family]],
        repeats=10,
        rng=rng,
        noise=noise,
        dropout_p=dropout_p,
    )

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
        "grounding_score": grounding_score,
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(rows[0].keys())
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def objective(row: Dict[str, float]) -> float:
    return float(
        2.8 * row["crossmodal_consistency"]
        + 1.1 * row["overall_concept_accuracy"]
        + 1.0 * row["retention_concept_accuracy"]
        + 0.8 * row["novel_concept_accuracy"]
        + 0.6 * row["grounding_score"]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage C5 direct multimodal anti-tradeoff search")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.22)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_c5_direct_multimodal_anti_tradeoff_search_20260311.json",
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

    for anchor_pull in [0.0, 0.15, 0.3]:
        for anchor_blend in [0.0, 0.25, 0.5]:
            for family_blend in [0.0, 0.1, 0.25]:
                for replay_pull in [0.0, 0.05]:
                    rows = []
                    for offset in range(int(args.num_seeds)):
                        rows.append(
                            run_system(
                                seed=int(args.seed) + offset,
                                noise=float(args.noise),
                                dropout_p=float(args.dropout_p),
                                missing_modality_p=float(args.missing_modality_p),
                                anchor_pull=anchor_pull,
                                anchor_blend=anchor_blend,
                                family_blend=family_blend,
                                replay_pull=replay_pull,
                            )
                        )
                    summary = summarize(rows)
                    score = objective(summary)
                    candidate = {
                        "config": {
                            "anchor_pull": anchor_pull,
                            "anchor_blend": anchor_blend,
                            "family_blend": family_blend,
                            "replay_pull": replay_pull,
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

    stage_c4 = json.loads((TEMP_DIR / "stage_c4_direct_multimodal_consistency_reinforcement_20260311.json").read_text(encoding="utf-8"))
    strong_multimodal_target = float(stage_c4["headline_metrics"]["strong_multimodal_target"])

    if best_anti_tradeoff is not None:
        status = "stage_c5_partial_direct_anti_tradeoff_found"
        core_answer = (
            "A real anti-tradeoff family exists: it keeps direct crossmodal consistency at or above baseline while also lifting overall concept and retention."
        )
        main_open_gap = "anti_tradeoff_family_still_far_from_strong_target"
    else:
        status = "stage_c5_no_direct_anti_tradeoff_family_found_yet"
        core_answer = (
            "No searched family simultaneously preserved baseline direct crossmodal consistency and improved concept stability. "
            "The direct multimodal tradeoff remains the main unsolved defect."
        )
        main_open_gap = "no_family_preserves_baseline_consistency_while_lifting_retention"

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "missing_modality_p": float(args.missing_modality_p),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageC5_direct_multimodal_anti_tradeoff_search",
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
            "best_consistency_overall_concept": float(best_consistency["summary"]["overall_concept_accuracy"]),
            "best_consistency_retention_concept": float(best_consistency["summary"]["retention_concept_accuracy"]),
        },
        "best_objective_candidate": best_objective,
        "best_consistency_candidate": best_consistency,
        "best_anti_tradeoff_candidate": best_anti_tradeoff,
        "hypotheses": {
            "H1_any_candidate_beats_baseline_consistency": bool(best_consistency["summary"]["crossmodal_consistency"] >= baseline_consistency_ceiling),
            "H2_any_candidate_reaches_strong_target": bool(best_consistency["summary"]["crossmodal_consistency"] >= strong_multimodal_target),
            "H3_anti_tradeoff_candidate_exists": bool(best_anti_tradeoff is not None),
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
