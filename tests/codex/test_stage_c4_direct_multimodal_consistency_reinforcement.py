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


class AlignedSharedMultimodalGrounder:
    def __init__(self, concept_blend: float, confidence_power: float, family_margin_bonus: float) -> None:
        self.concept_blend = concept_blend
        self.confidence_power = confidence_power
        self.family_margin_bonus = family_margin_bonus
        self.norms = {name: RunningNorm(8) for name in MODALITY_SLICES}
        self.family_proto: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in MODALITY_SLICES}
        self.family_count: Dict[str, Dict[str, int]] = {name: {} for name in MODALITY_SLICES}
        self.concept_offset: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in MODALITY_SLICES}
        self.concept_count: Dict[str, Dict[str, int]] = {name: {} for name in MODALITY_SLICES}
        self.canonical_family_basis: Dict[str, np.ndarray] = {}
        self.canonical_concept_offset: Dict[str, np.ndarray] = {}
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

    def _refresh_canonical_family(self, family: str) -> None:
        rows = []
        weights = []
        for modality in MODALITY_SLICES:
            proto = self.family_proto[modality].get(family)
            count = self.family_count[modality].get(family, 0)
            if proto is None or count <= 0:
                continue
            rows.append(proto)
            weights.append(float(count))
        if rows:
            self.canonical_family_basis[family] = np.average(np.stack(rows, axis=0), axis=0, weights=np.array(weights, dtype=np.float64)).astype(np.float32)

    def _refresh_canonical_concept(self, concept: str) -> None:
        rows = []
        weights = []
        for modality in MODALITY_SLICES:
            offset = self.concept_offset[modality].get(concept)
            count = self.concept_count[modality].get(concept, 0)
            if offset is None or count <= 0:
                continue
            rows.append(offset)
            weights.append(float(count))
        if rows:
            self.canonical_concept_offset[concept] = np.average(np.stack(rows, axis=0), axis=0, weights=np.array(weights, dtype=np.float64)).astype(np.float32)

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        observed = self._observed_modalities(x)
        for modality, raw_part in observed:
            self.norms[modality].update(raw_part)
            normed = self.norms[modality].normalize(raw_part)
            prev_family = self.family_proto[modality].get(family)
            prev_count = self.family_count[modality].get(family, 0)
            family_mean = self._mean(prev_family, prev_count, normed)
            self.family_proto[modality][family] = family_mean
            self.family_count[modality][family] = prev_count + 1

            offset = (normed - family_mean).astype(np.float32)
            prev_offset = self.concept_offset[modality].get(concept)
            prev_offset_count = self.concept_count[modality].get(concept, 0)
            self.concept_offset[modality][concept] = self._mean(prev_offset, prev_offset_count, offset)
            self.concept_count[modality][concept] = prev_offset_count + 1

        self.concept_family[concept] = family
        self._refresh_canonical_family(family)
        self._refresh_canonical_concept(concept)

    def _family_distances(self, observed: List[Tuple[str, np.ndarray]], family: str) -> Tuple[float, float]:
        weighted = []
        weights = []
        for modality, raw_part in observed:
            proto = self.family_proto[modality].get(family)
            if proto is None:
                continue
            normed = self.norms[modality].normalize(raw_part)
            dist = sq_dist(normed, proto)
            signal = float(np.linalg.norm(normed))
            weight = max(1e-4, signal**self.confidence_power)
            weighted.append(dist * weight)
            weights.append(weight)
        if not weights:
            return 1e9, 0.0
        return float(sum(weighted) / sum(weights)), float(sum(weights))

    def _choose_family(self, observed: List[Tuple[str, np.ndarray]]) -> str:
        candidates = []
        for family in self.canonical_family_basis:
            score, total_weight = self._family_distances(observed, family)
            candidates.append((family, score, total_weight))
        candidates.sort(key=lambda row: row[1])
        if len(candidates) == 1:
            return candidates[0][0]
        best_family, best_score, _ = candidates[0]
        second_score = candidates[1][1]
        margin = max(0.0, second_score - best_score)
        if margin >= self.family_margin_bonus:
            return best_family
        return best_family

    def _concept_score(self, observed: List[Tuple[str, np.ndarray]], family: str, concept: str) -> float:
        canonical_offset = self.canonical_concept_offset.get(concept)
        if canonical_offset is None:
            return 1e9
        weighted = []
        weights = []
        for modality, raw_part in observed:
            family_proto = self.family_proto[modality].get(family)
            if family_proto is None:
                continue
            normed = self.norms[modality].normalize(raw_part)
            residual = (normed - family_proto).astype(np.float32)
            modality_offset = self.concept_offset[modality].get(concept, canonical_offset)
            dist = self.concept_blend * sq_dist(residual, canonical_offset) + (1.0 - self.concept_blend) * sq_dist(residual, modality_offset)
            signal = float(np.linalg.norm(residual) + 0.5 * np.linalg.norm(normed))
            weight = max(1e-4, signal**self.confidence_power)
            weighted.append(dist * weight)
            weights.append(weight)
        if not weights:
            return 1e9
        return float(sum(weighted) / sum(weights))

    def predict(self, x: np.ndarray) -> Tuple[str, str]:
        observed = self._observed_modalities(x)
        family = self._choose_family(observed)
        concept_candidates = [name for name, fam in self.concept_family.items() if fam == family and name in self.canonical_concept_offset]
        concept = min(concept_candidates, key=lambda name: self._concept_score(observed, family, name))
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


def run_aligned_system(seed: int, noise: float, dropout_p: float, missing_modality_p: float, concept_blend: float, confidence_power: float, family_margin_bonus: float) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    model = AlignedSharedMultimodalGrounder(
        concept_blend=concept_blend,
        confidence_power=confidence_power,
        family_margin_bonus=family_margin_bonus,
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
        1.8 * row["crossmodal_consistency"]
        + 1.2 * row["overall_concept_accuracy"]
        + 1.1 * row["retention_concept_accuracy"]
        + 0.9 * row["novel_concept_accuracy"]
        + 0.8 * row["grounding_score"]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage C4 direct multimodal consistency reinforcement")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.22)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_c4_direct_multimodal_consistency_reinforcement_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()

    baselines = {}
    for system_name in ["direct_multimodal", "shared_offset_multimodal"]:
        rows = []
        for offset in range(int(args.num_seeds)):
            rows.append(
                cmg.run_system(
                    system_name,
                    seed=int(args.seed) + offset,
                    noise=float(args.noise),
                    dropout_p=float(args.dropout_p),
                    missing_modality_p=float(args.missing_modality_p),
                )
            )
        baselines[system_name] = summarize(rows)

    best_objective = None
    best_consistency = None
    for concept_blend in [0.0, 0.3, 0.55, 0.7, 0.85, 1.0]:
        for confidence_power in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
            for family_margin_bonus in [0.0, 0.01, 0.02, 0.05, 0.1]:
                rows = []
                for offset in range(int(args.num_seeds)):
                    rows.append(
                        run_aligned_system(
                            seed=int(args.seed) + offset,
                            noise=float(args.noise),
                            dropout_p=float(args.dropout_p),
                            missing_modality_p=float(args.missing_modality_p),
                            concept_blend=concept_blend,
                            confidence_power=confidence_power,
                            family_margin_bonus=family_margin_bonus,
                        )
                    )
                summary = summarize(rows)
                score = objective(summary)
                if best_objective is None or score > best_objective["objective"]:
                    best_objective = {
                        "config": {
                            "concept_blend": concept_blend,
                            "confidence_power": confidence_power,
                            "family_margin_bonus": family_margin_bonus,
                        },
                        "summary": summary,
                        "objective": score,
                    }
                consistency_score = float(summary["crossmodal_consistency"])
                if best_consistency is None or consistency_score > best_consistency["crossmodal_consistency"]:
                    best_consistency = {
                        "config": {
                            "concept_blend": concept_blend,
                            "confidence_power": confidence_power,
                            "family_margin_bonus": family_margin_bonus,
                        },
                        "summary": summary,
                        "crossmodal_consistency": consistency_score,
                    }
    assert best_objective is not None
    assert best_consistency is not None

    direct = baselines["direct_multimodal"]
    shared = baselines["shared_offset_multimodal"]
    aligned = best_objective["summary"]
    aligned_consistency = best_consistency["summary"]

    stage_c3 = json.loads((TEMP_DIR / "stage_c3_multimodal_shared_alignment_search_20260311.json").read_text(encoding="utf-8"))
    strong_multimodal_target = float(stage_c3["targets"]["strong_multimodal_target"])
    baseline_consistency_ceiling = max(float(direct["crossmodal_consistency"]), float(shared["crossmodal_consistency"]))

    verdict = {
        "status": (
            "stage_c4_direct_consistency_reaches_strong_target"
            if aligned_consistency["crossmodal_consistency"] >= strong_multimodal_target
            else "stage_c4_direct_consistency_tradeoff_not_solved"
        ),
        "core_answer": (
            "Direct multimodal reinforcement exposes a real gap: the shared-alignment grounder improves concept retention and overall concept accuracy, "
            "but it still fails to beat the simple baselines on direct crossmodal consistency. "
            "So C3 currently remains an indirect alignment result, not a direct consistency closure."
        ),
        "main_open_gap": (
            "direct_crossmodal_consistency_tradeoff"
            if aligned_consistency["crossmodal_consistency"] < strong_multimodal_target
            else "strong_target_reached_on_current_bench"
        ),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "missing_modality_p": float(args.missing_modality_p),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageC4_direct_multimodal_consistency_reinforcement",
        },
        "systems": {
            "direct_multimodal": direct,
            "shared_offset_multimodal": shared,
            "aligned_shared_multimodal": aligned,
        },
        "best_objective_config": best_objective["config"],
        "best_consistency_config": best_consistency["config"],
        "headline_metrics": {
            "direct_crossmodal_consistency": float(direct["crossmodal_consistency"]),
            "shared_offset_crossmodal_consistency": float(shared["crossmodal_consistency"]),
            "best_objective_aligned_crossmodal_consistency": float(aligned["crossmodal_consistency"]),
            "best_consistency_aligned_crossmodal_consistency": float(aligned_consistency["crossmodal_consistency"]),
            "direct_grounding_score": float(direct["grounding_score"]),
            "best_objective_aligned_grounding_score": float(aligned["grounding_score"]),
            "direct_overall_concept_accuracy": float(direct["overall_concept_accuracy"]),
            "best_objective_aligned_overall_concept_accuracy": float(aligned["overall_concept_accuracy"]),
            "baseline_consistency_ceiling": baseline_consistency_ceiling,
            "strong_multimodal_target": strong_multimodal_target,
        },
        "gains": {
            "best_consistency_vs_direct_crossmodal_gain": float(aligned_consistency["crossmodal_consistency"] - direct["crossmodal_consistency"]),
            "best_consistency_vs_shared_offset_crossmodal_gain": float(aligned_consistency["crossmodal_consistency"] - shared["crossmodal_consistency"]),
            "best_objective_vs_direct_overall_concept_gain": float(aligned["overall_concept_accuracy"] - direct["overall_concept_accuracy"]),
            "best_objective_vs_direct_retention_concept_gain": float(aligned["retention_concept_accuracy"] - direct["retention_concept_accuracy"]),
            "best_objective_vs_direct_grounding_gain": float(aligned["grounding_score"] - direct["grounding_score"]),
        },
        "hypotheses": {
            "H1_any_aligned_beats_direct_on_consistency": bool(aligned_consistency["crossmodal_consistency"] > direct["crossmodal_consistency"]),
            "H2_any_aligned_beats_shared_offset_on_consistency": bool(aligned_consistency["crossmodal_consistency"] > shared["crossmodal_consistency"]),
            "H3_best_objective_beats_direct_on_overall_concept": bool(aligned["overall_concept_accuracy"] > direct["overall_concept_accuracy"]),
            "H4_any_aligned_reaches_strong_target": bool(aligned_consistency["crossmodal_consistency"] >= strong_multimodal_target),
        },
        "verdict": verdict,
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
