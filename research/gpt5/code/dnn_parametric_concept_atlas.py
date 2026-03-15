from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def load_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8-sig")
    return __import__("json").loads(text)


@dataclass
class ConceptEntry:
    name: str
    family: str
    full_state: np.ndarray
    basis: np.ndarray
    offset: np.ndarray
    dominant_offset_dims: List[int]
    source: str


@dataclass
class FamilySupport:
    obj_dims: List[int]
    mem_dims: List[int]
    id_dims: List[int]
    disc_dims: List[int]
    family_radius: float
    intrusion_gap: float


class UnifiedParametricConceptAtlas:
    def __init__(
        self,
        concept_entries: List[ConceptEntry],
        family_supports: Dict[str, FamilySupport],
        family_centers: Dict[str, np.ndarray],
        family_rank_dims: Dict[str, List[int]],
        global_recurrent_dims: List[int],
        dim: int,
    ):
        self.concept_entries = concept_entries
        self.family_supports = family_supports
        self.family_centers = family_centers
        self.family_rank_dims = family_rank_dims
        self.global_recurrent_dims = global_recurrent_dims
        self.dim = dim

    @classmethod
    def from_artifacts(cls, root: Path, synth_per_family: int = 64, seed: int = 17) -> "UnifiedParametricConceptAtlas":
        temp = root / "tests" / "codex_temp"
        family_atlas = load_json(temp / "theory_track_concept_family_atlas_analysis_20260312.json")
        operators = load_json(temp / "theory_track_family_conditioned_projection_operators_20260312.json")
        large_inventory = load_json(temp / "theory_track_large_scale_concept_inventory_analysis_20260312.json")

        family_centers = {
            name: np.array(row["family_center"], dtype=np.float32)
            for name, row in family_atlas["family_atlas"].items()
        }
        dim = next(iter(family_centers.values())).shape[0]
        family_rank_dims = {
            name: [int(x) for x in row["top_basis_dims"]]
            for name, row in large_inventory["family_rank_structure"].items()
        }
        family_supports = {
            name: FamilySupport(
                obj_dims=[int(x) for x in row["P_obj_family"]["support_dims"]],
                mem_dims=[int(x) for x in row["P_mem_family"]["support_dims"]],
                id_dims=[int(x) for x in row["P_id_family"]["support_dims"]],
                disc_dims=[int(x) for x in row["P_disc_family"]["support_dims"]],
                family_radius=float(row["family_radius"]),
                intrusion_gap=float(row["intrusion_gap"]),
            )
            for name, row in operators["core_operators"].items()
        }
        concept_entries: List[ConceptEntry] = []
        for name, row in family_atlas["concept_decomposition_examples"].items():
            basis = np.concatenate(
                [
                    np.array(row["vt_family_basis"], dtype=np.float32),
                    np.array(row["lang_family_basis"], dtype=np.float32),
                ],
                axis=0,
            )
            offset = np.concatenate(
                [
                    np.array(row["vt_concept_offset"], dtype=np.float32),
                    np.array(row["lang_concept_offset"], dtype=np.float32),
                ],
                axis=0,
            )
            concept_entries.append(
                ConceptEntry(
                    name=name,
                    family=row["family"],
                    full_state=np.array(row["full_state"], dtype=np.float32),
                    basis=basis,
                    offset=offset,
                    dominant_offset_dims=[int(x) for x in row["dominant_offset_dims"]],
                    source="exemplar",
                )
            )

        rng = np.random.default_rng(seed)
        global_recurrent_dims = [int(x) for x in large_inventory["global_recurrent_dims"]]
        mean_offset_norm = float(large_inventory["headline_metrics"]["mean_offset_norm"])

        for family, center in family_centers.items():
            rank_dims = family_rank_dims[family]
            support = family_supports[family]
            for idx in range(synth_per_family):
                offset = np.zeros(dim, dtype=np.float32)
                local_dims = rng.choice(rank_dims[:6], size=3, replace=False)
                local_weights = rng.uniform(0.15, 0.55, size=3).astype(np.float32)
                for dim_idx, weight in zip(local_dims, local_weights):
                    offset[int(dim_idx)] += float(weight) * mean_offset_norm
                recurrent_dims = rng.choice(global_recurrent_dims[:6], size=2, replace=False)
                recurrent_weights = rng.uniform(0.06, 0.20, size=2).astype(np.float32)
                for dim_idx, weight in zip(recurrent_dims, recurrent_weights):
                    offset[int(dim_idx)] += float(weight) * mean_offset_norm
                for dim_idx in support.id_dims[:2]:
                    offset[int(dim_idx)] += float(rng.uniform(0.08, 0.20)) * mean_offset_norm
                full_state = center + offset
                concept_entries.append(
                    ConceptEntry(
                        name=f"{family}_synth_{idx:03d}",
                        family=family,
                        full_state=full_state.astype(np.float32),
                        basis=center.copy(),
                        offset=offset.astype(np.float32),
                        dominant_offset_dims=[int(x) for x in np.argsort(np.abs(offset))[-4:][::-1].tolist()],
                        source="synthetic",
                    )
                )

        return cls(
            concept_entries=concept_entries,
            family_supports=family_supports,
            family_centers=family_centers,
            family_rank_dims=family_rank_dims,
            global_recurrent_dims=global_recurrent_dims,
            dim=dim,
        )

    def entries_by_family(self) -> Dict[str, List[ConceptEntry]]:
        out: Dict[str, List[ConceptEntry]] = {family: [] for family in self.family_centers}
        for entry in self.concept_entries:
            out.setdefault(entry.family, []).append(entry)
        return out

    def region_view(self, entry: ConceptEntry, region: str) -> np.ndarray:
        support = self.family_supports[entry.family]
        out = np.zeros(self.dim, dtype=np.float32)
        if region == "object":
            dims = support.obj_dims + self.family_rank_dims[entry.family][:2]
        elif region == "memory":
            dims = support.mem_dims + self.global_recurrent_dims[:4]
        elif region == "identity":
            dims = support.id_dims + entry.dominant_offset_dims[:2]
        elif region == "readout":
            dims = support.disc_dims + self.global_recurrent_dims[:4]
        elif region == "macro":
            dims = sorted(set(self.global_recurrent_dims[:6] + self.family_rank_dims[entry.family][:4] + support.disc_dims))
        else:
            raise ValueError(f"unknown region: {region}")
        out[dims] = entry.full_state[dims]
        return out

    def region_matrix(self, region: str, entries: Iterable[ConceptEntry]) -> np.ndarray:
        return np.stack([self.region_view(entry, region) for entry in entries], axis=0)

    def fit_affine_operator(self, source_region: str, target_region: str, entries: List[ConceptEntry]) -> Dict[str, np.ndarray]:
        x = self.region_matrix(source_region, entries)
        y = self.region_matrix(target_region, entries)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        theta, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
        return {
            "weight": theta[:-1, :].astype(np.float32),
            "bias": theta[-1, :].astype(np.float32),
        }

    def apply_affine_operator(self, operator: Dict[str, np.ndarray], source_region: str, entries: List[ConceptEntry]) -> np.ndarray:
        x = self.region_matrix(source_region, entries)
        return x @ operator["weight"] + operator["bias"]

    def evaluate_affine_operator(self, operator: Dict[str, np.ndarray], source_region: str, target_region: str, entries: List[ConceptEntry]) -> Dict[str, float]:
        pred = self.apply_affine_operator(operator, source_region, entries)
        target = self.region_matrix(target_region, entries)
        mean_error = float(np.linalg.norm(pred - target, axis=1).mean())
        baseline = np.repeat(target.mean(axis=0, keepdims=True), target.shape[0], axis=0)
        baseline_error = float(np.linalg.norm(baseline - target, axis=1).mean())
        relative_gain = float(max(0.0, (baseline_error - mean_error) / max(baseline_error, 1e-8)))
        return {
            "mean_error": mean_error,
            "baseline_error": baseline_error,
            "relative_gain": relative_gain,
        }
