from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8-sig")
    return __import__("json").loads(text)


@dataclass
class RealCodeEntry:
    name: str
    family: str
    specific_dims: List[int]
    family_dims: List[int]
    shared_dims: List[int]
    concept_hist: np.ndarray
    family_hist: np.ndarray
    shared_hist: np.ndarray
    family_margin: float
    subspace_margin: float


class RealCodebookAtlas:
    def __init__(self, entries: List[RealCodeEntry], n_layers: int):
        self.entries = entries
        self.n_layers = n_layers

    @classmethod
    def from_codebook(cls, root: Path) -> "RealCodebookAtlas":
        payload = load_json(root / "tests" / "codex_temp" / "concept_family_unified_codebook_20260308.json")
        n_layers = int(payload["meta"]["n_layers"])
        d_ff = int(payload["meta"]["d_ff"])
        entries: List[RealCodeEntry] = []
        for family, row in payload["family_stats"].items():
            family_dims = [int(x) for x in row["prototype_top_dims"]]
            shared_dims = [int(x) for x in row["robust_shared_dims"]]
            family_hist = cls._dims_to_layer_hist(family_dims, n_layers, d_ff)
            shared_hist = cls._dims_to_layer_hist(shared_dims, n_layers, d_ff)
            family_margin = float(row["family_cosine_margin"])
            subspace_margin = float(row["subspace_margin"])
            for concept, c_row in row["concept_specific"].items():
                specific_dims = [int(x) for x in c_row["top_specific_dims"]]
                concept_hist = cls._dims_to_layer_hist(specific_dims, n_layers, d_ff)
                entries.append(
                    RealCodeEntry(
                        name=concept,
                        family=family,
                        specific_dims=specific_dims,
                        family_dims=family_dims,
                        shared_dims=shared_dims,
                        concept_hist=concept_hist,
                        family_hist=family_hist,
                        shared_hist=shared_hist,
                        family_margin=family_margin,
                        subspace_margin=subspace_margin,
                    )
                )
        return cls(entries=entries, n_layers=n_layers)

    @staticmethod
    def _dims_to_layer_hist(dims: List[int], n_layers: int, d_ff: int) -> np.ndarray:
        hist = np.zeros(n_layers, dtype=np.float32)
        if not dims:
            return hist
        for dim in dims:
            layer = max(0, min(n_layers - 1, int(dim) // d_ff))
            hist[layer] += 1.0
        hist /= max(1.0, hist.sum())
        return hist

    def entries_by_family(self) -> Dict[str, List[RealCodeEntry]]:
        out: Dict[str, List[RealCodeEntry]] = {}
        for entry in self.entries:
            out.setdefault(entry.family, []).append(entry)
        return out

    def region_view(self, entry: RealCodeEntry, region: str) -> np.ndarray:
        third = max(1, self.n_layers // 3)
        early = slice(0, third)
        mid = slice(third, min(self.n_layers, 2 * third))
        late = slice(min(self.n_layers, 2 * third), self.n_layers)
        if region == "specific":
            return np.concatenate(
                [
                    entry.concept_hist[early],
                    entry.concept_hist[mid],
                    entry.concept_hist[late],
                    np.array([entry.family_margin, entry.subspace_margin], dtype=np.float32),
                ],
                axis=0,
            )
        if region == "family":
            return np.concatenate(
                [
                    entry.family_hist[early],
                    entry.family_hist[mid],
                    entry.family_hist[late],
                    np.array([entry.family_margin, entry.subspace_margin], dtype=np.float32),
                ],
                axis=0,
            )
        if region == "shared":
            return np.concatenate(
                [
                    entry.shared_hist[early],
                    entry.shared_hist[mid],
                    entry.shared_hist[late],
                    np.array([entry.family_margin, entry.subspace_margin], dtype=np.float32),
                ],
                axis=0,
            )
        if region == "early":
            return np.array(
                [
                    float(entry.concept_hist[early].sum()),
                    float(entry.family_hist[early].sum()),
                    float(entry.shared_hist[early].sum()),
                    float(entry.concept_hist[mid].sum()),
                    float(entry.family_hist[mid].sum()),
                    float(entry.shared_hist[mid].sum()),
                    entry.family_margin,
                    entry.subspace_margin,
                ],
                dtype=np.float32,
            )
        if region == "mid":
            return np.array(
                [
                    float(entry.concept_hist[mid].sum()),
                    float(entry.family_hist[mid].sum()),
                    float(entry.shared_hist[mid].sum()),
                    float(entry.concept_hist[late].sum()),
                    float(entry.family_hist[late].sum()),
                    float(entry.shared_hist[late].sum()),
                    entry.family_margin,
                    entry.subspace_margin,
                ],
                dtype=np.float32,
            )
        if region == "late":
            return np.array(
                [
                    float(entry.concept_hist[late].sum()),
                    float(entry.family_hist[late].sum()),
                    float(entry.shared_hist[late].sum()),
                    float(entry.concept_hist[mid].sum()),
                    float(entry.family_hist[mid].sum()),
                    float(entry.shared_hist[mid].sum()),
                    entry.family_margin,
                    entry.subspace_margin,
                ],
                dtype=np.float32,
            )
        raise ValueError(f"unknown region: {region}")

    def region_matrix(self, region: str, entries: List[RealCodeEntry]) -> np.ndarray:
        return np.stack([self.region_view(entry, region) for entry in entries], axis=0)

    def fit_affine_operator(self, source_region: str, target_region: str, entries: List[RealCodeEntry]) -> Dict[str, np.ndarray]:
        x = self.region_matrix(source_region, entries)
        y = self.region_matrix(target_region, entries)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        theta, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
        return {"weight": theta[:-1, :].astype(np.float32), "bias": theta[-1, :].astype(np.float32)}

    def evaluate_affine_operator(self, operator: Dict[str, np.ndarray], source_region: str, target_region: str, entries: List[RealCodeEntry]) -> Dict[str, float]:
        x = self.region_matrix(source_region, entries)
        y = self.region_matrix(target_region, entries)
        pred = x @ operator["weight"] + operator["bias"]
        mean_error = float(np.linalg.norm(pred - y, axis=1).mean())
        baseline = np.repeat(y.mean(axis=0, keepdims=True), y.shape[0], axis=0)
        baseline_error = float(np.linalg.norm(baseline - y, axis=1).mean())
        relative_gain = float(max(0.0, (baseline_error - mean_error) / max(baseline_error, 1e-8)))
        return {"mean_error": mean_error, "baseline_error": baseline_error, "relative_gain": relative_gain}
