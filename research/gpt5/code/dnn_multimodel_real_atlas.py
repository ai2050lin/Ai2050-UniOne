from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from research.gpt5.code.dnn_real_codebook_atlas import RealCodebookAtlas, load_json as load_codebook_json


@dataclass
class MultiModelRealEntry:
    name: str
    family: str
    source: str
    views: Dict[str, np.ndarray]


class MultiModelRealAtlas:
    def __init__(self, entries: List[MultiModelRealEntry]):
        self.entries = self._normalize_entries(entries)

    @staticmethod
    def _normalize_entries(entries: List[MultiModelRealEntry]) -> List[MultiModelRealEntry]:
        if not entries:
            return entries
        view_dims: Dict[str, int] = {}
        for entry in entries:
            for view_name, arr in entry.views.items():
                view_dims[view_name] = max(view_dims.get(view_name, 0), int(arr.shape[0]))

        normalized: List[MultiModelRealEntry] = []
        for entry in entries:
            views: Dict[str, np.ndarray] = {}
            for view_name, target_dim in view_dims.items():
                arr = entry.views.get(view_name)
                if arr is None:
                    views[view_name] = np.zeros(target_dim, dtype=np.float32)
                    continue
                if arr.shape[0] == target_dim:
                    views[view_name] = arr.astype(np.float32, copy=False)
                    continue
                padded = np.zeros(target_dim, dtype=np.float32)
                padded[: arr.shape[0]] = arr.astype(np.float32, copy=False)
                views[view_name] = padded
            normalized.append(
                MultiModelRealEntry(
                    name=entry.name,
                    family=entry.family,
                    source=entry.source,
                    views=views,
                )
            )
        return normalized

    @classmethod
    def from_artifacts(cls, root: Path) -> "MultiModelRealAtlas":
        entries: List[MultiModelRealEntry] = []

        codebook = RealCodebookAtlas.from_codebook(root)
        for entry in codebook.entries:
            specific = codebook.region_view(entry, "specific")
            family = codebook.region_view(entry, "family")
            shared = codebook.region_view(entry, "shared")
            early = codebook.region_view(entry, "early")
            mid = codebook.region_view(entry, "mid")
            late = codebook.region_view(entry, "late")
            stage = np.array(
                [
                    early[0],
                    early[1],
                    mid[0],
                    mid[1],
                    late[0],
                    late[1],
                    entry.family_margin,
                    entry.subspace_margin,
                ],
                dtype=np.float32,
            )
            macro = np.array(
                [
                    early[2],
                    mid[2],
                    late[2],
                    float(specific[:3].sum()),
                    float(family[:3].sum()),
                    entry.family_margin,
                    entry.subspace_margin,
                    float(shared[:3].sum()),
                ],
                dtype=np.float32,
            )
            contextual_family = np.concatenate([family, stage, macro], axis=0).astype(np.float32)
            entries.append(
                MultiModelRealEntry(
                    name=entry.name,
                    family=entry.family,
                    source="codebook",
                    views={
                        "specific": specific.astype(np.float32),
                        "family": family.astype(np.float32),
                        "shared": shared.astype(np.float32),
                        "stage": stage,
                        "macro": macro,
                        "contextual_family": contextual_family,
                    },
                )
            )

        refresh_specs = [
            (
                "qwen3_4b",
                root / "tests" / "codex_temp" / "qwen3_deepseek7b_concept_encoding_decomposition_qwen_refresh_20260315.json",
            ),
            (
                "deepseek_7b",
                root / "tests" / "codex_temp" / "qwen3_deepseek7b_concept_encoding_decomposition_deepseek_refresh_20260315.json",
            ),
        ]
        structure = load_codebook_json(root / "tests" / "codex_temp" / "qwen3_deepseek7b_real_model_structure_atlas_20260310.json")
        mechanism = load_codebook_json(root / "tests" / "codex_temp" / "qwen3_deepseek7b_mechanism_bridge_20260309.json")

        for model_key, refresh_path in refresh_specs:
            payload = load_codebook_json(refresh_path)["models"][model_key]
            layer_atlas = structure["models"][model_key]["layer_atlas"]
            counts = {
                "concept": 0.0,
                "relation": 0.0,
                "balanced": 0.0,
                "shared": 0.0,
                "targeted": 0.0,
                "control": 0.0,
            }
            for row in layer_atlas:
                stage = row["support_stage"]
                if stage == "concept_biased":
                    counts["concept"] += 1.0
                elif stage == "relation_biased":
                    counts["relation"] += 1.0
                else:
                    counts["balanced"] += 1.0
                if row["is_shared_band"]:
                    counts["shared"] += 1.0
                if row["is_targeted_band"]:
                    counts["targeted"] += 1.0
                if row["is_control_band"]:
                    counts["control"] += 1.0
            total_layers = max(1.0, float(len(layer_atlas)))
            stage_base = np.array(
                [
                    counts["concept"] / total_layers,
                    counts["relation"] / total_layers,
                    counts["balanced"] / total_layers,
                    counts["shared"] / total_layers,
                    counts["targeted"] / total_layers,
                    counts["control"] / total_layers,
                ],
                dtype=np.float32,
            )
            comps = mechanism["models"][model_key]["components"]
            bridge_score = float(mechanism["models"][model_key]["mechanism_bridge_score"])

            for target_name, target_row in payload["targets"].items():
                best = target_row["best_layer"]
                layer_norm = float(best["layer"]) / max(1.0, total_layers - 1.0)
                family_fit = float(1.0 - best["true_residual_ratio"])
                specific = np.array(
                    [
                        family_fit,
                        float(best["margin_vs_best_wrong"]),
                        float(best["offset_top32_energy_ratio"]),
                        float(best["shared_norm_ratio"]),
                        float(best["best_wrong_residual_ratio"]),
                        layer_norm,
                        bridge_score,
                        float(comps["protocol_calling"]),
                    ],
                    dtype=np.float32,
                )
                family = np.array(
                    [
                        float(comps["shared_basis"]),
                        float(comps["offset"]),
                        float(comps["H_representation"]),
                        float(comps["G_gating"]),
                        family_fit,
                        float(best["shared_norm_ratio"]),
                        layer_norm,
                        bridge_score,
                    ],
                    dtype=np.float32,
                )
                shared = np.array(
                    [
                        float(comps["shared_basis"]),
                        float(comps["R_relation"]),
                        float(comps["T_topology"]),
                        float(comps["protocol_calling"]),
                        float(comps["evidence_directness"]),
                        bridge_score,
                        stage_base[3],
                        stage_base[4],
                    ],
                    dtype=np.float32,
                )
                stage = np.array(
                    [
                        stage_base[0],
                        stage_base[1],
                        stage_base[2],
                        stage_base[3],
                        stage_base[4],
                        stage_base[5],
                        layer_norm,
                        float(total_layers / 36.0),
                    ],
                    dtype=np.float32,
                )
                macro = np.array(
                    [
                        float(comps["protocol_calling"]),
                        float(comps["R_relation"]),
                        float(comps["T_topology"]),
                        float(comps["G_gating"]),
                        float(comps["H_representation"]),
                        float(comps["evidence_directness"]),
                        bridge_score,
                        family_fit,
                    ],
                    dtype=np.float32,
                )
                contextual_family = np.concatenate([family, stage, macro], axis=0).astype(np.float32)
                entries.append(
                    MultiModelRealEntry(
                        name=f"{model_key}:{target_name}",
                        family=str(target_row["true_family"]),
                        source=model_key,
                        views={
                            "specific": specific,
                            "family": family,
                            "shared": shared,
                            "stage": stage,
                            "macro": macro,
                            "contextual_family": contextual_family,
                        },
                    )
                )

        return cls(entries=entries)

    def entries_by_family(self) -> Dict[str, List[MultiModelRealEntry]]:
        out: Dict[str, List[MultiModelRealEntry]] = {}
        for entry in self.entries:
            out.setdefault(entry.family, []).append(entry)
        return out

    def view_matrix(self, view: str, entries: List[MultiModelRealEntry]) -> np.ndarray:
        return np.stack([entry.views[view] for entry in entries], axis=0)

    def stacked_views(self, view_names: Sequence[str], entries: List[MultiModelRealEntry]) -> np.ndarray:
        blocks = [self.view_matrix(view_name, entries) for view_name in view_names]
        return np.concatenate(blocks, axis=1)

    def fit_affine_operator(self, source_view: str, target_view: str, entries: List[MultiModelRealEntry]) -> Dict[str, np.ndarray]:
        x = self.view_matrix(source_view, entries)
        y = self.view_matrix(target_view, entries)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        theta, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
        return {"weight": theta[:-1, :].astype(np.float32), "bias": theta[-1, :].astype(np.float32)}

    def evaluate_affine_operator(self, operator: Dict[str, np.ndarray], source_view: str, target_view: str, entries: List[MultiModelRealEntry]) -> Dict[str, float]:
        x = self.view_matrix(source_view, entries)
        y = self.view_matrix(target_view, entries)
        pred = x @ operator["weight"] + operator["bias"]
        mean_error = float(np.linalg.norm(pred - y, axis=1).mean())
        baseline = np.repeat(y.mean(axis=0, keepdims=True), y.shape[0], axis=0)
        baseline_error = float(np.linalg.norm(baseline - y, axis=1).mean())
        relative_gain = float(max(0.0, (baseline_error - mean_error) / max(baseline_error, 1e-8)))
        return {"mean_error": mean_error, "baseline_error": baseline_error, "relative_gain": relative_gain}

    def fit_structured_canonical_operator(
        self,
        source_views: Sequence[str],
        target_view: str,
        entries: List[MultiModelRealEntry],
        family_condition_view: str = "family",
    ) -> Dict[str, object]:
        x = self.stacked_views(source_views, entries)
        y = self.view_matrix(target_view, entries)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        theta, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
        base_weight = theta[:-1, :].astype(np.float32)
        base_bias = theta[-1, :].astype(np.float32)

        pred = x @ base_weight + base_bias
        family_inputs = self.view_matrix(family_condition_view, entries)
        family_residuals: Dict[str, np.ndarray] = {}
        family_gates: Dict[str, np.ndarray] = {}
        family_to_indices: Dict[str, List[int]] = {}
        for idx, entry in enumerate(entries):
            family_to_indices.setdefault(entry.family, []).append(idx)
        for family_name, idx_list in family_to_indices.items():
            if not idx_list:
                continue
            residual = y[idx_list] - pred[idx_list]
            family_residuals[family_name] = residual.mean(axis=0).astype(np.float32)
            family_gates[family_name] = family_inputs[idx_list].mean(axis=0).astype(np.float32)

        return {
            "source_views": list(source_views),
            "target_view": target_view,
            "family_condition_view": family_condition_view,
            "base_weight": base_weight,
            "base_bias": base_bias,
            "family_residuals": family_residuals,
            "family_gates": family_gates,
        }

    def evaluate_structured_canonical_operator(
        self,
        operator: Dict[str, object],
        entries: List[MultiModelRealEntry],
    ) -> Dict[str, float]:
        source_views = list(operator["source_views"])
        target_view = str(operator["target_view"])
        family_condition_view = str(operator["family_condition_view"])
        x = self.stacked_views(source_views, entries)
        y = self.view_matrix(target_view, entries)
        pred = x @ operator["base_weight"] + operator["base_bias"]
        family_inputs = self.view_matrix(family_condition_view, entries)

        adjusted_rows = []
        for idx, entry in enumerate(entries):
            row = pred[idx].copy()
            family_residual = operator["family_residuals"].get(entry.family)
            family_gate = operator["family_gates"].get(entry.family)
            if family_residual is not None and family_gate is not None:
                current_gate = family_inputs[idx]
                denom = float(np.linalg.norm(family_gate) * np.linalg.norm(current_gate))
                gate_scale = 0.0 if denom < 1e-8 else float(np.dot(current_gate, family_gate) / denom)
                row = row + max(0.0, gate_scale) * family_residual
            adjusted_rows.append(row)
        pred = np.stack(adjusted_rows, axis=0)

        mean_error = float(np.linalg.norm(pred - y, axis=1).mean())
        baseline = np.repeat(y.mean(axis=0, keepdims=True), y.shape[0], axis=0)
        baseline_error = float(np.linalg.norm(baseline - y, axis=1).mean())
        relative_gain = float(max(0.0, (baseline_error - mean_error) / max(baseline_error, 1e-8)))
        return {"mean_error": mean_error, "baseline_error": baseline_error, "relative_gain": relative_gain}
