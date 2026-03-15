from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class DnnSpecificMathBridge:
    dense_real: Dict[str, Any]
    signatures: Dict[str, Any]
    multimodel_specific: Dict[str, Any]
    real_heldout: Dict[str, Any]
    restoration: Dict[str, Any]

    @classmethod
    def from_artifacts(cls, root: Path) -> "DnnSpecificMathBridge":
        temp = root / "tests" / "codex_temp"
        return cls(
            dense_real=load_json(temp / "dnn_dense_real_unit_corpus_block_20260315.json"),
            signatures=load_json(temp / "dnn_activation_signature_mining_block_20260315.json"),
            multimodel_specific=load_json(temp / "dnn_multimodel_specific_reconstruction_block_20260315.json"),
            real_heldout=load_json(temp / "dnn_real_heldout_region_reconstruction_block_20260315.json"),
            restoration=load_json(temp / "dnn_math_restoration_status_block_20260315.json"),
        )

    def summary(self) -> Dict[str, Any]:
        dense_metrics = self.dense_real["headline_metrics"]
        sig_metrics = self.signatures["headline_metrics"]
        multi_metrics = self.multimodel_specific["headline_metrics"]
        real_metrics = self.real_heldout["headline_metrics"]
        real_pairs = self.real_heldout["pair_results"]
        restoration_terms = self.restoration["restoration_terms"]

        specific_real_support_score = min(
            1.0,
            0.55 * clamp01(dense_metrics["specific_weight"] / 560.0)
            + 0.45 * clamp01(dense_metrics["unit_count"] / 400.0),
        )
        specific_signature_score = min(
            1.0,
            0.40 * clamp01(sig_metrics["specific_signature_rows"] / 194.0)
            + 0.30 * clamp01(sig_metrics["unique_concepts"] / 158.0)
            + 0.30 * clamp01(sig_metrics["mean_specific_layer_count"] / 11.44),
        )
        contextual_specific_bridge_score = min(
            1.0,
            0.60 * clamp01(multi_metrics["contextual_family_to_specific_gain"] / 0.70)
            + 0.40 * clamp01(multi_metrics["contextual_recovery_gain"] / 0.25),
        )
        real_specific_bridge_score = min(
            1.0,
            0.65 * clamp01(real_metrics["heldout_gain_mean"] / 0.75)
            + 0.35 * clamp01(real_metrics["positive_pair_count"] / 3.0),
        )
        family_to_specific_gap = clamp01(
            1.0 - real_pairs["family_to_specific"]["test_relative_gain"]
        )

        specific_parametric_restoration_score = (
            0.25 * specific_real_support_score
            + 0.25 * specific_signature_score
            + 0.25 * contextual_specific_bridge_score
            + 0.15 * real_specific_bridge_score
            + 0.10 * restoration_terms["concept_offset_parametric_score"]
        )
        exact_specific_closure_score = (
            0.45 * clamp01(real_pairs["family_to_specific"]["test_relative_gain"] / 0.85)
            + 0.30 * clamp01(multi_metrics["contextual_family_to_specific_gain"] / 0.85)
            + 0.25 * clamp01(restoration_terms["concept_offset_parametric_score"] / 0.99)
        )

        candidate_law = {
            "equation": "h_specific(c, ctx) ~= B_f + Delta_c + C_ctx(c, ctx) + P_proto(c, ctx) + epsilon_specific",
            "offset_law": "Delta_c ~= Delta_family_local(c) + Delta_contextual(c, ctx) + Delta_protocol(c, ctx) + xi_c",
            "meaning": "Specific concept structure is family basis plus bounded offset, then refined by contextual and protocol corrections.",
        }

        return {
            "specific_real_support_score": float(specific_real_support_score),
            "specific_signature_score": float(specific_signature_score),
            "contextual_specific_bridge_score": float(contextual_specific_bridge_score),
            "real_specific_bridge_score": float(real_specific_bridge_score),
            "family_to_specific_gap": float(family_to_specific_gap),
            "specific_parametric_restoration_score": float(specific_parametric_restoration_score),
            "exact_specific_closure_score": float(exact_specific_closure_score),
            "candidate_law": candidate_law,
        }
