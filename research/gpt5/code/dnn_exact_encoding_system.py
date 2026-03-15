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
class DnnExactEncodingSystem:
    systematic: Dict[str, Any]
    restoration: Dict[str, Any]
    specific_bridge: Dict[str, Any]
    successor_restoration: Dict[str, Any]

    @classmethod
    def from_artifacts(cls, root: Path) -> "DnnExactEncodingSystem":
        temp = root / "tests" / "codex_temp"
        return cls(
            systematic=load_json(temp / "dnn_systematic_mass_extraction_block_20260315.json"),
            restoration=load_json(temp / "dnn_math_restoration_status_block_20260315.json"),
            specific_bridge=load_json(temp / "dnn_specific_math_bridge_block_20260315.json"),
            successor_restoration=load_json(temp / "dnn_successor_math_restoration_block_20260315.json"),
        )

    def summary(self) -> Dict[str, Any]:
        sys_head = self.systematic["headline_metrics"]
        sys_support = self.systematic["support_metrics"]
        restore = self.restoration["restoration_terms"]
        specific = self.specific_bridge["headline_metrics"]
        succ = self.successor_restoration["restoration_terms"]

        basis_offset_core_score = (
            0.35 * restore["family_basis_parametric_score"]
            + 0.35 * restore["concept_offset_parametric_score"]
            + 0.30 * specific["specific_parametric_restoration_score"]
        )
        contextual_protocol_score = (
            0.35 * restore["protocol_field_parametric_score"]
            + 0.25 * restore["topology_parametric_score"]
            + 0.20 * specific["contextual_specific_bridge_score"]
            + 0.20 * clamp01(sys_support["structured_specific_gain"] / 0.60)
        )
        successor_system_score = (
            0.55 * succ["successor_restoration_score"]
            + 0.25 * succ["successor_exactness_score"]
            + 0.20 * succ["successor_transport_score"]
        )
        evidence_exactness_score = (
            0.40 * sys_head["exact_real_fraction"]
            + 0.35 * (1.0 - specific["family_to_specific_gap"])
            + 0.25 * succ["successor_exactness_score"]
        )

        system_parametric_score = (
            0.34 * basis_offset_core_score
            + 0.28 * contextual_protocol_score
            + 0.22 * successor_system_score
            + 0.16 * evidence_exactness_score
        )
        exact_system_closure_score = (
            0.30 * evidence_exactness_score
            + 0.25 * (1.0 - specific["family_to_specific_gap"])
            + 0.25 * succ["successor_restoration_score"]
            + 0.20 * clamp01(sys_head["exact_real_fraction"] / 0.80)
        )

        theorem_candidate = {
            "equation": "h(c, ctx, stage) ~= B_f + Delta_c + C_ctx(c, ctx) + P_proto(c, ctx, stage) + T_succ(c, ctx, stage) + epsilon",
            "decomposition": {
                "basis": "B_f = family basis / family patch",
                "offset": "Delta_c = bounded concept-specific offset",
                "contextual": "C_ctx = contextual and relation-conditioned correction",
                "protocol": "P_proto = protocol / task / bridge correction",
                "successor": "T_succ = stage-conditioned successor transport and continuation term",
            },
            "meaning": "The exact encoding principle is no longer just basis plus offset. It is a system law where family basis and concept offset are refined by contextual, protocol, and successor operators.",
        }

        return {
            "basis_offset_core_score": float(basis_offset_core_score),
            "contextual_protocol_score": float(contextual_protocol_score),
            "successor_system_score": float(successor_system_score),
            "evidence_exactness_score": float(evidence_exactness_score),
            "system_parametric_score": float(system_parametric_score),
            "exact_system_closure_score": float(exact_system_closure_score),
            "theorem_candidate": theorem_candidate,
        }
