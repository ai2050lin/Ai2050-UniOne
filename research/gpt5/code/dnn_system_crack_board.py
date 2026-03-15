from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class DnnSystemCrackBoard:
    systematic: Dict[str, Any]
    signatures: Dict[str, Any]
    specific: Dict[str, Any]
    exact_system: Dict[str, Any]
    successor: Dict[str, Any]
    math_restoration: Dict[str, Any]
    progress_board: Dict[str, Any]

    @classmethod
    def from_artifacts(cls, root: Path) -> "DnnSystemCrackBoard":
        temp = root / "tests" / "codex_temp"
        return cls(
            systematic=load_json(temp / "dnn_systematic_mass_extraction_block_20260315.json"),
            signatures=load_json(temp / "dnn_activation_signature_mining_block_20260315.json"),
            specific=load_json(temp / "dnn_specific_math_bridge_block_20260315.json"),
            exact_system=load_json(temp / "dnn_exact_encoding_system_block_20260315.json"),
            successor=load_json(temp / "dnn_successor_math_restoration_block_20260315.json"),
            math_restoration=load_json(temp / "dnn_math_restoration_status_block_20260315.json"),
            progress_board=load_json(temp / "dnn_corpus_to_full_theory_progress_board_20260315.json"),
        )

    def summary(self) -> Dict[str, Any]:
        systematic_general = self.systematic["headline_metrics"]
        systematic_support = self.systematic["support_metrics"]
        signature_metrics = self.signatures["headline_metrics"]
        specific_metrics = self.specific["headline_metrics"]
        exact_metrics = self.exact_system["headline_metrics"]
        successor_metrics = self.successor["restoration_terms"]
        restoration_terms = self.math_restoration["restoration_terms"]
        progress_metrics = self.progress_board["headline_metrics"]

        extraction_base_score = (
            0.38 * systematic_general["exact_real_fraction"]
            + 0.16 * systematic_general["scale_coverage"]
            + 0.10 * systematic_general["family_coverage"]
            + 0.16 * clamp01(signature_metrics["activation_signature_mining_score"])
            + 0.20 * clamp01(signature_metrics["unique_concepts"] / 180.0)
        )

        parametric_system_score = (
            0.26 * exact_metrics["basis_offset_core_score"]
            + 0.20 * exact_metrics["contextual_protocol_score"]
            + 0.20 * specific_metrics["specific_parametric_restoration_score"]
            + 0.18 * restoration_terms["protocol_field_parametric_score"]
            + 0.16 * restoration_terms["topology_parametric_score"]
        )

        specific_exactness_score = (
            0.50 * specific_metrics["exact_specific_closure_score"]
            + 0.25 * (1.0 - specific_metrics["family_to_specific_gap"])
            + 0.25 * specific_metrics["real_specific_bridge_score"]
        )

        successor_exactness_score = (
            0.35 * successor_metrics["successor_structure_score"]
            + 0.35 * successor_metrics["successor_exactness_score"]
            + 0.30 * successor_metrics["successor_restoration_score"]
        )

        theorem_candidate_strength = (
            0.45 * exact_metrics["system_parametric_score"]
            + 0.30 * restoration_terms["full_restoration_score"]
            + 0.25 * clamp01(progress_metrics["full_math_theory_percent"] / 100.0)
        )

        exact_theorem_closure_score = (
            0.30 * exact_metrics["exact_system_closure_score"]
            + 0.30 * specific_exactness_score
            + 0.25 * successor_exactness_score
            + 0.15 * exact_metrics["evidence_exactness_score"]
        )

        blockers = [
            {
                "name": "family-to-specific 精确闭合不足",
                "severity": round(float(specific_metrics["family_to_specific_gap"]), 4),
                "why": "specific 参数桥已经很强，但 family 到 exact specific 的真实闭合仍没有打穿。",
            },
            {
                "name": "successor 精确闭合不足",
                "severity": round(float(1.0 - successor_metrics["successor_restoration_score"]), 4),
                "why": "successor 仍然是系统侧最弱项，直接拖低 exact theorem closure。",
            },
            {
                "name": "dense exact evidence 不足",
                "severity": round(float(1.0 - exact_metrics["evidence_exactness_score"]), 4),
                "why": "row/signature 证据已很强，但 dense neuron-level exact tensors 仍然偏薄。",
            },
            {
                "name": "神经元级普遍结构不足",
                "severity": round(float(1.0 - progress_metrics["neuron_level_general_structure_percent"] / 100.0), 4),
                "why": "系统理论已出现，但 neuron-level 的一般结构还没有形成最终统一规律。",
            },
        ]
        blockers = sorted(blockers, key=lambda item: item["severity"], reverse=True)

        metric_lines_cn = [
            f"（DNN结构提取底座）extraction_base_score = {extraction_base_score:.4f}",
            f"（DNN系统参数原理）parametric_system_score = {parametric_system_score:.4f}",
            f"（concept-specific精确闭合）specific_exactness_score = {specific_exactness_score:.4f}",
            f"（successor精确闭合）successor_exactness_score = {successor_exactness_score:.4f}",
            f"（系统候选定理强度）theorem_candidate_strength = {theorem_candidate_strength:.4f}",
            f"（系统终局闭合度）exact_theorem_closure_score = {exact_theorem_closure_score:.4f}",
        ]

        return {
            "extraction_base_score": float(extraction_base_score),
            "parametric_system_score": float(parametric_system_score),
            "specific_exactness_score": float(specific_exactness_score),
            "successor_exactness_score": float(successor_exactness_score),
            "theorem_candidate_strength": float(theorem_candidate_strength),
            "exact_theorem_closure_score": float(exact_theorem_closure_score),
            "top_blockers": blockers,
            "metric_lines_cn": metric_lines_cn,
            "critical_path": [
                "继续扩大 dense neuron-level exact evidence，减少 row/signature 代理依赖",
                "把 family-to-specific exact closure 与 successor exact closure 作为同一系统闭合目标联合推进",
                "在更强 exact 证据基础上重算系统候选定理，检查是否逼近 final theorem closure",
            ],
        }
