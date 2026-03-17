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
class DnnJointExactClosureBoard:
    exact_system: Dict[str, Any]
    specific_bridge: Dict[str, Any]
    successor_restoration: Dict[str, Any]
    system_crack_board: Dict[str, Any]
    systematic_extraction: Dict[str, Any]

    @classmethod
    def from_artifacts(cls, root: Path) -> "DnnJointExactClosureBoard":
        temp = root / "tests" / "codex_temp"
        return cls(
            exact_system=load_json(temp / "dnn_exact_encoding_system_block_20260315.json"),
            specific_bridge=load_json(temp / "dnn_specific_math_bridge_block_20260315.json"),
            successor_restoration=load_json(temp / "dnn_successor_math_restoration_block_20260315.json"),
            system_crack_board=load_json(temp / "dnn_system_crack_board_block_20260315.json"),
            systematic_extraction=load_json(temp / "dnn_systematic_mass_extraction_block_20260315.json"),
        )

    def summary(self) -> Dict[str, Any]:
        exact_metrics = self.exact_system["headline_metrics"]
        specific_metrics = self.specific_bridge["headline_metrics"]
        successor_terms = self.successor_restoration["restoration_terms"]
        crack_metrics = self.system_crack_board["headline_metrics"]
        extraction_metrics = self.systematic_extraction["headline_metrics"]

        dense_evidence_score = (
            0.55 * exact_metrics["evidence_exactness_score"]
            + 0.45 * extraction_metrics["exact_real_fraction"]
        )
        family_specific_closure_score = (
            0.55 * specific_metrics["exact_specific_closure_score"]
            + 0.25 * specific_metrics["real_specific_bridge_score"]
            + 0.20 * (1.0 - specific_metrics["family_to_specific_gap"])
        )
        successor_closure_score = (
            0.45 * successor_terms["successor_restoration_score"]
            + 0.30 * successor_terms["successor_exactness_score"]
            + 0.25 * successor_terms["successor_transport_score"]
        )
        coupled_exact_closure_score = (
            0.30 * dense_evidence_score
            + 0.35 * family_specific_closure_score
            + 0.35 * successor_closure_score
        )
        theorem_readiness_under_coupling = (
            0.40 * crack_metrics["theorem_candidate_strength"]
            + 0.35 * coupled_exact_closure_score
            + 0.25 * exact_metrics["exact_system_closure_score"]
        )

        bottleneck_table: List[Dict[str, Any]] = [
            {
                "block": "dense exact evidence",
                "score": round(dense_evidence_score, 4),
                "status": "weak_bottleneck" if dense_evidence_score < 0.45 else "partial",
                "why": "决定系统定理能否从 row/signature 证据推进到 dense neuron-level exact evidence。",
            },
            {
                "block": "family-to-specific exact closure",
                "score": round(family_specific_closure_score, 4),
                "status": "weak_bottleneck" if family_specific_closure_score < 0.50 else "partial",
                "why": "决定具体概念细节能否从 family-level 结构逼近 exact closure。",
            },
            {
                "block": "successor exact closure",
                "score": round(successor_closure_score, 4),
                "status": "weak_bottleneck" if successor_closure_score < 0.55 else "partial",
                "why": "决定系统编码能否真正进入连续后继与语言链路。",
            },
        ]

        metric_lines_cn = [
            f"（dense精确证据强度）dense_evidence_score = {dense_evidence_score:.4f}",
            f"（family到specific精确闭合）family_specific_closure_score = {family_specific_closure_score:.4f}",
            f"（successor精确闭合）successor_closure_score = {successor_closure_score:.4f}",
            f"（三瓶颈联合闭合度）coupled_exact_closure_score = {coupled_exact_closure_score:.4f}",
            f"（系统定理联合准备度）theorem_readiness_under_coupling = {theorem_readiness_under_coupling:.4f}",
        ]

        return {
            "dense_evidence_score": float(dense_evidence_score),
            "family_specific_closure_score": float(family_specific_closure_score),
            "successor_closure_score": float(successor_closure_score),
            "coupled_exact_closure_score": float(coupled_exact_closure_score),
            "theorem_readiness_under_coupling": float(theorem_readiness_under_coupling),
            "bottleneck_table": bottleneck_table,
            "metric_lines_cn": metric_lines_cn,
            "critical_path": [
                "先提高 dense neuron-level exact evidence，避免所有闭合继续被代理证据上限锁死",
                "同步推进 family-to-specific exact closure，不再让 specific 只能停留在 parametric bridge",
                "同步推进 successor exact closure，让系统结构真正进入连续后继和语言链路",
            ],
        }
