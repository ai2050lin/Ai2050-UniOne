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
class AgiBreakthroughPreparationBoard:
    dnn_systematic: Dict[str, Any]
    dnn_specific: Dict[str, Any]
    dnn_exact_system: Dict[str, Any]
    dnn_successor: Dict[str, Any]
    spike_route: Dict[str, Any]
    spike_scaling: Dict[str, Any]
    spike_inventory: Dict[str, Any]
    spike_homeostasis: Dict[str, Any]
    spike_successor: Dict[str, Any]

    @classmethod
    def from_artifacts(cls, root: Path) -> "AgiBreakthroughPreparationBoard":
        temp = root / "tests" / "codex_temp"
        return cls(
            dnn_systematic=load_json(temp / "dnn_systematic_mass_extraction_block_20260315.json"),
            dnn_specific=load_json(temp / "dnn_specific_math_bridge_block_20260315.json"),
            dnn_exact_system=load_json(temp / "dnn_exact_encoding_system_block_20260315.json"),
            dnn_successor=load_json(temp / "dnn_successor_math_restoration_block_20260315.json"),
            spike_route=load_json(temp / "spike_icspb_non_attention_non_bp_language_architecture_route_20260315.json"),
            spike_scaling=load_json(temp / "spike_icspb_3d_scaling_readiness_block_20260315.json"),
            spike_inventory=load_json(temp / "spike_icspb_3d_feature_inventory_measurement_block_20260315.json"),
            spike_homeostasis=load_json(temp / "spike_icspb_3d_homeostatic_control_law_block_20260315.json"),
            spike_successor=load_json(temp / "spike_icspb_3d_successor_quality_audit_block_20260315.json"),
        )

    def summary(self) -> Dict[str, Any]:
        dnn_sys = self.dnn_systematic["headline_metrics"]
        dnn_sys_support = self.dnn_systematic["support_metrics"]
        dnn_specific = self.dnn_specific["headline_metrics"]
        dnn_exact = self.dnn_exact_system["headline_metrics"]
        dnn_succ = self.dnn_successor["restoration_terms"]

        spike_route = self.spike_route["progress_estimate"]
        spike_route_support = self.spike_route["why_non_attention_non_bp_is_plausible"]["supporting_scores"]
        spike_scaling = self.spike_scaling["headline_metrics"]
        spike_inventory = self.spike_inventory["headline_metrics"]
        spike_homeo = self.spike_homeostasis["headline_metrics"]
        spike_successor = self.spike_successor["headline_metrics"]

        dnn_foundation_score = (
            0.40 * dnn_sys["exact_real_fraction"]
            + 0.30 * clamp01(dnn_sys_support["dense_real_specific_weight"] / 600.0)
            + 0.30 * clamp01(dnn_sys_support["dense_real_macro_weight"] / 650.0)
        )
        dnn_parametric_score = (
            0.35 * dnn_exact["basis_offset_core_score"]
            + 0.35 * dnn_exact["contextual_protocol_score"]
            + 0.30 * dnn_specific["specific_parametric_restoration_score"]
        )
        dnn_exactness_score = (
            0.40 * dnn_exact["evidence_exactness_score"]
            + 0.30 * (1.0 - dnn_specific["family_to_specific_gap"])
            + 0.30 * dnn_succ["successor_exactness_score"]
        )
        spike_architecture_score = (
            0.30 * spike_route_support["architecture_route_score"]
            + 0.25 * spike_scaling["scaling_readiness_score"]
            + 0.20 * spike_inventory["inventory_grounded_score"]
            + 0.25 * spike_homeo["homeostatic_closure_score"]
        )
        spike_language_score = (
            0.55 * spike_successor["successor_quality_score"]
            + 0.25 * clamp01(1.0 - spike_successor["normalized_entropy"])
            + 0.20 * clamp01(max(0.0, spike_successor["next_token_margin"] + 0.02) / 0.08)
        )
        final_breakthrough_readiness = (
            0.22 * dnn_foundation_score
            + 0.22 * dnn_parametric_score
            + 0.22 * dnn_exactness_score
            + 0.18 * spike_architecture_score
            + 0.16 * spike_language_score
        )

        puzzle_table: List[Dict[str, Any]] = [
            {
                "puzzle": "DNN 结构提取底座",
                "score": round(dnn_foundation_score, 4),
                "status": "strong_candidate" if dnn_foundation_score >= 0.65 else "partial",
                "meaning": "决定我们是否真的有足够厚的真实结构证据来谈编码原理。",
            },
            {
                "puzzle": "DNN 参数原理",
                "score": round(dnn_parametric_score, 4),
                "status": "strong_candidate" if dnn_parametric_score >= 0.80 else "partial",
                "meaning": "决定 basis / offset / contextual / protocol 这几条数学线是否已经成形。",
            },
            {
                "puzzle": "DNN 精确闭合",
                "score": round(dnn_exactness_score, 4),
                "status": "weak_bottleneck" if dnn_exactness_score < 0.45 else "partial",
                "meaning": "决定系统候选定理能不能从参数理解推进到 exact closure。",
            },
            {
                "puzzle": "Spike 可规模化架构",
                "score": round(spike_architecture_score, 4),
                "status": "strong_candidate" if spike_architecture_score >= 0.70 else "partial",
                "meaning": "决定非 Attention + 非 BP 路线是否真的具备放大潜力。",
            },
            {
                "puzzle": "Spike 语言与后继",
                "score": round(spike_language_score, 4),
                "status": "weak_bottleneck" if spike_language_score < 0.35 else "partial",
                "meaning": "决定 SpikeICSPB 能否把稳定结构转成真实语言连续体。",
            },
        ]

        blockers = [
            {
                "name": "dense exact evidence 不足",
                "severity": round(1.0 - dnn_exact["evidence_exactness_score"], 4),
                "why": "row/signature 级证据已经强，但 dense neuron-level exact tensors 仍不足。",
            },
            {
                "name": "family-to-specific exact closure 不足",
                "severity": round(dnn_specific["family_to_specific_gap"], 4),
                "why": "specific 数学桥已强，但真实 held-out 的 exact family->specific 闭合仍未完成。",
            },
            {
                "name": "successor exact closure 不足",
                "severity": round(1.0 - dnn_succ["successor_restoration_score"], 4),
                "why": "DNN 侧与 Spike 侧都显示 successor 仍然是当前最弱系统项。",
            },
            {
                "name": "Spike 语言连续体过弱",
                "severity": round(1.0 - spike_successor["successor_quality_score"], 4),
                "why": "非 Attention + 非 BP 路线已能设计和扩展，但还不能给出强语言后继。",
            },
        ]
        blockers = sorted(blockers, key=lambda x: x["severity"], reverse=True)

        critical_path = [
            "提高 dense neuron-level exact evidence",
            "同时打 family-to-specific exact closure 与 successor exact closure",
            "把强化后的系统定理迁移到 scalable SpikeICSPB Phase-A 候选上",
            "在 stronger dense export 与 unseen-family 条件下重测 theorem candidate",
        ]

        return {
            "dnn_foundation_score": float(dnn_foundation_score),
            "dnn_parametric_score": float(dnn_parametric_score),
            "dnn_exactness_score": float(dnn_exactness_score),
            "spike_architecture_score": float(spike_architecture_score),
            "spike_language_score": float(spike_language_score),
            "final_breakthrough_readiness": float(final_breakthrough_readiness),
            "puzzle_table": puzzle_table,
            "top_blockers": blockers,
            "critical_path": critical_path,
        }
