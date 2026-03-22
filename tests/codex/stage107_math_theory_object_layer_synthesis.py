from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage107_math_theory_object_layer_synthesis_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage100_backfeed_suppression_hardening import build_backfeed_suppression_hardening_summary
from stage101_brain_evidence_joint_closure import build_brain_evidence_joint_closure_summary
from stage102_real_world_falsification_bridge import build_real_world_falsification_bridge_summary
from stage103_native_brain_anchor_search import build_native_brain_anchor_search_summary
from stage104_tensor_level_language_projection_rebuild import build_tensor_level_language_projection_rebuild_summary
from stage105_tensor_level_route_scale_rebuild import build_tensor_level_route_scale_rebuild_summary
from stage106_forward_backward_trace_rebuild import build_forward_backward_trace_rebuild_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def build_math_theory_object_layer_synthesis_summary() -> dict:
    projection = build_tensor_level_language_projection_rebuild_summary()["headline_metrics"]
    route = build_tensor_level_route_scale_rebuild_summary()["headline_metrics"]
    fb = build_forward_backward_trace_rebuild_summary()["headline_metrics"]
    backfeed = build_backfeed_suppression_hardening_summary()["headline_metrics"]
    joint = build_brain_evidence_joint_closure_summary()["headline_metrics"]
    bridge = build_real_world_falsification_bridge_summary()["headline_metrics"]
    anchors = build_native_brain_anchor_search_summary()["headline_metrics"]

    object_records = [
        {
            "name": "conditional_projection_field",
            "meaning": "语言在上下文条件下的投影场",
            "support": _clip01(
                0.34 * projection["reconstructed_context_gate_coherence"]
                + 0.28 * projection["reconstructed_bias_transport"]
                + 0.22 * projection["cross_dimension_projection_stability"]
                + 0.16 * projection["cross_dimension_separation"]
            ),
            "basis": [
                "reconstructed_context_gate_coherence",
                "reconstructed_bias_transport",
                "cross_dimension_projection_stability",
            ],
        },
        {
            "name": "distributed_route_fiber",
            "meaning": "分布式路由与纤维复用对象",
            "support": _clip01(
                0.34 * route["distributed_network_support"]
                + 0.28 * route["route_structure_coupling_strength"]
                + 0.18 * route["degradation_tolerance"]
                + 0.20 * (0.5 + 0.5 * _clip01(route["route_scale_margin"] + 0.20))
            ),
            "basis": [
                "distributed_network_support",
                "route_structure_coupling_strength",
                "degradation_tolerance",
            ],
        },
        {
            "name": "repair_closure_loop",
            "meaning": "前向选路与反向修复闭环对象",
            "support": _clip01(
                0.32 * fb["raw_forward_selectivity"]
                + 0.30 * fb["raw_backward_fidelity"]
                + 0.18 * fb["frontier_boundary_coupling"]
                + 0.20 * fb["raw_novelty_binding_capacity"]
            ),
            "basis": [
                "raw_forward_selectivity",
                "raw_backward_fidelity",
                "frontier_boundary_coupling",
            ],
        },
        {
            "name": "anchor_recurrence_family",
            "meaning": "跨随机种子的原生脑锚点家族",
            "support": _clip01(
                0.34 * anchors["generic_seed_recurrence_strength"]
                + 0.28 * anchors["dimension_specific_anchor_strength"]
                + 0.18 * anchors["layer_anchor_stability"]
                + 0.20 * (1.0 - anchors["anchor_ambiguity_penalty"] * 0.5)
            ),
            "basis": [
                "generic_seed_recurrence_strength",
                "dimension_specific_anchor_strength",
                "layer_anchor_stability",
            ],
        },
        {
            "name": "falsification_boundary_shell",
            "meaning": "理论成立与失效之间的边界壳层",
            "support": _clip01(
                0.28 * (1.0 - backfeed["summary_backfeed_risk_after"])
                + 0.24 * joint["evidence_isolation_joint"]
                + 0.20 * bridge["bridge_alignment_support"]
                + 0.28 * bridge["falsification_triggerability"]
            ),
            "basis": [
                "1-summary_backfeed_risk_after",
                "evidence_isolation_joint",
                "bridge_alignment_support",
            ],
        },
    ]

    axiom_records = [
        {
            "name": "projection_covariance_axiom",
            "statement": "语言现象是条件投影场在时间轴上的协变投影，而不是孤立模块输出。",
            "support": _clip01(
                0.38 * projection["cross_dimension_projection_stability"]
                + 0.32 * projection["reconstructed_context_gate_coherence"]
                + 0.30 * projection["raw_language_projection_score"]
            ),
        },
        {
            "name": "distributed_routing_axiom",
            "statement": "有效路由主要由分布式网络与纤维复用共同主导，而非单点神经元开关。",
            "support": _clip01(
                0.38 * route["distributed_network_support"]
                + 0.24 * route["route_structure_coupling_strength"]
                + 0.18 * route["degradation_tolerance"]
                + 0.20 * (0.5 + 0.5 * _clip01(route["route_scale_margin"] + 0.20))
            ),
        },
        {
            "name": "bounded_repair_axiom",
            "statement": "前向选路与反向修复形成有界闭环，闭环失效时首先表现为修复保真不足。",
            "support": _clip01(
                0.34 * fb["raw_forward_backward_rebuild_score"]
                + 0.26 * fb["raw_backward_fidelity"]
                + 0.20 * fb["loss_monotonicity"]
                + 0.20 * fb["raw_novelty_binding_capacity"]
            ),
        },
        {
            "name": "anchor_separability_axiom",
            "statement": "原生脑锚点必须在跨随机种子重现的同时保持足够可分性，否则不能进入理论主核。",
            "support": _clip01(
                0.34 * anchors["generic_seed_recurrence_strength"]
                + 0.22 * anchors["dimension_specific_anchor_strength"]
                + 0.14 * anchors["layer_anchor_stability"]
                + 0.30 * (1.0 - anchors["anchor_ambiguity_penalty"])
            ),
        },
        {
            "name": "falsifiable_boundary_axiom",
            "statement": "理论必须能被外部任务语境击穿，并且击穿路径要与内部弱链一致。",
            "support": _clip01(
                0.28 * bridge["bridge_alignment_support"]
                + 0.22 * bridge["falsification_triggerability"]
                + 0.22 * joint["real_world_bridge_joint"]
                + 0.28 * (1.0 - backfeed["summary_backfeed_risk_after"])
            ),
        },
    ]

    boundary_records = [
        {
            "name": "projection_boundary",
            "pressure": _clip01(1.0 - projection["raw_language_projection_score"]),
            "meaning": "语言投影协变不足时的失真边界",
        },
        {
            "name": "routing_boundary",
            "pressure": _clip01(1.0 - route["reconstructed_route_scale_score"]),
            "meaning": "路由退化与结构耦合破裂边界",
        },
        {
            "name": "repair_boundary",
            "pressure": _clip01(1.0 - fb["raw_forward_backward_rebuild_score"]),
            "meaning": "前后向闭环失效边界",
        },
        {
            "name": "evidence_boundary",
            "pressure": _clip01(backfeed["summary_backfeed_risk_after"]),
            "meaning": "证据回灌与隔离失败边界",
        },
        {
            "name": "anchor_boundary",
            "pressure": _clip01(anchors["anchor_ambiguity_penalty"]),
            "meaning": "脑锚点歧义导致对象不可分边界",
        },
    ]

    weakest_axiom = min(axiom_records, key=lambda item: item["support"])
    strongest_object = max(object_records, key=lambda item: item["support"])
    highest_boundary = max(boundary_records, key=lambda item: item["pressure"])

    object_layer_viability_score = _clip01(statistics_mean(item["support"] for item in object_records))
    axiom_layer_viability_score = _clip01(statistics_mean(item["support"] for item in axiom_records))
    boundary_layer_viability_score = _clip01(1.0 - statistics_mean(item["pressure"] for item in boundary_records))
    theorem_core_transition_gap = _clip01(
        max(
            1.0 - weakest_axiom["support"],
            highest_boundary["pressure"],
        )
    )
    math_theory_object_layer_score = _clip01(
        0.32 * object_layer_viability_score
        + 0.30 * axiom_layer_viability_score
        + 0.18 * boundary_layer_viability_score
        + 0.20 * (1.0 - theorem_core_transition_gap)
    )

    return {
        "headline_metrics": {
            "object_layer_viability_score": object_layer_viability_score,
            "axiom_layer_viability_score": axiom_layer_viability_score,
            "boundary_layer_viability_score": boundary_layer_viability_score,
            "strongest_object_name": strongest_object["name"],
            "weakest_axiom_name": weakest_axiom["name"],
            "weakest_axiom_score": weakest_axiom["support"],
            "highest_boundary_name": highest_boundary["name"],
            "highest_boundary_pressure": highest_boundary["pressure"],
            "theorem_core_transition_gap": theorem_core_transition_gap,
            "math_theory_object_layer_score": math_theory_object_layer_score,
        },
        "object_records": object_records,
        "axiom_records": axiom_records,
        "boundary_records": boundary_records,
        "status": {
            "status_short": (
                "math_theory_object_layer_ready"
                if math_theory_object_layer_score >= 0.62
                and weakest_axiom["support"] >= 0.45
                else "math_theory_object_layer_transition"
            ),
            "status_label": "新数学理论的对象层、公理层和边界层已经开始成形，但当前最弱公理与最高边界压力仍不足以支撑闭式理论。",
        },
        "project_readout": {
            "summary": "这一轮第一次把当前稳定拼图压成候选数学对象、候选公理和候选边界，不再只是分散的脚本指标。",
            "next_question": "下一步要把这些对象与公理组织成局部生成律，并用真实反例去打错的公理。",
        },
    }


def statistics_mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage107 Math Theory Object Layer Synthesis",
        "",
        f"- object_layer_viability_score: {hm['object_layer_viability_score']:.6f}",
        f"- axiom_layer_viability_score: {hm['axiom_layer_viability_score']:.6f}",
        f"- boundary_layer_viability_score: {hm['boundary_layer_viability_score']:.6f}",
        f"- strongest_object_name: {hm['strongest_object_name']}",
        f"- weakest_axiom_name: {hm['weakest_axiom_name']}",
        f"- weakest_axiom_score: {hm['weakest_axiom_score']:.6f}",
        f"- highest_boundary_name: {hm['highest_boundary_name']}",
        f"- highest_boundary_pressure: {hm['highest_boundary_pressure']:.6f}",
        f"- theorem_core_transition_gap: {hm['theorem_core_transition_gap']:.6f}",
        f"- math_theory_object_layer_score: {hm['math_theory_object_layer_score']:.6f}",
        f"- status_short: {summary['status']['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_math_theory_object_layer_synthesis_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
