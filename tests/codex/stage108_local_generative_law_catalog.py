from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage108_local_generative_law_catalog_20260322"
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
from stage107_math_theory_object_layer_synthesis import build_math_theory_object_layer_synthesis_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


@lru_cache(maxsize=1)
def build_local_generative_law_catalog_summary() -> dict:
    projection = build_tensor_level_language_projection_rebuild_summary()["headline_metrics"]
    route = build_tensor_level_route_scale_rebuild_summary()["headline_metrics"]
    fb = build_forward_backward_trace_rebuild_summary()["headline_metrics"]
    backfeed = build_backfeed_suppression_hardening_summary()["headline_metrics"]
    joint = build_brain_evidence_joint_closure_summary()["headline_metrics"]
    bridge = build_real_world_falsification_bridge_summary()["headline_metrics"]
    anchors = build_native_brain_anchor_search_summary()["headline_metrics"]
    theory = build_math_theory_object_layer_synthesis_summary()

    object_records = {record["name"]: record for record in theory["object_records"]}
    axiom_records = {record["name"]: record for record in theory["axiom_records"]}
    boundary_records = {record["name"]: record for record in theory["boundary_records"]}

    law_records = [
        {
            "name": "projection_transport_law",
            "object_name": "conditional_projection_field",
            "axiom_name": "projection_covariance_axiom",
            "failure_boundary_name": "projection_boundary",
            "statement": "当上下文门控、偏置传输与跨维稳定性同时维持时，语言投影会沿条件场稳定转运。",
            "local_update": "P_{t+1}=clip(0.34*C_t + 0.28*B_t + 0.22*S_t + 0.16*R_t, 0, 1)",
            "support": _clip01(
                0.34 * object_records["conditional_projection_field"]["support"]
                + 0.28 * axiom_records["projection_covariance_axiom"]["support"]
                + 0.22 * projection["raw_language_projection_score"]
                + 0.16 * (1.0 - boundary_records["projection_boundary"]["pressure"])
            ),
            "premise_support": [
                "reconstructed_context_gate_coherence",
                "reconstructed_bias_transport",
                "cross_dimension_projection_stability",
            ],
        },
        {
            "name": "distributed_route_settlement_law",
            "object_name": "distributed_route_fiber",
            "axiom_name": "distributed_routing_axiom",
            "failure_boundary_name": "routing_boundary",
            "statement": "当分布式网络支撑持续高于局部锚点支撑时，路由会在纤维复用结构上稳定沉降，而非退化成单点开关。",
            "local_update": "R_{t+1}=clip(0.36*D_t + 0.28*F_t + 0.20*T_t + 0.16*M_t, 0, 1)",
            "support": _clip01(
                0.34 * object_records["distributed_route_fiber"]["support"]
                + 0.28 * axiom_records["distributed_routing_axiom"]["support"]
                + 0.22 * route["reconstructed_route_scale_score"]
                + 0.16 * (1.0 - boundary_records["routing_boundary"]["pressure"])
            ),
            "premise_support": [
                "distributed_network_support",
                "route_structure_coupling_strength",
                "degradation_tolerance",
            ],
        },
        {
            "name": "bounded_repair_contraction_law",
            "object_name": "repair_closure_loop",
            "axiom_name": "bounded_repair_axiom",
            "failure_boundary_name": "repair_boundary",
            "statement": "当前向选择性与反向保真度同时成立时，误差修复会在有界闭环内收缩，而不是无限放大。",
            "local_update": "E_{t+1}=clip(0.38*F_t + 0.30*B_t + 0.18*N_t + 0.14*(1-L_t), 0, 1)",
            "support": _clip01(
                0.34 * object_records["repair_closure_loop"]["support"]
                + 0.28 * axiom_records["bounded_repair_axiom"]["support"]
                + 0.22 * fb["raw_forward_backward_rebuild_score"]
                + 0.16 * (1.0 - boundary_records["repair_boundary"]["pressure"])
            ),
            "premise_support": [
                "raw_forward_selectivity",
                "raw_backward_fidelity",
                "loss_monotonicity",
            ],
        },
        {
            "name": "anchor_refinement_law",
            "object_name": "anchor_recurrence_family",
            "axiom_name": "anchor_separability_axiom",
            "failure_boundary_name": "anchor_boundary",
            "statement": "脑锚点只有在跨随机种子重现且维度可分时，才能从候选对象提升为理论主核对象。",
            "local_update": "A_{t+1}=clip(0.34*G_t + 0.24*D_t + 0.22*L_t + 0.20*(1-U_t), 0, 1)",
            "support": _clip01(
                0.34 * object_records["anchor_recurrence_family"]["support"]
                + 0.28 * axiom_records["anchor_separability_axiom"]["support"]
                + 0.22 * anchors["native_brain_anchor_search_score"]
                + 0.16 * (1.0 - boundary_records["anchor_boundary"]["pressure"])
            ),
            "premise_support": [
                "generic_seed_recurrence_strength",
                "dimension_specific_anchor_strength",
                "layer_anchor_stability",
            ],
        },
        {
            "name": "boundary_exposure_law",
            "object_name": "falsification_boundary_shell",
            "axiom_name": "falsifiable_boundary_axiom",
            "failure_boundary_name": "evidence_boundary",
            "statement": "理论只有在外部任务语境能沿内部弱链稳定击穿时，才算真正暴露了自己的失效边界。",
            "local_update": "X_{t+1}=clip(0.30*B_t + 0.26*T_t + 0.22*J_t + 0.22*(1-R_t), 0, 1)",
            "support": _clip01(
                0.34 * object_records["falsification_boundary_shell"]["support"]
                + 0.28 * axiom_records["falsifiable_boundary_axiom"]["support"]
                + 0.18 * bridge["real_world_falsification_bridge_score"]
                + 0.20 * (1.0 - boundary_records["evidence_boundary"]["pressure"])
            ),
            "premise_support": [
                "bridge_alignment_support",
                "falsification_triggerability",
                "evidence_isolation_joint",
            ],
        },
    ]

    weakest_law = min(law_records, key=lambda item: item["support"])
    strongest_law = max(law_records, key=lambda item: item["support"])
    highest_failure_boundary = max(
        (boundary_records[record["failure_boundary_name"]] for record in law_records),
        key=lambda item: item["pressure"],
    )

    law_catalog_coverage = _clip01(_mean(record["support"] for record in law_records))
    law_composability_score = _clip01(
        0.30 * theory["headline_metrics"]["object_layer_viability_score"]
        + 0.26 * theory["headline_metrics"]["axiom_layer_viability_score"]
        + 0.22 * law_catalog_coverage
        + 0.22 * fb["frontier_boundary_coupling"]
    )
    law_failure_resilience = _clip01(
        1.0
        - _mean(boundary_records[record["failure_boundary_name"]]["pressure"] for record in law_records)
    )
    local_generative_law_catalog_score = _clip01(
        0.34 * law_catalog_coverage
        + 0.26 * law_composability_score
        + 0.18 * law_failure_resilience
        + 0.22 * (1.0 - boundary_records[weakest_law["failure_boundary_name"]]["pressure"])
    )

    return {
        "headline_metrics": {
            "law_catalog_coverage": law_catalog_coverage,
            "law_composability_score": law_composability_score,
            "law_failure_resilience": law_failure_resilience,
            "strongest_law_name": strongest_law["name"],
            "weakest_law_name": weakest_law["name"],
            "weakest_law_score": weakest_law["support"],
            "highest_failure_boundary_name": highest_failure_boundary["name"],
            "highest_failure_boundary_pressure": highest_failure_boundary["pressure"],
            "local_generative_law_catalog_score": local_generative_law_catalog_score,
        },
        "law_records": law_records,
        "status": {
            "status_short": (
                "local_generative_law_catalog_ready"
                if local_generative_law_catalog_score >= 0.60 and weakest_law["support"] >= 0.45
                else "local_generative_law_catalog_transition"
            ),
            "status_label": "候选数学对象已经进一步压成局部生成律目录，但最弱局部律和最高失败边界仍不足以支撑闭式理论。",
        },
        "project_readout": {
            "summary": "这一轮把对象层和公理层继续压成了可写局部更新式的局部生成律目录，使后续守恒量与边界量搜索有了直接抓手。",
            "next_question": "下一步要继续寻找这些局部律对应的守恒量、单调量和真实反例触发阈值。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage108 Local Generative Law Catalog",
        "",
        f"- law_catalog_coverage: {hm['law_catalog_coverage']:.6f}",
        f"- law_composability_score: {hm['law_composability_score']:.6f}",
        f"- law_failure_resilience: {hm['law_failure_resilience']:.6f}",
        f"- strongest_law_name: {hm['strongest_law_name']}",
        f"- weakest_law_name: {hm['weakest_law_name']}",
        f"- weakest_law_score: {hm['weakest_law_score']:.6f}",
        f"- highest_failure_boundary_name: {hm['highest_failure_boundary_name']}",
        f"- highest_failure_boundary_pressure: {hm['highest_failure_boundary_pressure']:.6f}",
        f"- local_generative_law_catalog_score: {hm['local_generative_law_catalog_score']:.6f}",
        f"- status_short: {summary['status']['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_local_generative_law_catalog_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
