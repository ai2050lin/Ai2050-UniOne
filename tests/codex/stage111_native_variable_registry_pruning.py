from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from stage100_backfeed_suppression_hardening import build_backfeed_suppression_hardening_summary
from stage101_brain_evidence_joint_closure import build_brain_evidence_joint_closure_summary
from stage102_real_world_falsification_bridge import build_real_world_falsification_bridge_summary
from stage103_native_brain_anchor_search import build_native_brain_anchor_search_summary
from stage104_tensor_level_language_projection_rebuild import build_tensor_level_language_projection_rebuild_summary
from stage105_tensor_level_route_scale_rebuild import build_tensor_level_route_scale_rebuild_summary
from stage106_forward_backward_trace_rebuild import build_forward_backward_trace_rebuild_summary
from stage107_math_theory_object_layer_synthesis import build_math_theory_object_layer_synthesis_summary
from stage108_local_generative_law_catalog import build_local_generative_law_catalog_summary
from stage109_invariant_boundary_quantity_search import build_invariant_boundary_quantity_search_summary
from stage110_axiom_falsification_suite import build_axiom_falsification_suite_summary


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage111_native_variable_registry_pruning_20260322"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


@lru_cache(maxsize=1)
def build_native_variable_registry_pruning_summary() -> dict:
    projection = build_tensor_level_language_projection_rebuild_summary()["headline_metrics"]
    route = build_tensor_level_route_scale_rebuild_summary()["headline_metrics"]
    fb = build_forward_backward_trace_rebuild_summary()["headline_metrics"]
    backfeed = build_backfeed_suppression_hardening_summary()["headline_metrics"]
    joint = build_brain_evidence_joint_closure_summary()["headline_metrics"]
    bridge = build_real_world_falsification_bridge_summary()["headline_metrics"]
    anchors = build_native_brain_anchor_search_summary()["headline_metrics"]
    theory = build_math_theory_object_layer_synthesis_summary()
    laws = build_local_generative_law_catalog_summary()["headline_metrics"]
    quantities = build_invariant_boundary_quantity_search_summary()
    attacks = build_axiom_falsification_suite_summary()["headline_metrics"]

    quantity_map = {item["name"]: item for item in quantities["quantity_records"]}
    object_map = {item["name"]: item for item in theory["object_records"]}
    boundary_map = {item["name"]: item for item in quantities["boundary_records"]}

    registry_records = [
        {
            "name": "conditional_projection_field",
            "meaning": "语言在上下文条件场上的稳定投影对象。",
            "support": object_map["conditional_projection_field"]["support"],
            "observability": projection["raw_language_projection_score"],
            "independence": _clip01(0.58 * projection["cross_dimension_separation"] + 0.42 * (1.0 - backfeed["summary_backfeed_risk_after"])),
            "attack_resilience": _clip01(0.62 * (1.0 - attacks["strongest_attack_intensity"]) + 0.38 * projection["cross_dimension_projection_stability"]),
            "suggested_role": "native_core",
        },
        {
            "name": "distributed_route_fiber",
            "meaning": "分布式路由与纤维复用的主导对象。",
            "support": object_map["distributed_route_fiber"]["support"],
            "observability": route["reconstructed_route_scale_score"],
            "independence": _clip01(0.50 * route["route_scale_margin"] * 0.5 + 0.25 + 0.50 * route["route_structure_coupling_strength"]),
            "attack_resilience": _clip01(0.56 * (1.0 - attacks["strongest_attack_intensity"]) + 0.44 * route["degradation_tolerance"]),
            "suggested_role": "native_core",
        },
        {
            "name": "repair_closure_loop",
            "meaning": "前向选路与反向修复的闭环对象。",
            "support": object_map["repair_closure_loop"]["support"],
            "observability": fb["raw_forward_backward_rebuild_score"],
            "independence": _clip01(0.52 * fb["loss_monotonicity"] + 0.48 * (1.0 - backfeed["summary_backfeed_risk_after"])),
            "attack_resilience": _clip01(0.54 * fb["raw_backward_fidelity"] + 0.46 * attacks["falsification_survival_score"]),
            "suggested_role": "native_core",
        },
        {
            "name": "anchor_recurrence_family",
            "meaning": "跨随机种子重现的脑锚点家族。",
            "support": object_map["anchor_recurrence_family"]["support"],
            "observability": anchors["native_brain_anchor_search_score"],
            "independence": _clip01(0.44 * anchors["dimension_specific_anchor_strength"] + 0.56 * (1.0 - anchors["anchor_ambiguity_penalty"])),
            "attack_resilience": _clip01(0.52 * anchors["closure_bridge_support"] + 0.48 * (1.0 - attacks["strongest_attack_intensity"])),
            "suggested_role": "native_core",
        },
        {
            "name": "hierarchical_concept_span_quantity",
            "meaning": "概念在微观、中观、宏观层之间可转运的跨度量。",
            "support": quantity_map["hierarchical_concept_span_quantity"]["support"],
            "observability": _clip01(0.54 * quantities["headline_metrics"]["invariant_quantity_strength"] + 0.46 * projection["raw_language_projection_score"]),
            "independence": _clip01(0.56 * (1.0 - boundary_map["macro_data_gap_boundary"]["pressure"]) + 0.44 * quantities["headline_metrics"]["boundary_quantity_resilience"]),
            "attack_resilience": _clip01(0.48 * attacks["falsification_survival_score"] + 0.52 * (1.0 - boundary_map["task_bridge_boundary"]["pressure"])),
            "suggested_role": "native_core",
        },
        {
            "name": "context_covariant_uniqueness_quantity",
            "meaning": "风格、逻辑、语法协变下的全局唯一性量。",
            "support": quantity_map["context_covariant_uniqueness_quantity"]["support"],
            "observability": _clip01(0.52 * projection["reconstructed_context_gate_coherence"] + 0.48 * projection["reconstructed_route_projection"]),
            "independence": _clip01(0.52 * projection["cross_dimension_separation"] + 0.48 * (1.0 - backfeed["summary_backfeed_risk_after"])),
            "attack_resilience": _clip01(0.50 * (1.0 - attacks["strongest_attack_intensity"]) + 0.50 * attacks["falsification_survival_score"]),
            "suggested_role": "native_core",
        },
        {
            "name": "minimal_transport_efficiency_quantity",
            "meaning": "最小传送与分布式沉降共同决定的效率量。",
            "support": quantity_map["minimal_transport_efficiency_quantity"]["support"],
            "observability": _clip01(0.52 * route["reconstructed_route_scale_score"] + 0.48 * route["degradation_tolerance"]),
            "independence": _clip01(0.44 * route["route_scale_margin"] * 0.5 + 0.22 + 0.56 * anchors["generic_seed_recurrence_strength"]),
            "attack_resilience": _clip01(0.52 * route["degradation_tolerance"] + 0.48 * (1.0 - attacks["strongest_attack_intensity"])),
            "suggested_role": "native_core",
        },
        {
            "name": "relational_linearity_quantity",
            "meaning": "关系可线性搬运的残余结构量。",
            "support": quantity_map["relational_linearity_quantity"]["support"],
            "observability": _clip01(0.48 * projection["cross_dimension_projection_stability"] + 0.52 * quantities["headline_metrics"]["invariant_quantity_strength"]),
            "independence": _clip01(1.0 - boundary_map["linearity_proof_boundary"]["pressure"]),
            "attack_resilience": _clip01(0.42 * attacks["falsification_survival_score"] + 0.58 * (1.0 - boundary_map["linearity_proof_boundary"]["pressure"])),
            "suggested_role": "projection",
        },
        {
            "name": "repair_stability_quantity",
            "meaning": "及时学习与全局稳态同时成立的稳态量。",
            "support": quantity_map["repair_stability_quantity"]["support"],
            "observability": _clip01(0.50 * fb["raw_backward_fidelity"] + 0.50 * fb["loss_monotonicity"]),
            "independence": _clip01(0.52 * (1.0 - backfeed["summary_backfeed_risk_after"]) + 0.48 * joint["field_observability_joint"]),
            "attack_resilience": _clip01(0.50 * attacks["falsification_survival_score"] + 0.50 * fb["raw_backward_fidelity"]),
            "suggested_role": "projection",
        },
        {
            "name": "falsification_boundary_shell",
            "meaning": "理论成立与失效之间的边界壳层。",
            "support": object_map["falsification_boundary_shell"]["support"],
            "observability": bridge["real_world_falsification_bridge_score"],
            "independence": _clip01(0.44 * joint["evidence_isolation_joint"] + 0.56 * (1.0 - backfeed["summary_backfeed_risk_after"])),
            "attack_resilience": _clip01(0.36 * attacks["weakest_axiom_after_attack_score"] + 0.64 * (1.0 - quantities["headline_metrics"]["highest_boundary_pressure"])),
            "suggested_role": "projection",
        },
        {
            "name": "raw_language_projection_score",
            "meaning": "语言投影重建链的摘要读数。",
            "support": projection["raw_language_projection_score"],
            "observability": 1.0,
            "independence": _clip01(1.0 - backfeed["summary_backfeed_risk_after"]),
            "attack_resilience": _clip01(1.0 - attacks["strongest_attack_intensity"]),
            "suggested_role": "proxy",
        },
        {
            "name": "reconstructed_route_scale_score",
            "meaning": "路由尺度重建链的摘要读数。",
            "support": route["reconstructed_route_scale_score"],
            "observability": 1.0,
            "independence": _clip01(0.55 * route["route_scale_margin"] * 0.5 + 0.25 + 0.45 * (1.0 - backfeed["summary_backfeed_risk_after"])),
            "attack_resilience": _clip01(1.0 - attacks["strongest_attack_intensity"]),
            "suggested_role": "proxy",
        },
        {
            "name": "summary_backfeed_risk_after",
            "meaning": "摘要回灌残余风险。",
            "support": _clip01(1.0 - backfeed["summary_backfeed_risk_after"]),
            "observability": 1.0,
            "independence": _clip01(1.0 - backfeed["summary_backfeed_risk_after"]),
            "attack_resilience": _clip01(1.0 - backfeed["summary_backfeed_risk_after"]),
            "suggested_role": "proxy",
        },
        {
            "name": "task_bridge_boundary",
            "meaning": "候选理论量跨进真实任务世界时的最高边界。",
            "support": _clip01(1.0 - boundary_map["task_bridge_boundary"]["pressure"]),
            "observability": bridge["real_world_falsification_bridge_score"],
            "independence": _clip01(1.0 - boundary_map["task_bridge_boundary"]["pressure"]),
            "attack_resilience": _clip01(1.0 - attacks["task_bridge_retest_pressure"]),
            "suggested_role": "deferred",
        },
    ]

    for record in registry_records:
        record["native_eligibility"] = _clip01(
            0.34 * record["support"]
            + 0.24 * record["observability"]
            + 0.20 * record["independence"]
            + 0.22 * record["attack_resilience"]
        )

    for record in registry_records:
        if record["suggested_role"] == "native_core":
            if record["native_eligibility"] >= 0.68 and record["attack_resilience"] >= 0.40:
                record["registry_role"] = "native_core"
            else:
                record["registry_role"] = "projection"
        elif record["suggested_role"] == "projection":
            if record["native_eligibility"] >= 0.52:
                record["registry_role"] = "projection"
            else:
                record["registry_role"] = "deferred"
        elif record["suggested_role"] == "proxy":
            record["registry_role"] = "proxy"
        else:
            record["registry_role"] = "deferred"

    native_records = [r for r in registry_records if r["registry_role"] == "native_core"]
    projection_records = [r for r in registry_records if r["registry_role"] == "projection"]
    proxy_records = [r for r in registry_records if r["registry_role"] == "proxy"]
    deferred_records = [r for r in registry_records if r["registry_role"] == "deferred"]

    strongest_native = max(native_records, key=lambda item: item["native_eligibility"])
    weakest_native = min(native_records, key=lambda item: item["native_eligibility"])

    native_variable_purity = _clip01(
        0.40 * _mean(item["independence"] for item in native_records)
        + 0.32 * _mean(item["attack_resilience"] for item in native_records)
        + 0.28 * _mean(item["observability"] for item in native_records)
    )
    proxy_load_penalty = _clip01(len(proxy_records) / max(1, len(registry_records)))
    deferred_pressure = _clip01(_mean(1.0 - item["native_eligibility"] for item in deferred_records))
    native_variable_registry_pruning_score = _clip01(
        0.34 * native_variable_purity
        + 0.24 * _clip01(len(native_records) / 7.0)
        + 0.18 * (1.0 - proxy_load_penalty)
        + 0.24 * (1.0 - deferred_pressure)
    )

    return {
        "headline_metrics": {
            "native_core_variable_count": len(native_records),
            "projection_variable_count": len(projection_records),
            "proxy_variable_count": len(proxy_records),
            "deferred_variable_count": len(deferred_records),
            "strongest_native_name": strongest_native["name"],
            "weakest_native_name": weakest_native["name"],
            "weakest_native_score": weakest_native["native_eligibility"],
            "native_variable_purity": native_variable_purity,
            "proxy_load_penalty": proxy_load_penalty,
            "native_variable_registry_pruning_score": native_variable_registry_pruning_score,
        },
        "registry_records": registry_records,
        "status": {
            "status_short": (
                "native_variable_registry_pruning_ready"
                if native_variable_registry_pruning_score >= 0.58 and len(native_records) >= 5
                else "native_variable_registry_pruning_transition"
            ),
            "status_label": "变量注册表已经开始把理论主核变量、投影变量和代理变量分层，但最弱主核变量与待延期变量仍然偏多。",
        },
        "project_readout": {
            "summary": "这一轮第一次把对象、局部律、候选守恒量、攻击结果统一整理成变量注册表，不再让主核变量和摘要代理量继续混在一起。",
            "next_question": "下一步要继续验证当前主核变量是否能跨真实任务稳定成立，以及哪些延期变量可以被替换或淘汰。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage111 Native Variable Registry Pruning",
        "",
        f"- native_core_variable_count: {hm['native_core_variable_count']}",
        f"- projection_variable_count: {hm['projection_variable_count']}",
        f"- proxy_variable_count: {hm['proxy_variable_count']}",
        f"- deferred_variable_count: {hm['deferred_variable_count']}",
        f"- strongest_native_name: {hm['strongest_native_name']}",
        f"- weakest_native_name: {hm['weakest_native_name']}",
        f"- weakest_native_score: {hm['weakest_native_score']:.6f}",
        f"- native_variable_purity: {hm['native_variable_purity']:.6f}",
        f"- proxy_load_penalty: {hm['proxy_load_penalty']:.6f}",
        f"- native_variable_registry_pruning_score: {hm['native_variable_registry_pruning_score']:.6f}",
        f"- status_short: {summary['status']['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_native_variable_registry_pruning_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
