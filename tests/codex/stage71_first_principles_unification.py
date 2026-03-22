from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage71_first_principles_unification_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_native_variable_candidate_mapping import build_native_variable_candidate_mapping_summary
from stage57_context_native_grounding import build_context_native_grounding_summary
from stage57_fiber_reuse_reinforcement import build_fiber_reuse_reinforcement_summary
from stage70_direct_identity_lock import build_direct_identity_lock_summary
from stage70_direct_stability_counterexample_probe import build_direct_stability_counterexample_probe_summary
from stage70_native_observability_bridge import build_native_observability_bridge_summary
from stage70_native_variable_improvement_audit import build_native_variable_improvement_audit_summary
from stage72_language_projection_covariance import build_language_projection_covariance_summary
from stage73_falsifiability_boundary_hardening import build_falsifiability_boundary_hardening_summary
from stage77_brain_grounded_route_scaling import build_brain_grounded_route_scaling_summary
from stage79_route_conflict_native_measure import build_route_conflict_native_measure_summary
from stage80_intelligence_closure_failure_map import build_intelligence_closure_failure_map_summary
from stage81_forward_backward_unification import build_forward_backward_unification_summary
from stage82_novelty_generalization_repair import build_novelty_generalization_repair_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_first_principles_unification_summary() -> dict:
    native = build_native_variable_candidate_mapping_summary()
    context = build_context_native_grounding_summary()
    fiber = build_fiber_reuse_reinforcement_summary()
    obs = build_native_observability_bridge_summary()
    audit = build_native_variable_improvement_audit_summary()
    counter = build_direct_stability_counterexample_probe_summary()
    identity = build_direct_identity_lock_summary()
    projection = build_language_projection_covariance_summary()
    fals_boundary = build_falsifiability_boundary_hardening_summary()
    route_scaling = build_brain_grounded_route_scaling_summary()
    route_conflict = build_route_conflict_native_measure_summary()
    intelligence_failure = build_intelligence_closure_failure_map_summary()
    forward_backward = build_forward_backward_unification_summary()
    novelty_repair = build_novelty_generalization_repair_summary()

    hn = native["headline_metrics"]
    hc = context["headline_metrics"]
    hf = fiber["headline_metrics"]
    ho = obs["headline_metrics"]
    ha = audit["headline_metrics"]
    hp = counter["headline_metrics"]
    hi = identity["headline_metrics"]
    hl = projection["headline_metrics"]
    hfz = fals_boundary["headline_metrics"]
    hrs = route_scaling["headline_metrics"]
    hrc = route_conflict["headline_metrics"]
    hif = intelligence_failure["headline_metrics"]
    hfb = forward_backward["headline_metrics"]
    hnr = novelty_repair["headline_metrics"]

    unified_state_readiness = _clip01(
        0.28 * hn["primitive_set_readiness"]
        + 0.22 * hn["native_mapping_completeness"]
        + 0.24 * ho["observability_bridge_score"]
        + 0.26 * ha["metric_traceability_gain"]
    )
    language_projection_coherence = _clip01(
        0.16 * hc["context_native_readiness"]
        + 0.14 * hc["conditional_gate_stability"]
        + 0.14 * hc["context_route_alignment"]
        + 0.12 * (1.0 - ho["hidden_proxy_gap"])
        + 0.18 * hl["context_covariance_stability"]
        + 0.14 * hl["route_conditioned_projection"]
        + 0.12 * hl["language_projection_repair_score"]
    )
    base_brain_encoding_groundedness = _clip01(
        0.28 * hn["native_mapping_completeness"]
        + 0.30 * ho["observability_bridge_score"]
        + 0.22 * ho["proxy_traceability_score"]
        + 0.20 * (1.0 - ho["hidden_proxy_gap"])
    )
    brain_encoding_groundedness = _clip01(
        0.74 * base_brain_encoding_groundedness
        + 0.14 * hrs["brain_constrained_repair_score"]
        + 0.12 * hrs["route_scale_grounding_score"]
    )
    base_intelligence_functional_closure = _clip01(
        0.24 * hi["locked_identity_readiness"]
        + 0.20 * hi["identity_lock_confidence"]
        + 0.18 * hp["adversarial_stability_support"]
        + 0.16 * ha["theorem_transparency_gain"]
        + 0.12 * hrc["inference_route_coherence"]
        + 0.10 * hrc["training_route_alignment"]
    )
    intelligence_functional_closure = _clip01(
        0.60 * base_intelligence_functional_closure
        + 0.08 * hif["intelligence_closure_failure_map_score"]
        + 0.12 * hfb["forward_backward_unification_score"]
        + 0.08 * hfb["novelty_binding_alignment"]
        + 0.08 * hnr["best_repaired_novelty_score"]
        + 0.04 * (1.0 - hnr["best_failure_after"])
    )
    local_generation_closure = _clip01(
        0.30 * hc["context_native_readiness"]
        + 0.28 * hf["reinforcement_readiness"]
        + 0.22 * hf["route_fiber_coupling_balance"]
        + 0.20 * hf["cross_region_share_stability"]
    )
    base_falsifiability_boundary_strength = _clip01(
        0.26 * hi["locked_identity_readiness"]
        + 0.24 * (1.0 - hp["counterexample_pressure"])
        + 0.24 * hp["adversarial_stability_support"]
        + 0.26 * ha["theorem_transparency_gain"]
    )
    falsifiability_boundary_strength = _clip01(
        0.74 * base_falsifiability_boundary_strength
        + 0.16 * hfz["falsifiability_boundary_hardening_score"]
        + 0.10 * hfz["weakest_failure_mode_score"]
    )
    first_principles_unification_score = _clip01(
        0.18 * unified_state_readiness
        + 0.16 * language_projection_coherence
        + 0.16 * brain_encoding_groundedness
        + 0.18 * intelligence_functional_closure
        + 0.16 * local_generation_closure
        + 0.16 * falsifiability_boundary_strength
    )

    weakest_axis_name, weakest_axis_score = min(
        (
            ("language_projection", language_projection_coherence),
            ("brain_grounding", brain_encoding_groundedness),
            ("intelligence_closure", intelligence_functional_closure),
            ("local_generation", local_generation_closure),
            ("falsifiability_boundary", falsifiability_boundary_strength),
        ),
        key=lambda item: item[1],
    )

    unified_state_variables = {
        "a": {
            "meaning": "局部激活密度",
            "language_role": "承载局部语义与句法片区激活",
            "brain_role": "承载局部回路活跃度",
            "intelligence_role": "承载任务状态进入与保持",
        },
        "r": {
            "meaning": "近邻回返一致性",
            "language_role": "约束局部结构连续性",
            "brain_role": "对应局部回返回路一致性",
            "intelligence_role": "支持短程稳定复用",
        },
        "f": {
            "meaning": "跨区共享纤维流",
            "language_role": "承载跨概念属性复用",
            "brain_role": "对应跨区共享路径",
            "intelligence_role": "支持迁移与组合",
        },
        "g": {
            "meaning": "门控路由概率",
            "language_role": "承载语义与逻辑路由选择",
            "brain_role": "对应回路门控选择",
            "intelligence_role": "控制推理路径切换",
        },
        "q": {
            "meaning": "条件门控场",
            "language_role": "承载上下文条件化",
            "brain_role": "对应条件依赖调制",
            "intelligence_role": "支持条件任务切换",
        },
        "b": {
            "meaning": "上下文偏置张量",
            "language_role": "承载风格与语境偏置",
            "brain_role": "对应背景调制偏置",
            "intelligence_role": "承载长期条件设定",
        },
        "p": {
            "meaning": "可塑性预算",
            "language_role": "限制新知识并入强度",
            "brain_role": "对应局部可塑资源",
            "intelligence_role": "约束学习与遗忘平衡",
        },
        "h": {
            "meaning": "稳态偏差",
            "language_role": "表现为结构漂移压力",
            "brain_role": "对应偏离稳态的局部失衡",
            "intelligence_role": "决定系统恢复难度",
        },
        "m": {
            "meaning": "拥塞与抑制负载",
            "language_role": "表现为路由拥挤与干扰",
            "brain_role": "对应抑制负载与冲突",
            "intelligence_role": "抬高任务切换成本",
        },
        "c": {
            "meaning": "最小传送成本梯度",
            "language_role": "约束跨结构传输代价",
            "brain_role": "对应路径组织成本",
            "intelligence_role": "约束推理与迁移耗散",
        },
    }

    local_law = {
        "patch_update": "a_plus = clip(0.32*a + 0.26*r + 0.18*q + 0.12*p - 0.12*h, 0, 1)",
        "route_update": "g_plus = clip(0.28*g + 0.22*r + 0.20*q + 0.12*f + 0.10*p - 0.14*c - 0.08*m, 0, 1)",
        "context_update": "q_plus = clip(0.34*q + 0.24*b + 0.18*g + 0.14*r + 0.10*p, 0, 1)",
        "fiber_update": "f_plus = clip(0.30*f + 0.24*min(g_i,g_j) + 0.22*min(a_i,a_j) + 0.14*min(q_i,q_j) + 0.10*(1-|h_i-h_j|), 0, 1)",
        "plasticity_update": "p_plus = clip(0.62*p + 0.16*(1-h) + 0.12*(1-m) - 0.10*novelty_load, 0, 1)",
        "pressure_update": "h_plus = clip(0.60*h + 0.18*c + 0.12*m + 0.10*route_conflict, 0, 1)",
    }

    projections = {
        "language_structure": "Y_lang = Phi_lang(a, r, g, q, b, f)",
        "brain_encoding": "Y_brain = Phi_brain(a, r, f, g, p, h, m, c)",
        "intelligence_function": "Y_intel = Phi_intel(g, q, f, p, h, m, c)",
    }

    falsification_boundaries = {
        "must_fail_if_context_is_not_covariant": "上下文切换后，q 与 g 的协同关系系统性失真且无法恢复",
        "must_fail_if_fiber_cannot_emerge": "局部更新长期无法稳定产生足够的跨区纤维复用",
        "must_fail_if_learning_breaks_stability": "新知识并入时，p 无法约束 h/m 上升，导致系统持续失稳",
        "must_fail_if_domains_cannot_share_state": "语言、脑编码、智能三个观测面无法由同一组状态变量共同解释",
    }

    status_short = (
        "first_principles_unification_frontier"
        if first_principles_unification_score >= 0.79 and falsifiability_boundary_strength >= 0.78
        else "first_principles_unification_transition"
    )

    return {
        "headline_metrics": {
            "unified_state_readiness": unified_state_readiness,
            "language_projection_coherence": language_projection_coherence,
            "brain_encoding_groundedness": brain_encoding_groundedness,
            "intelligence_functional_closure": intelligence_functional_closure,
            "local_generation_closure": local_generation_closure,
            "falsifiability_boundary_strength": falsifiability_boundary_strength,
            "first_principles_unification_score": first_principles_unification_score,
            "weakest_axis_name": weakest_axis_name,
            "weakest_axis_score": weakest_axis_score,
        },
        "unified_state_variables": unified_state_variables,
        "local_law": local_law,
        "language_projection_bridge": projection,
        "falsifiability_boundary_bridge": fals_boundary,
        "route_scaling_bridge": route_scaling,
        "route_conflict_bridge": route_conflict,
        "intelligence_closure_bridge": intelligence_failure,
        "forward_backward_bridge": forward_backward,
        "novelty_repair_bridge": novelty_repair,
        "projections": projections,
        "falsification_boundaries": falsification_boundaries,
        "status": {
            "status_short": status_short,
            "status_label": "语言结构、大脑编码、智能理论已经被压到同一状态系统上，但仍处在第一性原理统一过渡区",
        },
        "project_readout": {
            "summary": "这一轮首次把原生变量、上下文原生化、纤维复用、稳定性与身份锁定并成同一个第一性原理统一框架。",
            "next_question": "下一步要把统一局部生成律跑成可复现实验，并为四条判伪边界逐条设计反例测试。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage71 First Principles Unification",
        "",
        f"- unified_state_readiness: {hm['unified_state_readiness']:.6f}",
        f"- language_projection_coherence: {hm['language_projection_coherence']:.6f}",
        f"- brain_encoding_groundedness: {hm['brain_encoding_groundedness']:.6f}",
        f"- intelligence_functional_closure: {hm['intelligence_functional_closure']:.6f}",
        f"- local_generation_closure: {hm['local_generation_closure']:.6f}",
        f"- falsifiability_boundary_strength: {hm['falsifiability_boundary_strength']:.6f}",
        f"- first_principles_unification_score: {hm['first_principles_unification_score']:.6f}",
        f"- weakest_axis_name: {hm['weakest_axis_name']}",
        f"- weakest_axis_score: {hm['weakest_axis_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_first_principles_unification_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
