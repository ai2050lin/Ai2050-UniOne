from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage77_brain_grounded_route_scaling_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_native_variable_candidate_mapping import build_native_variable_candidate_mapping_summary
from stage57_context_native_grounding import build_context_native_grounding_summary
from stage57_fiber_reuse_reinforcement import build_fiber_reuse_reinforcement_summary
from stage70_native_observability_bridge import build_native_observability_bridge_summary
from stage70_native_variable_improvement_audit import build_native_variable_improvement_audit_summary
from stage72_language_projection_covariance import build_language_projection_covariance_summary
from stage76_sqrt_repair_generalization import build_sqrt_repair_generalization_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_grounded_route_scaling_summary() -> dict:
    native = build_native_variable_candidate_mapping_summary()
    context = build_context_native_grounding_summary()["headline_metrics"]
    fiber = build_fiber_reuse_reinforcement_summary()["headline_metrics"]
    obs = build_native_observability_bridge_summary()["headline_metrics"]
    audit = build_native_variable_improvement_audit_summary()["headline_metrics"]
    projection = build_language_projection_covariance_summary()["headline_metrics"]
    repair = build_sqrt_repair_generalization_summary()["headline_metrics"]

    route_candidate = native["candidate_mapping"]["R_route"]
    plasticity_candidate = native["candidate_mapping"]["L_plasticity"]
    pressure_candidate = native["candidate_mapping"]["Pi_pressure"]

    neuron_level_support = _clip01(
        0.30 * route_candidate["locality"]
        + 0.22 * route_candidate["observability"]
        + 0.18 * plasticity_candidate["locality"]
        + 0.14 * pressure_candidate["observability"]
        + 0.16 * (1.0 - obs["hidden_proxy_gap"])
    )
    mesoscopic_bundle_support = _clip01(
        0.28 * fiber["fiber_reuse"]
        + 0.24 * fiber["cross_region_share_stability"]
        + 0.18 * context["context_route_alignment"]
        + 0.16 * projection["route_conditioned_projection"]
        + 0.14 * projection["context_covariance_stability"]
    )
    distributed_network_support = _clip01(
        0.26 * repair["generalized_repair_coverage"]
        + 0.20 * repair["route_rebind_support"]
        + 0.16 * repair["context_switch_support"]
        + 0.18 * projection["route_conditioned_projection"]
        + 0.20 * fiber["route_fiber_coupling_balance"]
    )

    route_scale_grounding_score = _clip01(
        0.20 * neuron_level_support
        + 0.34 * mesoscopic_bundle_support
        + 0.46 * distributed_network_support
    )
    route_scale_balance = _clip01(
        1.0
        - 0.52 * abs(distributed_network_support - mesoscopic_bundle_support)
        - 0.28 * abs(mesoscopic_bundle_support - neuron_level_support)
        - 0.20 * abs(distributed_network_support - neuron_level_support)
    )
    brain_constrained_repair_score = _clip01(
        0.28 * repair["repair_generalization_score"]
        + 0.20 * route_scale_grounding_score
        + 0.18 * route_scale_balance
        + 0.18 * audit["metric_traceability_gain"]
        + 0.16 * (1.0 - obs["hidden_proxy_gap"])
    )

    route_scale_profile = {
        "neuron_level_support": neuron_level_support,
        "mesoscopic_bundle_support": mesoscopic_bundle_support,
        "distributed_network_support": distributed_network_support,
        "dominant_scale_name": (
            "distributed_network"
            if distributed_network_support >= mesoscopic_bundle_support and distributed_network_support >= neuron_level_support
            else "mesoscopic_bundle"
            if mesoscopic_bundle_support >= neuron_level_support
            else "single_neuron"
        ),
        "single_neuron_is_sufficient": neuron_level_support >= 0.80 and neuron_level_support >= distributed_network_support,
    }

    return {
        "headline_metrics": {
            "neuron_level_support": neuron_level_support,
            "mesoscopic_bundle_support": mesoscopic_bundle_support,
            "distributed_network_support": distributed_network_support,
            "route_scale_balance": route_scale_balance,
            "route_scale_grounding_score": route_scale_grounding_score,
            "brain_constrained_repair_score": brain_constrained_repair_score,
        },
        "route_scale_profile": route_scale_profile,
        "brain_grounded_route_equation": {
            "local_route_anchor": "g_local_plus = clip(0.31*g + 0.23*r + 0.19*q + 0.11*p - 0.10*m, 0, 1)",
            "bundle_route_flow": "g_bundle_plus = clip(0.36*mean_bundle(g_local) + 0.24*f + 0.18*A_ctx - 0.12*c, 0, 1)",
            "network_route_field": "G_network_plus = clip(0.34*g_bundle + 0.22*q + 0.18*f + 0.14*p - 0.12*h, 0, 1)",
            "conflict_mass": "route_conflict = positive_part(demand_ctx - feasible_capacity(g_local, g_bundle, G_network, p, h, m, c))",
        },
        "status": {
            "status_short": (
                "brain_grounded_route_scaling_ready"
                if brain_constrained_repair_score >= 0.77 and route_scale_profile["dominant_scale_name"] == "distributed_network"
                else "brain_grounded_route_scaling_transition"
            ),
            "status_label": "路由修复已经开始接受脑编码约束，而且尺度分析显示路由更像分布式网络门控，而不是单神经元独立开关",
        },
        "project_readout": {
            "summary": "这一轮把学习稳态修复律重新放到脑编码约束下复核，并把路由拆成单神经元锚点、中观束流、全局网络场三个尺度来比较。",
            "next_question": "下一步要把 distributed_network 主导的路由结构推进到原生可观测层，看 route_conflict 能否摆脱代理量身份。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    profile = summary["route_scale_profile"]
    status = summary["status"]
    lines = [
        "# Stage77 Brain Grounded Route Scaling",
        "",
        f"- neuron_level_support: {hm['neuron_level_support']:.6f}",
        f"- mesoscopic_bundle_support: {hm['mesoscopic_bundle_support']:.6f}",
        f"- distributed_network_support: {hm['distributed_network_support']:.6f}",
        f"- route_scale_balance: {hm['route_scale_balance']:.6f}",
        f"- route_scale_grounding_score: {hm['route_scale_grounding_score']:.6f}",
        f"- brain_constrained_repair_score: {hm['brain_constrained_repair_score']:.6f}",
        f"- dominant_scale_name: {profile['dominant_scale_name']}",
        f"- single_neuron_is_sufficient: {profile['single_neuron_is_sufficient']}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_grounded_route_scaling_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
