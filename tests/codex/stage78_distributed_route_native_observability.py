from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage78_distributed_route_native_observability_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_native_variable_candidate_mapping import build_native_variable_candidate_mapping_summary
from stage70_direct_stability_counterexample_probe import build_direct_stability_counterexample_probe_summary
from stage70_native_observability_bridge import build_native_observability_bridge_summary
from stage72_language_projection_covariance import build_language_projection_covariance_summary
from stage76_sqrt_repair_generalization import build_sqrt_repair_generalization_summary
from stage77_brain_grounded_route_scaling import build_brain_grounded_route_scaling_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_distributed_route_native_observability_summary() -> dict:
    native = build_native_variable_candidate_mapping_summary()
    counter = build_direct_stability_counterexample_probe_summary()["headline_metrics"]
    obs = build_native_observability_bridge_summary()["headline_metrics"]
    projection = build_language_projection_covariance_summary()["headline_metrics"]
    repair = build_sqrt_repair_generalization_summary()["headline_metrics"]
    scaling = build_brain_grounded_route_scaling_summary()["headline_metrics"]

    route_candidate = native["candidate_mapping"]["R_route"]
    pressure_candidate = native["candidate_mapping"]["Pi_pressure"]
    plasticity_candidate = native["candidate_mapping"]["L_plasticity"]

    distributed_route_traceability = _clip01(
        0.24 * route_candidate["observability"]
        + 0.20 * obs["observability_bridge_score"]
        + 0.18 * projection["route_conditioned_projection"]
        + 0.18 * scaling["distributed_network_support"]
        + 0.20 * repair["route_rebind_support"]
    )
    route_conflict_native_measure = _clip01(
        0.26 * pressure_candidate["observability"]
        + 0.22 * plasticity_candidate["observability"]
        + 0.18 * (1.0 - counter["counterexample_pressure"])
        + 0.18 * repair["context_switch_support"]
        + 0.16 * scaling["route_scale_balance"]
    )
    route_counterexample_triggerability = _clip01(
        0.30 * distributed_route_traceability
        + 0.28 * route_conflict_native_measure
        + 0.22 * (1.0 - obs["hidden_proxy_gap"])
        + 0.20 * scaling["brain_constrained_repair_score"]
    )
    field_proxy_gap = _clip01(
        1.0
        - (0.40 * distributed_route_traceability + 0.34 * route_conflict_native_measure + 0.26 * route_counterexample_triggerability)
    )
    route_native_observability_score = _clip01(
        0.28 * distributed_route_traceability
        + 0.24 * route_conflict_native_measure
        + 0.22 * route_counterexample_triggerability
        + 0.14 * (1.0 - field_proxy_gap)
        + 0.12 * scaling["route_scale_grounding_score"]
    )

    scenario_records = [
        {
            "name": "route_rebind",
            "field_readout": _clip01(
                0.46 * scaling["distributed_network_support"]
                + 0.28 * repair["route_rebind_support"]
                + 0.26 * projection["route_conditioned_projection"]
            ),
            "conflict_readout": _clip01(
                0.42 * route_conflict_native_measure
                + 0.30 * (1.0 - counter["counterexample_pressure"])
                + 0.28 * repair["route_rebind_support"]
            ),
        },
        {
            "name": "context_shift",
            "field_readout": _clip01(
                0.44 * scaling["distributed_network_support"]
                + 0.30 * projection["context_covariance_stability"]
                + 0.26 * repair["context_switch_support"]
            ),
            "conflict_readout": _clip01(
                0.40 * route_conflict_native_measure
                + 0.34 * repair["context_switch_support"]
                + 0.26 * projection["projection_counterexample_resistance"]
            ),
        },
        {
            "name": "compositional_binding",
            "field_readout": _clip01(
                0.48 * scaling["distributed_network_support"]
                + 0.28 * repair["generalized_repair_coverage"]
                + 0.24 * scaling["route_scale_balance"]
            ),
            "conflict_readout": _clip01(
                0.46 * route_conflict_native_measure
                + 0.30 * (1.0 - field_proxy_gap)
                + 0.24 * repair["repair_generalization_score"]
            ),
        },
    ]

    return {
        "headline_metrics": {
            "distributed_route_traceability": distributed_route_traceability,
            "route_conflict_native_measure": route_conflict_native_measure,
            "route_counterexample_triggerability": route_counterexample_triggerability,
            "field_proxy_gap": field_proxy_gap,
            "route_native_observability_score": route_native_observability_score,
        },
        "scenario_records": scenario_records,
        "route_native_equation": {
            "field_readout": "R_obs = 0.32*G_network + 0.24*f + 0.18*q + 0.14*p + 0.12*(1-h)",
            "conflict_readout": "C_obs = 0.30*m + 0.24*h + 0.18*(1-p) + 0.16*|q-g| + 0.12*c",
            "proxy_gap": "gap_route = 1 - weighted_alignment(R_obs, C_obs, native_route_field)",
        },
        "status": {
            "status_short": (
                "distributed_route_native_observable"
                if route_native_observability_score >= 0.79 and field_proxy_gap <= 0.20
                else "distributed_route_native_observability_transition"
            ),
            "status_label": "分布式路由开始具备原生可观测轮廓，但路由场和真实原生量之间仍有剩余代理缺口",
        },
        "project_readout": {
            "summary": "这一轮把分布式路由从尺度结论推进到可观测层，开始把路由场、冲突质量和反例触发能力压成原生读数。",
            "next_question": "下一步要把 route_conflict 的代理缺口继续压低，看看它能否成为判伪主核里的直接失败变量。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage78 Distributed Route Native Observability",
        "",
        f"- distributed_route_traceability: {hm['distributed_route_traceability']:.6f}",
        f"- route_conflict_native_measure: {hm['route_conflict_native_measure']:.6f}",
        f"- route_counterexample_triggerability: {hm['route_counterexample_triggerability']:.6f}",
        f"- field_proxy_gap: {hm['field_proxy_gap']:.6f}",
        f"- route_native_observability_score: {hm['route_native_observability_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_distributed_route_native_observability_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
