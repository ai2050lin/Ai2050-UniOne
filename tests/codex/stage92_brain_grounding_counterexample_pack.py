from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage92_brain_grounding_counterexample_pack_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage77_brain_grounded_route_scaling import build_brain_grounded_route_scaling_summary
from stage78_distributed_route_native_observability import build_distributed_route_native_observability_summary
from stage84_falsifiable_computation_core import build_falsifiable_computation_core_summary
from stage90_independent_observation_planes import build_independent_observation_planes_summary
from stage91_counterexample_attack_suite import build_counterexample_attack_suite_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def build_brain_grounding_counterexample_pack_summary() -> dict:
    scaling = build_brain_grounded_route_scaling_summary()["headline_metrics"]
    route_obs = build_distributed_route_native_observability_summary()["headline_metrics"]
    falsifiable = build_falsifiable_computation_core_summary()["headline_metrics"]
    planes = build_independent_observation_planes_summary()["headline_metrics"]
    attacks = build_counterexample_attack_suite_summary()

    brain_shock = next(
        record for record in attacks["attack_records"] if record["name"] == "brain_grounding_shock"
    )

    component_baseline = {
        "neuron_anchor": scaling["neuron_level_support"],
        "bundle_sync": scaling["mesoscopic_bundle_support"],
        "distributed_field": scaling["distributed_network_support"],
        "field_observability": 1.0 - route_obs["field_proxy_gap"],
        "repair_grounding": scaling["brain_constrained_repair_score"],
    }
    component_thresholds = {
        "neuron_anchor": 0.66,
        "bundle_sync": 0.64,
        "distributed_field": 0.74,
        "field_observability": 0.72,
        "repair_grounding": 0.74,
    }

    scenarios = [
        {
            "name": "neuron_anchor_collapse",
            "severity": 0.76,
            "damage": {
                "neuron_anchor": 0.26,
                "bundle_sync": 0.12,
                "distributed_field": 0.06,
                "field_observability": 0.10,
                "repair_grounding": 0.14,
            },
        },
        {
            "name": "bundle_desynchronization",
            "severity": 0.80,
            "damage": {
                "neuron_anchor": 0.10,
                "bundle_sync": 0.28,
                "distributed_field": 0.14,
                "field_observability": 0.14,
                "repair_grounding": 0.18,
            },
        },
        {
            "name": "field_proxy_blind_spot",
            "severity": 0.84,
            "damage": {
                "neuron_anchor": 0.06,
                "bundle_sync": 0.12,
                "distributed_field": 0.18,
                "field_observability": 0.30,
                "repair_grounding": 0.16,
            },
        },
        {
            "name": "plasticity_pressure_inversion",
            "severity": 0.78,
            "damage": {
                "neuron_anchor": 0.08,
                "bundle_sync": 0.16,
                "distributed_field": 0.16,
                "field_observability": 0.12,
                "repair_grounding": 0.30,
            },
        },
        {
            "name": "cross_modal_grounding_gap",
            "severity": 0.74,
            "damage": {
                "neuron_anchor": 0.10,
                "bundle_sync": 0.18,
                "distributed_field": 0.12,
                "field_observability": 0.18,
                "repair_grounding": 0.20,
            },
        },
        {
            "name": "distributed_field_fragmentation",
            "severity": 0.86,
            "damage": {
                "neuron_anchor": 0.06,
                "bundle_sync": 0.14,
                "distributed_field": 0.30,
                "field_observability": 0.24,
                "repair_grounding": 0.18,
            },
        },
    ]

    coupling_spillover = 0.10 * planes["variable_coupling_overlap"]
    split_resilience = 0.04 * planes["surface_anchor_independence"]
    brain_shock_spillover = 0.08 * brain_shock["attack_intensity"]
    refutation_drag = 0.06 * (1.0 - falsifiable["shared_state_refutation_power"])

    scenario_records = []
    multi_axis_break_count = 0
    weakest_component_name = None
    weakest_component_floor = 1.0
    for scenario in scenarios:
        component_after = {}
        breached_components = []
        for component_name, baseline in component_baseline.items():
            spillover = coupling_spillover + brain_shock_spillover + refutation_drag
            if component_name in {"distributed_field", "field_observability"}:
                spillover += 0.03 * planes["backfeed_risk_after_split"]
            if component_name == "repair_grounding":
                spillover += 0.04 * (1.0 - scaling["route_scale_balance"])
            attacked_value = _clip01(
                baseline
                - scenario["severity"] * scenario["damage"][component_name]
                - spillover
                + split_resilience
            )
            component_after[component_name] = attacked_value
            if attacked_value < component_thresholds[component_name]:
                breached_components.append(component_name)
            if attacked_value < weakest_component_floor:
                weakest_component_floor = attacked_value
                weakest_component_name = component_name

        if len(breached_components) >= 3:
            multi_axis_break_count += 1

        brain_break_intensity = _clip01(
            0.34 * scenario["severity"]
            + 0.24 * (len(breached_components) / len(component_baseline))
            + 0.22 * (1.0 - min(component_after.values()))
            + 0.20 * (coupling_spillover + brain_shock_spillover)
        )
        scenario_records.append(
            {
                "name": scenario["name"],
                "severity": scenario["severity"],
                "breached_components": breached_components,
                "breach_count": len(breached_components),
                "component_after": component_after,
                "brain_break_intensity": brain_break_intensity,
            }
        )

    hardest_counterexample = max(scenario_records, key=lambda item: item["brain_break_intensity"])
    multi_axis_grounding_break_rate = multi_axis_break_count / len(scenario_records)
    brain_counterexample_coverage = 1.0
    brain_grounding_residual = _clip01(
        sum(min(record["component_after"].values()) for record in scenario_records) / len(scenario_records)
    )
    brain_grounding_counterexample_score = _clip01(
        0.24 * brain_counterexample_coverage
        + 0.24 * multi_axis_grounding_break_rate
        + 0.18 * hardest_counterexample["brain_break_intensity"]
        + 0.18 * (1.0 - brain_grounding_residual)
        + 0.16 * (1.0 - weakest_component_floor)
    )

    return {
        "headline_metrics": {
            "brain_counterexample_coverage": brain_counterexample_coverage,
            "multi_axis_grounding_break_rate": multi_axis_grounding_break_rate,
            "hardest_counterexample_name": hardest_counterexample["name"],
            "hardest_counterexample_intensity": hardest_counterexample["brain_break_intensity"],
            "weakest_component_name": weakest_component_name,
            "weakest_component_floor": weakest_component_floor,
            "brain_grounding_residual": brain_grounding_residual,
            "brain_grounding_counterexample_score": brain_grounding_counterexample_score,
        },
        "scenario_records": scenario_records,
        "brain_shock_bridge": brain_shock,
        "status": {
            "status_short": (
                "brain_grounding_counterexample_pack_ready"
                if multi_axis_grounding_break_rate >= 0.66
                and hardest_counterexample["brain_break_intensity"] >= 0.60
                and weakest_component_name is not None
                else "brain_grounding_counterexample_pack_transition"
            ),
            "status_label": "脑编码落地反例包已经能把脑编码弱链拆成具体失配组件，但它首先证明的是脆弱性位置，而不是闭合已经完成。",
        },
        "project_readout": {
            "summary": "这一轮把脑编码落地拆成局部锚点、中观束流、分布式场、场可观测性和修复落地五个组件，并用独立反例包逐一攻击。",
            "next_question": "下一步要把这些脑编码失配接到跨平面失效传播图谱上，检查它们会如何同时拖垮语言、智能和可判伪面。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage92 Brain Grounding Counterexample Pack",
        "",
        f"- brain_counterexample_coverage: {hm['brain_counterexample_coverage']:.6f}",
        f"- multi_axis_grounding_break_rate: {hm['multi_axis_grounding_break_rate']:.6f}",
        f"- hardest_counterexample_name: {hm['hardest_counterexample_name']}",
        f"- hardest_counterexample_intensity: {hm['hardest_counterexample_intensity']:.6f}",
        f"- weakest_component_name: {hm['weakest_component_name']}",
        f"- weakest_component_floor: {hm['weakest_component_floor']:.6f}",
        f"- brain_grounding_residual: {hm['brain_grounding_residual']:.6f}",
        f"- brain_grounding_counterexample_score: {hm['brain_grounding_counterexample_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_grounding_counterexample_pack_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
