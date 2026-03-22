from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage94_cross_plane_failure_coupling_map_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage90_independent_observation_planes import build_independent_observation_planes_summary
from stage91_counterexample_attack_suite import build_counterexample_attack_suite_summary
from stage92_brain_grounding_counterexample_pack import build_brain_grounding_counterexample_pack_summary
from stage93_law_to_theorem_bridge import build_law_to_theorem_bridge_summary


PLANE_ORDER = [
    "language_plane",
    "brain_plane",
    "intelligence_plane",
    "falsification_plane",
]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _pair_key(left: str, right: str) -> str:
    return "__".join(sorted((left, right)))


@lru_cache(maxsize=1)
def build_cross_plane_failure_coupling_map_summary() -> dict:
    planes = build_independent_observation_planes_summary()
    attacks = build_counterexample_attack_suite_summary()
    brain = build_brain_grounding_counterexample_pack_summary()
    theorem = build_law_to_theorem_bridge_summary()["headline_metrics"]

    plane_signal = {
        record["name"]: record["signal_strength"]
        for record in planes["plane_records"]
    }
    overlap_lookup = {
        _pair_key(*record["pair"].split("__")): record["variable_overlap"]
        for record in planes["overlap_matrix"]
    }

    weakest_component = brain["headline_metrics"]["weakest_component_name"]
    weakest_floor = brain["headline_metrics"]["weakest_component_floor"]
    hardest_brain_intensity = brain["headline_metrics"]["hardest_counterexample_intensity"]
    hardest_attack_intensity = attacks["headline_metrics"]["hardest_attack_intensity"]
    theorem_ready_gap = theorem["theorem_ready_gap"]
    brain_compatibility_clause = theorem["brain_compatibility_clause"]

    edge_weights = []
    for source in PLANE_ORDER:
        for target in PLANE_ORDER:
            if source == target:
                continue
            variable_overlap = overlap_lookup[_pair_key(source, target)]
            edge_weight = _clip01(
                0.28
                + 0.30 * variable_overlap
                + 0.14 * (1.0 - plane_signal[source])
                + 0.12 * (1.0 - plane_signal[target])
                + (0.12 * (1.0 - brain_compatibility_clause) if source == "brain_plane" else 0.0)
                + (0.08 * theorem_ready_gap if target == "falsification_plane" else 0.0)
                + (0.05 * (1.0 - theorem["boundary_clause_strength"]) if target == "intelligence_plane" else 0.0)
                + (0.04 * (1.0 - theorem["premise_clause_strength"]) if target == "language_plane" else 0.0)
            )
            edge_weights.append(
                {
                    "source_plane": source,
                    "target_plane": target,
                    "variable_overlap": variable_overlap,
                    "edge_weight": edge_weight,
                }
            )

    edge_weight_lookup = {
        (record["source_plane"], record["target_plane"]): record["edge_weight"]
        for record in edge_weights
    }

    propagation_records = []

    for attack in attacks["attack_records"]:
        direct_impact = {
            plane_name: _clip01(1.0 - attacked_signal)
            for plane_name, attacked_signal in attack["plane_after"].items()
        }
        for source in PLANE_ORDER:
            if direct_impact[source] < 0.12:
                continue
            for target in PLANE_ORDER:
                if source == target:
                    continue
                path_intensity = _clip01(
                    direct_impact[source]
                    * edge_weight_lookup[(source, target)]
                    + 0.10 * attack["attack_intensity"]
                    + (0.04 * theorem_ready_gap if target == "falsification_plane" else 0.0)
                )
                propagation_records.append(
                    {
                        "event_name": attack["name"],
                        "event_type": "attack_suite",
                        "source_plane": source,
                        "target_plane": target,
                        "direct_source_impact": direct_impact[source],
                        "path_intensity": path_intensity,
                    }
                )

    scenario_component_bias = {
        "neuron_anchor": {
            "language_plane": 0.10,
            "brain_plane": 0.78,
            "intelligence_plane": 0.26,
            "falsification_plane": 0.18,
        },
        "bundle_sync": {
            "language_plane": 0.28,
            "brain_plane": 0.82,
            "intelligence_plane": 0.44,
            "falsification_plane": 0.26,
        },
        "distributed_field": {
            "language_plane": 0.34,
            "brain_plane": 0.86,
            "intelligence_plane": 0.40,
            "falsification_plane": 0.30,
        },
        "field_observability": {
            "language_plane": 0.30,
            "brain_plane": 0.72,
            "intelligence_plane": 0.34,
            "falsification_plane": 0.46,
        },
        "repair_grounding": {
            "language_plane": 0.18,
            "brain_plane": 0.70,
            "intelligence_plane": 0.46,
            "falsification_plane": 0.34,
        },
    }

    for scenario in brain["scenario_records"]:
        component_after = scenario["component_after"]
        direct_impact = {
            "brain_plane": _clip01(
                0.52 * scenario["brain_break_intensity"]
                + 0.28 * (1.0 - min(component_after.values()))
                + 0.20 * (len(scenario["breached_components"]) / len(component_after))
            )
        }
        for target in ("language_plane", "intelligence_plane", "falsification_plane"):
            component_bias = scenario_component_bias[weakest_component][target]
            target_component_loss = 1.0 - component_after[weakest_component]
            direct_impact[target] = _clip01(
                0.38 * scenario["brain_break_intensity"]
                + 0.24 * target_component_loss
                + 0.18 * component_bias
                + 0.12 * hardest_brain_intensity
                + 0.08 * theorem_ready_gap
            )

        for source in PLANE_ORDER:
            if direct_impact[source] < 0.12:
                continue
            for target in PLANE_ORDER:
                if source == target:
                    continue
                path_intensity = _clip01(
                    direct_impact[source]
                    * edge_weight_lookup[(source, target)]
                    + 0.12 * scenario["brain_break_intensity"]
                    + (0.05 * (1.0 - brain_compatibility_clause) if source == "brain_plane" else 0.0)
                )
                propagation_records.append(
                    {
                        "event_name": scenario["name"],
                        "event_type": "brain_counterexample",
                        "source_plane": source,
                        "target_plane": target,
                        "direct_source_impact": direct_impact[source],
                        "path_intensity": path_intensity,
                    }
                )

    receiver_loads = {plane_name: [] for plane_name in PLANE_ORDER}
    edge_grouped = {}
    for record in propagation_records:
        receiver_loads[record["target_plane"]].append(record["path_intensity"])
        edge_grouped.setdefault((record["source_plane"], record["target_plane"]), []).append(record["path_intensity"])

    coupling_paths = []
    for (source, target), values in edge_grouped.items():
        coupling_paths.append(
            {
                "source_plane": source,
                "target_plane": target,
                "mean_path_intensity": sum(values) / len(values),
                "max_path_intensity": max(values),
                "sample_count": len(values),
            }
        )

    hardest_path = max(coupling_paths, key=lambda item: item["max_path_intensity"])
    weakest_receiver_name = min(
        receiver_loads,
        key=lambda plane_name: plane_signal[plane_name] - (sum(receiver_loads[plane_name]) / len(receiver_loads[plane_name])),
    )
    weakest_receiver_floor = _clip01(
        plane_signal[weakest_receiver_name]
        - (sum(receiver_loads[weakest_receiver_name]) / len(receiver_loads[weakest_receiver_name]))
    )
    cross_plane_load_mean = sum(
        record["mean_path_intensity"] for record in coupling_paths
    ) / len(coupling_paths)
    theorem_spillover_pressure = _clip01(
        0.30 * (1.0 - brain_compatibility_clause)
        + 0.26 * theorem_ready_gap
        + 0.22 * hardest_brain_intensity
        + 0.22 * hardest_attack_intensity
    )
    propagation_coverage = len(coupling_paths) / (len(PLANE_ORDER) * (len(PLANE_ORDER) - 1))
    cross_plane_failure_coupling_score = _clip01(
        0.22 * propagation_coverage
        + 0.22 * hardest_path["max_path_intensity"]
        + 0.20 * cross_plane_load_mean
        + 0.18 * theorem_spillover_pressure
        + 0.18 * (1.0 - weakest_receiver_floor)
    )

    return {
        "headline_metrics": {
            "propagation_coverage": propagation_coverage,
            "hardest_coupling_path": f"{hardest_path['source_plane']}->{hardest_path['target_plane']}",
            "hardest_path_intensity": hardest_path["max_path_intensity"],
            "weakest_receiver_plane": weakest_receiver_name,
            "weakest_receiver_floor": weakest_receiver_floor,
            "cross_plane_load_mean": cross_plane_load_mean,
            "theorem_spillover_pressure": theorem_spillover_pressure,
            "cross_plane_failure_coupling_score": cross_plane_failure_coupling_score,
        },
        "edge_weights": edge_weights,
        "coupling_paths": coupling_paths,
        "propagation_records": propagation_records,
        "status": {
            "status_short": (
                "cross_plane_failure_coupling_map_ready"
                if propagation_coverage >= 0.95
                and hardest_path["max_path_intensity"] >= 0.58
                and theorem_spillover_pressure >= 0.55
                else "cross_plane_failure_coupling_map_transition"
            ),
            "status_label": "跨观测面失效耦合图谱已经能把脑编码、语言、智能与可判伪之间的失效传播写成路径，但目前更接近传播图谱，还不是闭式传播定理。",
        },
        "project_readout": {
            "summary": "这一轮把强攻击和脑编码反例接到同一张传播图谱上，开始明确失效不是单面破裂，而是会沿观测面之间的耦合路径扩散。",
            "next_question": "下一步要把这些传播路径推进到外部分布反例上，检验当前耦合图谱是否只在内部构造反例下成立。",
            "weakest_component_bridge": weakest_component,
            "brain_floor_bridge": weakest_floor,
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage94 Cross Plane Failure Coupling Map",
        "",
        f"- propagation_coverage: {hm['propagation_coverage']:.6f}",
        f"- hardest_coupling_path: {hm['hardest_coupling_path']}",
        f"- hardest_path_intensity: {hm['hardest_path_intensity']:.6f}",
        f"- weakest_receiver_plane: {hm['weakest_receiver_plane']}",
        f"- weakest_receiver_floor: {hm['weakest_receiver_floor']:.6f}",
        f"- cross_plane_load_mean: {hm['cross_plane_load_mean']:.6f}",
        f"- theorem_spillover_pressure: {hm['theorem_spillover_pressure']:.6f}",
        f"- cross_plane_failure_coupling_score: {hm['cross_plane_failure_coupling_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_cross_plane_failure_coupling_map_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
