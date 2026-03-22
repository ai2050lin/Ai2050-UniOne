from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage95_external_distribution_counterexample_pack_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage88_external_counterexample_expansion import build_external_counterexample_expansion_summary
from stage90_independent_observation_planes import build_independent_observation_planes_summary
from stage92_brain_grounding_counterexample_pack import build_brain_grounding_counterexample_pack_summary
from stage94_cross_plane_failure_coupling_map import build_cross_plane_failure_coupling_map_summary


PLANE_ORDER = [
    "language_plane",
    "brain_plane",
    "intelligence_plane",
    "falsification_plane",
]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def build_external_distribution_counterexample_pack_summary() -> dict:
    external = build_external_counterexample_expansion_summary()["headline_metrics"]
    planes = build_independent_observation_planes_summary()
    brain = build_brain_grounding_counterexample_pack_summary()["headline_metrics"]
    coupling = build_cross_plane_failure_coupling_map_summary()

    plane_signal = {
        record["name"]: record["signal_strength"]
        for record in planes["plane_records"]
    }
    edge_lookup = {
        (record["source_plane"], record["target_plane"]): record["edge_weight"]
        for record in coupling["edge_weights"]
    }
    internal_hardest_path = coupling["headline_metrics"]["hardest_coupling_path"]
    weakest_component = brain["weakest_component_name"]
    weakest_floor = brain["weakest_component_floor"]

    external_families = [
        {
            "name": "cross_corpus_domain_shift",
            "distribution_shift": 0.84,
            "temporal_irregularity": 0.40,
            "symbolic_aliasing": 0.62,
            "grounding_blindness": 0.58,
            "boundary_stress": 0.56,
        },
        {
            "name": "sensorimotor_channel_gap",
            "distribution_shift": 0.66,
            "temporal_irregularity": 0.34,
            "symbolic_aliasing": 0.44,
            "grounding_blindness": 0.88,
            "boundary_stress": 0.52,
        },
        {
            "name": "temporal_resolution_drop",
            "distribution_shift": 0.60,
            "temporal_irregularity": 0.86,
            "symbolic_aliasing": 0.42,
            "grounding_blindness": 0.54,
            "boundary_stress": 0.60,
        },
        {
            "name": "cross_lingual_projection_drift",
            "distribution_shift": 0.74,
            "temporal_irregularity": 0.42,
            "symbolic_aliasing": 0.82,
            "grounding_blindness": 0.62,
            "boundary_stress": 0.50,
        },
        {
            "name": "long_horizon_context_scatter",
            "distribution_shift": 0.68,
            "temporal_irregularity": 0.78,
            "symbolic_aliasing": 0.58,
            "grounding_blindness": 0.56,
            "boundary_stress": 0.68,
        },
        {
            "name": "embodied_reference_void",
            "distribution_shift": 0.70,
            "temporal_irregularity": 0.30,
            "symbolic_aliasing": 0.48,
            "grounding_blindness": 0.92,
            "boundary_stress": 0.58,
        },
        {
            "name": "rule_space_reindexing",
            "distribution_shift": 0.64,
            "temporal_irregularity": 0.54,
            "symbolic_aliasing": 0.78,
            "grounding_blindness": 0.46,
            "boundary_stress": 0.66,
        },
        {
            "name": "adversarial_external_mixture",
            "distribution_shift": 0.86,
            "temporal_irregularity": 0.70,
            "symbolic_aliasing": 0.80,
            "grounding_blindness": 0.84,
            "boundary_stress": 0.74,
        },
    ]

    source_bias = {
        "language_plane": {
            "distribution_shift": 0.28,
            "temporal_irregularity": 0.14,
            "symbolic_aliasing": 0.34,
            "grounding_blindness": 0.12,
            "boundary_stress": 0.12,
        },
        "brain_plane": {
            "distribution_shift": 0.18,
            "temporal_irregularity": 0.10,
            "symbolic_aliasing": 0.08,
            "grounding_blindness": 0.44,
            "boundary_stress": 0.20,
        },
        "intelligence_plane": {
            "distribution_shift": 0.18,
            "temporal_irregularity": 0.30,
            "symbolic_aliasing": 0.16,
            "grounding_blindness": 0.12,
            "boundary_stress": 0.24,
        },
        "falsification_plane": {
            "distribution_shift": 0.12,
            "temporal_irregularity": 0.10,
            "symbolic_aliasing": 0.14,
            "grounding_blindness": 0.20,
            "boundary_stress": 0.44,
        },
    }

    receiver_thresholds = {
        "language_plane": 0.46,
        "brain_plane": 0.44,
        "intelligence_plane": 0.45,
        "falsification_plane": 0.42,
    }

    sample_records = []
    aligned_count = 0
    triggered_count = 0
    weakest_receiver_name = None
    weakest_receiver_floor = 1.0

    for family in external_families:
        source_impacts = {}
        for plane_name in PLANE_ORDER:
            base = 0.0
            for axis_name, weight in source_bias[plane_name].items():
                base += weight * family[axis_name]
            if plane_name == "brain_plane":
                base += 0.08 * (1.0 - weakest_floor) + 0.06 * brain["hardest_counterexample_intensity"]
                if weakest_component == "field_observability":
                    base += 0.04
            if plane_name == "falsification_plane":
                base += 0.08 * external["expanded_trigger_rate"]
            source_impacts[plane_name] = _clip01(base)

        path_records = []
        for source in PLANE_ORDER:
            for target in PLANE_ORDER:
                if source == target:
                    continue
                path_intensity = _clip01(
                    source_impacts[source] * edge_lookup[(source, target)]
                    + 0.10 * family["distribution_shift"]
                    + (0.06 * family["grounding_blindness"] if source == "brain_plane" else 0.0)
                    + (0.05 * family["boundary_stress"] if target == "falsification_plane" else 0.0)
                    + (0.04 * family["symbolic_aliasing"] if target == "language_plane" else 0.0)
                    + (0.04 * family["temporal_irregularity"] if target == "intelligence_plane" else 0.0)
                )
                path_records.append(
                    {
                        "source_plane": source,
                        "target_plane": target,
                        "path": f"{source}->{target}",
                        "path_intensity": path_intensity,
                    }
                )

        strongest_path = max(path_records, key=lambda item: item["path_intensity"])
        if strongest_path["path"] == internal_hardest_path:
            aligned_count += 1

        receiver_floor_map = {}
        breached_receivers = []
        for plane_name in PLANE_ORDER:
            incoming_values = [
                record["path_intensity"]
                for record in path_records
                if record["target_plane"] == plane_name
            ]
            receiver_floor = _clip01(plane_signal[plane_name] - (sum(incoming_values) / len(incoming_values)))
            receiver_floor_map[plane_name] = receiver_floor
            if receiver_floor < receiver_thresholds[plane_name]:
                breached_receivers.append(plane_name)
            if receiver_floor < weakest_receiver_floor:
                weakest_receiver_floor = receiver_floor
                weakest_receiver_name = plane_name

        external_triggered = strongest_path["path_intensity"] >= 0.56 and len(breached_receivers) >= 2
        if external_triggered:
            triggered_count += 1

        sample_records.append(
            {
                "name": family["name"],
                "distribution_axes": {
                    "distribution_shift": family["distribution_shift"],
                    "temporal_irregularity": family["temporal_irregularity"],
                    "symbolic_aliasing": family["symbolic_aliasing"],
                    "grounding_blindness": family["grounding_blindness"],
                    "boundary_stress": family["boundary_stress"],
                },
                "source_impacts": source_impacts,
                "strongest_path": strongest_path["path"],
                "strongest_path_intensity": strongest_path["path_intensity"],
                "receiver_floor_map": receiver_floor_map,
                "breached_receivers": breached_receivers,
                "external_triggered": external_triggered,
            }
        )

    strongest_family = max(sample_records, key=lambda item: item["strongest_path_intensity"])
    external_family_coverage = 1.0
    external_trigger_rate = triggered_count / len(sample_records)
    path_alignment_rate = aligned_count / len(sample_records)
    mean_strongest_path_intensity = sum(item["strongest_path_intensity"] for item in sample_records) / len(sample_records)
    mean_receiver_floor = sum(
        min(item["receiver_floor_map"].values()) for item in sample_records
    ) / len(sample_records)
    external_distribution_counterexample_score = _clip01(
        0.22 * external_family_coverage
        + 0.22 * external_trigger_rate
        + 0.18 * path_alignment_rate
        + 0.18 * strongest_family["strongest_path_intensity"]
        + 0.10 * mean_strongest_path_intensity
        + 0.10 * (1.0 - mean_receiver_floor)
    )

    return {
        "headline_metrics": {
            "external_family_coverage": external_family_coverage,
            "external_trigger_rate": external_trigger_rate,
            "path_alignment_rate": path_alignment_rate,
            "hardest_external_family_name": strongest_family["name"],
            "hardest_external_path": strongest_family["strongest_path"],
            "hardest_external_intensity": strongest_family["strongest_path_intensity"],
            "weakest_external_receiver": weakest_receiver_name,
            "weakest_external_receiver_floor": weakest_receiver_floor,
            "mean_strongest_path_intensity": mean_strongest_path_intensity,
            "external_distribution_counterexample_score": external_distribution_counterexample_score,
        },
        "sample_records": sample_records,
        "internal_bridge": {
            "internal_hardest_path": internal_hardest_path,
            "cross_plane_failure_coupling_score": coupling["headline_metrics"]["cross_plane_failure_coupling_score"],
            "weakest_component_bridge": weakest_component,
        },
        "status": {
            "status_short": (
                "external_distribution_counterexample_pack_ready"
                if external_trigger_rate >= 0.60
                and path_alignment_rate >= 0.35
                and strongest_family["strongest_path_intensity"] >= 0.62
                else "external_distribution_counterexample_pack_transition"
            ),
            "status_label": "外部分布反例包已经能在更接近外部刺激的样本族上复现跨平面传播，但当前仍是外部分布近似，不是真实外部数据闭合。",
        },
        "project_readout": {
            "summary": "这一轮把外部分布型反例样本接到 Stage94 的传播图谱上，开始检查外部攻击是否会复现内部最危险传播主链。",
            "next_question": "下一步要把这些外部分布传播路径与统一主核的独立证据链对齐，检验当前高层判断是否仍然过度依赖内部摘要回灌。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage95 External Distribution Counterexample Pack",
        "",
        f"- external_family_coverage: {hm['external_family_coverage']:.6f}",
        f"- external_trigger_rate: {hm['external_trigger_rate']:.6f}",
        f"- path_alignment_rate: {hm['path_alignment_rate']:.6f}",
        f"- hardest_external_family_name: {hm['hardest_external_family_name']}",
        f"- hardest_external_path: {hm['hardest_external_path']}",
        f"- hardest_external_intensity: {hm['hardest_external_intensity']:.6f}",
        f"- weakest_external_receiver: {hm['weakest_external_receiver']}",
        f"- weakest_external_receiver_floor: {hm['weakest_external_receiver_floor']:.6f}",
        f"- mean_strongest_path_intensity: {hm['mean_strongest_path_intensity']:.6f}",
        f"- external_distribution_counterexample_score: {hm['external_distribution_counterexample_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_external_distribution_counterexample_pack_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
