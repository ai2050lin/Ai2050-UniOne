from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage91_counterexample_attack_suite_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage84_falsifiable_computation_core import build_falsifiable_computation_core_summary
from stage88_external_counterexample_expansion import build_external_counterexample_expansion_summary
from stage89_law_margin_separation import build_law_margin_separation_summary
from stage90_independent_observation_planes import build_independent_observation_planes_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def build_counterexample_attack_suite_summary() -> dict:
    planes = build_independent_observation_planes_summary()
    falsifiable = build_falsifiable_computation_core_summary()["headline_metrics"]
    external = build_external_counterexample_expansion_summary()["headline_metrics"]
    law_margin = build_law_margin_separation_summary()["headline_metrics"]

    plane_signal = {
        record["name"]: record["signal_strength"]
        for record in planes["plane_records"]
    }
    hm = planes["headline_metrics"]
    thresholds = {
        "language_plane": 0.78,
        "brain_plane": 0.75,
        "intelligence_plane": 0.76,
        "falsification_plane": 0.74,
    }

    coupling_spillover = 0.18 * hm["variable_coupling_overlap"]
    split_resilience = 0.05 * hm["surface_anchor_independence"]

    attacks = [
        {
            "name": "parameter_perturbation",
            "severity": 0.72,
            "damage": {
                "language_plane": 0.18,
                "brain_plane": 0.22,
                "intelligence_plane": 0.20,
                "falsification_plane": 0.16,
            },
        },
        {
            "name": "order_shuffle_attack",
            "severity": 0.76,
            "damage": {
                "language_plane": 0.26,
                "brain_plane": 0.08,
                "intelligence_plane": 0.22,
                "falsification_plane": 0.14,
            },
        },
        {
            "name": "scenario_replacement_attack",
            "severity": 0.74,
            "damage": {
                "language_plane": 0.12,
                "brain_plane": 0.10,
                "intelligence_plane": 0.28,
                "falsification_plane": 0.18,
            },
        },
        {
            "name": "boundary_overload_attack",
            "severity": 0.82,
            "damage": {
                "language_plane": 0.10,
                "brain_plane": 0.20,
                "intelligence_plane": 0.16,
                "falsification_plane": 0.30,
            },
        },
        {
            "name": "cross_plane_coupling_resonance",
            "severity": 0.78,
            "damage": {
                "language_plane": 0.14,
                "brain_plane": 0.20,
                "intelligence_plane": 0.22,
                "falsification_plane": 0.20,
            },
            "extra_spillover": 0.10 + coupling_spillover,
        },
        {
            "name": "brain_grounding_shock",
            "severity": 0.84,
            "damage": {
                "language_plane": 0.10,
                "brain_plane": 0.34,
                "intelligence_plane": 0.18,
                "falsification_plane": 0.16,
            },
            "extra_spillover": 0.06 + 0.10 * (1.0 - falsifiable["shared_state_refutation_power"]),
        },
    ]

    attack_records = []
    multi_plane_breach_count = 0
    weakest_plane_name = None
    weakest_plane_floor = 1.0
    for attack in attacks:
        plane_after = {}
        breached_planes = []
        extra_spillover = attack.get("extra_spillover", coupling_spillover)
        for plane_name, base_signal in plane_signal.items():
            spillover = extra_spillover if plane_name != "language_plane" else 0.06 * hm["variable_coupling_overlap"]
            if plane_name == "intelligence_plane":
                spillover += 0.04 * (1.0 - law_margin["minimum_pairwise_margin"])
            if plane_name == "falsification_plane":
                spillover += 0.05 * external["expanded_trigger_rate"]
            attacked_signal = _clip01(
                base_signal
                - attack["severity"] * attack["damage"][plane_name]
                - spillover
                + split_resilience
            )
            plane_after[plane_name] = attacked_signal
            if attacked_signal < thresholds[plane_name]:
                breached_planes.append(plane_name)
            if attacked_signal < weakest_plane_floor:
                weakest_plane_floor = attacked_signal
                weakest_plane_name = plane_name

        if len(breached_planes) >= 2:
            multi_plane_breach_count += 1

        attack_intensity = _clip01(
            0.34 * attack["severity"]
            + 0.24 * (len(breached_planes) / len(plane_signal))
            + 0.22 * (1.0 - min(plane_after.values()))
            + 0.20 * extra_spillover
        )
        attack_records.append(
            {
                "name": attack["name"],
                "severity": attack["severity"],
                "breached_planes": breached_planes,
                "breach_count": len(breached_planes),
                "plane_after": plane_after,
                "attack_intensity": attack_intensity,
            }
        )

    hardest_attack = max(attack_records, key=lambda item: item["attack_intensity"])
    multi_plane_breach_rate = multi_plane_breach_count / len(attack_records)
    attack_suite_coverage = 1.0
    system_attack_survival_score = _clip01(
        sum(min(record["plane_after"].values()) for record in attack_records) / len(attack_records)
    )
    counterexample_attack_suite_score = _clip01(
        0.24 * attack_suite_coverage
        + 0.24 * multi_plane_breach_rate
        + 0.20 * hardest_attack["attack_intensity"]
        + 0.18 * (1.0 - system_attack_survival_score)
        + 0.14 * (1.0 - weakest_plane_floor)
    )

    return {
        "headline_metrics": {
            "attack_suite_coverage": attack_suite_coverage,
            "multi_plane_breach_rate": multi_plane_breach_rate,
            "hardest_attack_name": hardest_attack["name"],
            "hardest_attack_intensity": hardest_attack["attack_intensity"],
            "weakest_plane_name": weakest_plane_name,
            "weakest_plane_attack_floor": weakest_plane_floor,
            "system_attack_survival_score": system_attack_survival_score,
            "counterexample_attack_suite_score": counterexample_attack_suite_score,
        },
        "attack_records": attack_records,
        "plane_bridge": planes["headline_metrics"],
        "status": {
            "status_short": (
                "counterexample_attack_suite_ready"
                if multi_plane_breach_rate >= 0.50
                and hardest_attack["attack_intensity"] >= 0.58
                and weakest_plane_name is not None
                else "counterexample_attack_suite_transition"
            ),
            "status_label": "强攻击测试包已经能在多类攻击下同时压低多个观测面，但它证明的是脆弱性暴露能力，不是理论已经稳固。",
        },
        "project_readout": {
            "summary": "这一轮把参数扰动、顺序打乱、场景替换、边界过载、跨面耦合共振和脑编码冲击统一成一个强攻击测试包。",
            "next_question": "下一步要把脑编码冲击单独展开成反例包，判断 brain_grounding 这条最弱轴到底会如何击穿统一主核。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage91 Counterexample Attack Suite",
        "",
        f"- attack_suite_coverage: {hm['attack_suite_coverage']:.6f}",
        f"- multi_plane_breach_rate: {hm['multi_plane_breach_rate']:.6f}",
        f"- hardest_attack_name: {hm['hardest_attack_name']}",
        f"- hardest_attack_intensity: {hm['hardest_attack_intensity']:.6f}",
        f"- weakest_plane_name: {hm['weakest_plane_name']}",
        f"- weakest_plane_attack_floor: {hm['weakest_plane_attack_floor']:.6f}",
        f"- system_attack_survival_score: {hm['system_attack_survival_score']:.6f}",
        f"- counterexample_attack_suite_score: {hm['counterexample_attack_suite_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_counterexample_attack_suite_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
