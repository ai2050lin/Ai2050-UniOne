from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from stage102_real_world_falsification_bridge import build_real_world_falsification_bridge_summary
from stage110_axiom_falsification_suite import build_axiom_falsification_suite_summary
from stage111_native_variable_registry_pruning import build_native_variable_registry_pruning_summary


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage112_world_task_boundary_bridge_20260322"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


@lru_cache(maxsize=1)
def build_world_task_boundary_bridge_summary() -> dict:
    bridge = build_real_world_falsification_bridge_summary()["headline_metrics"]
    attacks = build_axiom_falsification_suite_summary()
    registry = build_native_variable_registry_pruning_summary()

    native_records = [record for record in registry["registry_records"] if record["registry_role"] == "native_core"]
    native_map = {record["name"]: record for record in native_records}

    family_specs = [
        {
            "name": "style_dialogue_family",
            "source_attack": "style_context_flip_attack",
            "native_targets": {
                "anchor_recurrence_family": 0.45,
                "minimal_transport_efficiency_quantity": 0.25,
            },
            "bridge_weight": 0.70,
        },
        {
            "name": "logic_negation_family",
            "source_attack": "logic_negation_attack",
            "native_targets": {
                "minimal_transport_efficiency_quantity": 0.70,
                "anchor_recurrence_family": 0.25,
            },
            "bridge_weight": 0.92,
        },
        {
            "name": "syntax_rewrite_family",
            "source_attack": "syntax_voice_shift_attack",
            "native_targets": {
                "anchor_recurrence_family": 0.30,
                "minimal_transport_efficiency_quantity": 0.35,
            },
            "bridge_weight": 0.62,
        },
        {
            "name": "bilingual_alias_family",
            "source_attack": "bilingual_alias_attack",
            "native_targets": {
                "anchor_recurrence_family": 0.40,
                "minimal_transport_efficiency_quantity": 0.30,
            },
            "bridge_weight": 0.78,
        },
        {
            "name": "macro_abstract_family",
            "source_attack": "macro_abstract_bridge_attack",
            "native_targets": {
                "minimal_transport_efficiency_quantity": 0.62,
                "anchor_recurrence_family": 0.22,
            },
            "bridge_weight": 0.96,
        },
    ]

    attack_map = {record["name"]: record for record in attacks["attack_records"]}
    task_family_records = []
    native_task_totals = {name: [] for name in native_map}

    for spec in family_specs:
        attack = attack_map[spec["source_attack"]]
        family_pressure = _clip01(
            0.42 * attack["intensity"]
            + 0.34 * bridge["remaining_real_world_gap"]
            + 0.24 * spec["bridge_weight"] * bridge["task_context_bridge_strength"]
        )
        native_impacts = []
        for native_name, weight in spec["native_targets"].items():
            native = native_map[native_name]
            post_task_support = _clip01(
                native["native_eligibility"] - family_pressure * weight * 0.34
            )
            native_impacts.append(
                {
                    "name": native_name,
                    "post_task_support": post_task_support,
                    "drop": _clip01(native["native_eligibility"] - post_task_support),
                }
            )
            native_task_totals[native_name].append(post_task_support)

        weakest_native = min(native_impacts, key=lambda item: item["post_task_support"])
        task_family_records.append(
            {
                "name": spec["name"],
                "source_attack": spec["source_attack"],
                "family_pressure": family_pressure,
                "weakest_native_name": weakest_native["name"],
                "weakest_native_post_task_support": weakest_native["post_task_support"],
                "native_impacts": native_impacts,
            }
        )

    hardest_family = max(task_family_records, key=lambda item: item["family_pressure"])
    weakest_native_under_task = min(
        (
            {
                "name": name,
                "mean_post_task_support": _mean(values),
            }
            for name, values in native_task_totals.items()
        ),
        key=lambda item: item["mean_post_task_support"],
    )

    bridge_family_coverage = _clip01(len(task_family_records) / 5.0)
    task_boundary_closure_gain = _clip01(
        0.34 * (1.0 - hardest_family["family_pressure"])
        + 0.26 * bridge["bridge_alignment_support"]
        + 0.20 * bridge["multiseed_probe_stability"]
        + 0.20 * (1.0 - attacks["headline_metrics"]["task_bridge_retest_pressure"])
    )
    world_task_boundary_bridge_score = _clip01(
        0.30 * bridge_family_coverage
        + 0.24 * task_boundary_closure_gain
        + 0.22 * weakest_native_under_task["mean_post_task_support"]
        + 0.24 * (1.0 - hardest_family["family_pressure"])
    )

    return {
        "headline_metrics": {
            "bridge_family_coverage": bridge_family_coverage,
            "hardest_family_name": hardest_family["name"],
            "hardest_family_pressure": hardest_family["family_pressure"],
            "weakest_native_under_task_name": weakest_native_under_task["name"],
            "weakest_native_under_task_score": weakest_native_under_task["mean_post_task_support"],
            "task_boundary_closure_gain": task_boundary_closure_gain,
            "world_task_boundary_bridge_score": world_task_boundary_bridge_score,
        },
        "task_family_records": task_family_records,
        "status": {
            "status_short": (
                "world_task_boundary_bridge_ready"
                if world_task_boundary_bridge_score >= 0.56 and weakest_native_under_task["mean_post_task_support"] >= 0.52
                else "world_task_boundary_bridge_transition"
            ),
            "status_label": "真实任务边界桥已经开始按任务家族复核主核变量，但当前最难家族的压力仍然偏高。",
        },
        "project_readout": {
            "summary": "这一轮把主核变量直接放到风格、逻辑、语法、双语、宏观抽象五类任务家族下面复核，不再只看总的任务桥分数。",
            "next_question": "下一步要继续检查最难任务家族为什么会优先打穿最弱主核变量，并考虑是否要重新定义边界主核。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage112 World Task Boundary Bridge",
        "",
        f"- bridge_family_coverage: {hm['bridge_family_coverage']:.6f}",
        f"- hardest_family_name: {hm['hardest_family_name']}",
        f"- hardest_family_pressure: {hm['hardest_family_pressure']:.6f}",
        f"- weakest_native_under_task_name: {hm['weakest_native_under_task_name']}",
        f"- weakest_native_under_task_score: {hm['weakest_native_under_task_score']:.6f}",
        f"- task_boundary_closure_gain: {hm['task_boundary_closure_gain']:.6f}",
        f"- world_task_boundary_bridge_score: {hm['world_task_boundary_bridge_score']:.6f}",
        f"- status_short: {summary['status']['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_world_task_boundary_bridge_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
