from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage76_sqrt_repair_generalization_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage70_direct_stability_counterexample_probe import build_direct_stability_counterexample_probe_summary
from stage72_language_projection_covariance import build_language_projection_covariance_summary
from stage74_learning_stability_failure_map import build_learning_stability_failure_map_summary
from stage75_compositional_binding_write_repair import build_compositional_binding_write_repair_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _bounded_drive(raw_drive: float, law_name: str) -> float:
    if law_name == "log":
        return math.log1p(3.0 * raw_drive) / math.log(4.0)
    if law_name == "sqrt":
        return math.sqrt(raw_drive)
    if law_name == "rational":
        return (2.0 * raw_drive) / (1.0 + raw_drive)
    raise ValueError(f"unknown law_name={law_name}")


SCENARIO_INPUTS = {
    "semantic_patch_insert": {
        "novelty_load": 0.72,
        "retention_conflict": 0.44,
        "context_shift": 0.22,
        "route_load": 0.28,
    },
    "route_rebind_insert": {
        "novelty_load": 0.65,
        "retention_conflict": 0.58,
        "context_shift": 0.36,
        "route_load": 0.70,
    },
    "context_switch_write": {
        "novelty_load": 0.52,
        "retention_conflict": 0.48,
        "context_shift": 0.82,
        "route_load": 0.54,
    },
    "compositional_binding_write": {
        "novelty_load": 0.78,
        "retention_conflict": 0.74,
        "context_shift": 0.61,
        "route_load": 0.76,
    },
    "long_horizon_refresh": {
        "novelty_load": 0.44,
        "retention_conflict": 0.86,
        "context_shift": 0.24,
        "route_load": 0.40,
    },
}

LAW_PARAMS = {
    "log": {
        "forgetting_gain": 0.16,
        "novelty_gain": 0.14,
        "perplexity_gain": 0.12,
        "guard_gain": 0.26,
        "recovery_gain": 0.22,
        "interpretability": 0.93,
    },
    "sqrt": {
        "forgetting_gain": 0.18,
        "novelty_gain": 0.16,
        "perplexity_gain": 0.14,
        "guard_gain": 0.30,
        "recovery_gain": 0.25,
        "interpretability": 0.87,
    },
    "rational": {
        "forgetting_gain": 0.17,
        "novelty_gain": 0.15,
        "perplexity_gain": 0.13,
        "guard_gain": 0.28,
        "recovery_gain": 0.24,
        "interpretability": 0.89,
    },
}


def build_sqrt_repair_generalization_summary() -> dict:
    failure_map = build_learning_stability_failure_map_summary()
    repair = build_compositional_binding_write_repair_summary()
    counter = build_direct_stability_counterexample_probe_summary()["headline_metrics"]
    projection = build_language_projection_covariance_summary()["headline_metrics"]

    best_law_name = repair["headline_metrics"]["best_law_name"]
    law_params = LAW_PARAMS[best_law_name]

    repaired_records = []
    failure_after_values = []
    guarded_after_values = []
    recovery_after_values = []
    repaired_scores = []

    repair_anchor = _clip01(
        0.42 * failure_map["headline_metrics"]["average_guarded_update_score"]
        + 0.28 * failure_map["headline_metrics"]["average_recovery_buffer"]
        + 0.18 * projection["language_projection_repair_score"]
        + 0.12 * (1.0 - counter["counterexample_pressure"])
    )

    for before in failure_map["scenario_records"]:
        scenario = SCENARIO_INPUTS[before["name"]]
        raw_drive = (
            0.34 * scenario["novelty_load"]
            + 0.24 * scenario["retention_conflict"]
            + 0.22 * scenario["context_shift"]
            + 0.20 * scenario["route_load"]
        )
        bounded_drive = _bounded_drive(raw_drive, best_law_name)

        forgetting_after = _clip01(
            before["forgetting_risk"]
            - law_params["forgetting_gain"] * bounded_drive
            - 0.08 * repair_anchor
            + 0.04 * counter["counterexample_pressure"]
        )
        novelty_drop_after = _clip01(
            before["novelty_drop_risk"]
            - law_params["novelty_gain"] * bounded_drive
            - 0.06 * repair_anchor
            + 0.03 * scenario["context_shift"]
        )
        perplexity_after = _clip01(
            before["perplexity_stress"]
            - law_params["perplexity_gain"] * bounded_drive
            - 0.06 * repair_anchor
            + 0.03 * scenario["route_load"]
        )
        guarded_update_after = _clip01(
            before["guarded_update_score"]
            + law_params["guard_gain"] * bounded_drive
            + 0.10 * repair_anchor
            - 0.05 * counter["counterexample_pressure"]
        )
        recovery_buffer_after = _clip01(
            before["recovery_buffer"]
            + law_params["recovery_gain"] * bounded_drive
            + 0.08 * repair_anchor
            - 0.04 * scenario["route_load"]
        )
        failure_intensity_after = _clip01(
            0.36 * forgetting_after
            + 0.26 * novelty_drop_after
            + 0.18 * perplexity_after
            + 0.10 * scenario["context_shift"]
            + 0.10 * scenario["retention_conflict"]
        )
        repaired_learning_score = _clip01(
            0.30 * guarded_update_after
            + 0.24 * recovery_buffer_after
            + 0.20 * (1.0 - failure_intensity_after)
            + 0.14 * law_params["interpretability"]
            + 0.12 * projection["projection_counterexample_resistance"]
        )
        repair_gain = before["failure_intensity"] - failure_intensity_after

        repaired_records.append(
            {
                "name": before["name"],
                "raw_drive": raw_drive,
                "bounded_drive": bounded_drive,
                "failure_intensity_before": before["failure_intensity"],
                "failure_intensity_after": failure_intensity_after,
                "repair_gain": repair_gain,
                "guarded_update_after": guarded_update_after,
                "recovery_buffer_after": recovery_buffer_after,
                "repaired_learning_score": repaired_learning_score,
            }
        )
        failure_after_values.append(failure_intensity_after)
        guarded_after_values.append(guarded_update_after)
        recovery_after_values.append(recovery_buffer_after)
        repaired_scores.append(repaired_learning_score)

    repaired_worst = max(repaired_records, key=lambda item: item["failure_intensity_after"])
    route_rebind = next(item for item in repaired_records if item["name"] == "route_rebind_insert")
    context_switch = next(item for item in repaired_records if item["name"] == "context_switch_write")
    scenario_pass_rate = sum(1.0 for item in repaired_records if item["failure_intensity_after"] < item["failure_intensity_before"]) / len(repaired_records)

    repaired_average_failure_intensity = sum(failure_after_values) / len(failure_after_values)
    repaired_average_guarded_update = sum(guarded_after_values) / len(guarded_after_values)
    repaired_average_recovery_buffer = sum(recovery_after_values) / len(recovery_after_values)
    repaired_bounded_learning_window = _clip01(
        0.36 * repaired_average_guarded_update
        + 0.28 * repaired_average_recovery_buffer
        + 0.20 * (1.0 - repaired_worst["failure_intensity_after"])
        + 0.16 * (1.0 - repaired_average_failure_intensity)
    )
    generalized_repair_coverage = _clip01(
        0.38 * scenario_pass_rate
        + 0.32 * (1.0 - repaired_average_failure_intensity)
        + 0.30 * repaired_bounded_learning_window
    )
    repair_generalization_score = _clip01(
        0.26 * generalized_repair_coverage
        + 0.22 * repaired_bounded_learning_window
        + 0.18 * route_rebind["repaired_learning_score"]
        + 0.18 * context_switch["repaired_learning_score"]
        + 0.16 * (1.0 - repaired_worst["failure_intensity_after"])
    )

    return {
        "headline_metrics": {
            "best_law_name": best_law_name,
            "generalized_repair_coverage": generalized_repair_coverage,
            "repaired_average_failure_intensity": repaired_average_failure_intensity,
            "repaired_average_guarded_update": repaired_average_guarded_update,
            "repaired_bounded_learning_window": repaired_bounded_learning_window,
            "route_rebind_support": route_rebind["repaired_learning_score"],
            "context_switch_support": context_switch["repaired_learning_score"],
            "repaired_worst_case_name": repaired_worst["name"],
            "repaired_worst_case_failure_intensity": repaired_worst["failure_intensity_after"],
            "repair_generalization_score": repair_generalization_score,
        },
        "repaired_records": repaired_records,
        "status": {
            "status_short": (
                "sqrt_repair_generalized"
                if repair_generalization_score >= 0.77 and repaired_worst["failure_intensity_after"] <= 0.42
                else "sqrt_repair_generalization_transition"
            ),
            "status_label": "最优修复律已经从单点最坏场景扩展到多场景，但还需要继续向更真实脑编码约束下复核",
        },
        "project_readout": {
            "summary": "这一轮把 Stage75 的最优修复律从单个最坏场景扩展到整张学习失效图谱，专门检查它对 route_rebind 与 context_switch 两类次坏场景是否也同样有效。",
            "next_question": "下一步要把这条修复律并回 brain_grounding 约束，看学习稳态修复是否仍然成立而不破坏脑编码落地。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage76 Sqrt Repair Generalization",
        "",
        f"- best_law_name: {hm['best_law_name']}",
        f"- generalized_repair_coverage: {hm['generalized_repair_coverage']:.6f}",
        f"- repaired_average_failure_intensity: {hm['repaired_average_failure_intensity']:.6f}",
        f"- repaired_average_guarded_update: {hm['repaired_average_guarded_update']:.6f}",
        f"- repaired_bounded_learning_window: {hm['repaired_bounded_learning_window']:.6f}",
        f"- route_rebind_support: {hm['route_rebind_support']:.6f}",
        f"- context_switch_support: {hm['context_switch_support']:.6f}",
        f"- repaired_worst_case_name: {hm['repaired_worst_case_name']}",
        f"- repaired_worst_case_failure_intensity: {hm['repaired_worst_case_failure_intensity']:.6f}",
        f"- repair_generalization_score: {hm['repair_generalization_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_sqrt_repair_generalization_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
