from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage75_compositional_binding_write_repair_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage70_direct_stability_counterexample_probe import build_direct_stability_counterexample_probe_summary
from stage72_language_projection_covariance import build_language_projection_covariance_summary
from stage74_learning_stability_failure_map import build_learning_stability_failure_map_summary


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


def build_compositional_binding_write_repair_summary() -> dict:
    failure_map = build_learning_stability_failure_map_summary()
    counter = build_direct_stability_counterexample_probe_summary()["headline_metrics"]
    projection = build_language_projection_covariance_summary()["headline_metrics"]

    worst = next(
        item for item in failure_map["scenario_records"] if item["name"] == "compositional_binding_write"
    )

    scenario = {
        "novelty_load": 0.78,
        "retention_conflict": 0.74,
        "context_shift": 0.61,
        "route_load": 0.76,
    }
    raw_drive = (
        0.34 * scenario["novelty_load"]
        + 0.24 * scenario["retention_conflict"]
        + 0.22 * scenario["context_shift"]
        + 0.20 * scenario["route_load"]
    )

    laws = {
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

    law_results = {}
    for law_name, params in laws.items():
        bounded_drive = _bounded_drive(raw_drive, law_name)
        repair_anchor = _clip01(
            0.42 * failure_map["headline_metrics"]["average_guarded_update_score"]
            + 0.28 * failure_map["headline_metrics"]["average_recovery_buffer"]
            + 0.18 * projection["language_projection_repair_score"]
            + 0.12 * (1.0 - counter["counterexample_pressure"])
        )

        forgetting_after = _clip01(
            worst["forgetting_risk"]
            - params["forgetting_gain"] * bounded_drive
            - 0.08 * repair_anchor
            + 0.04 * counter["counterexample_pressure"]
        )
        novelty_drop_after = _clip01(
            worst["novelty_drop_risk"]
            - params["novelty_gain"] * bounded_drive
            - 0.06 * repair_anchor
            + 0.03 * scenario["context_shift"]
        )
        perplexity_after = _clip01(
            worst["perplexity_stress"]
            - params["perplexity_gain"] * bounded_drive
            - 0.06 * repair_anchor
            + 0.03 * scenario["route_load"]
        )
        guarded_update_after = _clip01(
            worst["guarded_update_score"]
            + params["guard_gain"] * bounded_drive
            + 0.10 * repair_anchor
            - 0.05 * counter["counterexample_pressure"]
        )
        recovery_buffer_after = _clip01(
            worst["recovery_buffer"]
            + params["recovery_gain"] * bounded_drive
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

        repair_gain = worst["failure_intensity"] - failure_intensity_after
        stability_window_gain = _clip01(
            0.42 * (guarded_update_after - worst["guarded_update_score"] + 0.50)
            + 0.34 * (recovery_buffer_after - worst["recovery_buffer"] + 0.50)
            + 0.24 * (worst["failure_intensity"] - failure_intensity_after + 0.40)
        )
        repaired_learning_stability_score = _clip01(
            0.30 * guarded_update_after
            + 0.24 * recovery_buffer_after
            + 0.20 * (1.0 - failure_intensity_after)
            + 0.14 * stability_window_gain
            + 0.12 * params["interpretability"]
        )
        law_results[law_name] = {
            "bounded_drive": bounded_drive,
            "forgetting_after": forgetting_after,
            "novelty_drop_after": novelty_drop_after,
            "perplexity_after": perplexity_after,
            "guarded_update_after": guarded_update_after,
            "recovery_buffer_after": recovery_buffer_after,
            "failure_intensity_after": failure_intensity_after,
            "repair_gain": repair_gain,
            "stability_window_gain": stability_window_gain,
            "repaired_learning_stability_score": repaired_learning_stability_score,
            "interpretability": params["interpretability"],
        }

    best_law_name, best_law = max(
        law_results.items(),
        key=lambda item: (
            item[1]["repaired_learning_stability_score"],
            item[1]["repair_gain"],
            -item[1]["failure_intensity_after"],
        ),
    )

    return {
        "headline_metrics": {
            "worst_case_failure_name": worst["name"],
            "raw_drive": raw_drive,
            "best_law_name": best_law_name,
            "best_repair_gain": best_law["repair_gain"],
            "best_failure_intensity_after": best_law["failure_intensity_after"],
            "best_stability_window_gain": best_law["stability_window_gain"],
            "best_repaired_learning_stability_score": best_law["repaired_learning_stability_score"],
        },
        "worst_case_before": worst,
        "law_results": law_results,
        "status": {
            "status_short": (
                "compositional_binding_repair_ready"
                if best_law["failure_intensity_after"] <= 0.45 and best_law["repaired_learning_stability_score"] >= 0.72
                else "compositional_binding_repair_transition"
            ),
            "status_label": "最坏写入场景已经出现首个有界更新修复候选，但还需要继续在更真实压力下复核",
        },
        "project_readout": {
            "summary": "这一轮直接围绕 compositional_binding_write 比较三类有界更新律，寻找能够压低最坏失稳强度并扩大安全写入窗口的首个候选修复方案。",
            "next_question": "下一步要把 best_law 真正并回学习稳态图谱，检查它对 route_rebind 和 context_switch 两类次坏场景是否也同样有效。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage75 Compositional Binding Write Repair",
        "",
        f"- worst_case_failure_name: {hm['worst_case_failure_name']}",
        f"- raw_drive: {hm['raw_drive']:.6f}",
        f"- best_law_name: {hm['best_law_name']}",
        f"- best_repair_gain: {hm['best_repair_gain']:.6f}",
        f"- best_failure_intensity_after: {hm['best_failure_intensity_after']:.6f}",
        f"- best_stability_window_gain: {hm['best_stability_window_gain']:.6f}",
        f"- best_repaired_learning_stability_score: {hm['best_repaired_learning_stability_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_compositional_binding_write_repair_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
