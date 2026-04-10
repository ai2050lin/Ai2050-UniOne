#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE587_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage587_measurement_upgrade_assessment_20260409" / "summary.json"
STAGE588_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage588_large_sample_discourse_unified_empirical_20260410" / "summary.json"
STAGE589_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage589_certainty_dynamics_candidate_empirical_20260410" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage590_frontier_state_dynamics_assessment_20260410"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    s587 = load_json(STAGE587_PATH)
    s588 = load_json(STAGE588_PATH)
    s589 = load_json(STAGE589_PATH)

    discourse_large_mean = mean([float(row["large_sample_discourse_mean_accuracy"]) for row in s588["model_rows"] if "error" not in row])
    discourse_large_spread = mean([float(row["subtask_spread"]) for row in s588["model_rows"] if "error" not in row])
    discourse_prev_natural = float(s587["means"]["discourse_natural_mean"])
    discourse_delta = discourse_large_mean - discourse_prev_natural

    certainty_early = mean([float(row["dynamics_means"]["early_mean"]) for row in s589["model_rows"] if "error" not in row])
    certainty_mid = mean([float(row["dynamics_means"]["mid_mean"]) for row in s589["model_rows"] if "error" not in row])
    certainty_late = mean([float(row["dynamics_means"]["late_mean"]) for row in s589["model_rows"] if "error" not in row])
    certainty_gain = mean([float(row["dynamics_means"]["late_gain_mean"]) for row in s589["model_rows"] if "error" not in row])
    certainty_mono = mean([float(row["dynamics_means"]["monotonic_gain_ratio_mean"]) for row in s589["model_rows"] if "error" not in row])
    certainty_acc = mean([float(row["dynamics_means"]["accuracy_mean"]) for row in s589["model_rows"] if "error" not in row])

    if discourse_large_spread >= 0.18:
        discourse_reading = "大样本下 P_discourse 仍显著分裂，暂时不能进入稳定状态变量集。"
    elif discourse_large_mean < 0.60:
        discourse_reading = "大样本下 P_discourse 仍偏弱，但没有继续裂解，更像弱而统一的瓶颈状态。"
    else:
        discourse_reading = "大样本下 P_discourse 已经达到中等稳定，暂时可以保留为统一瓶颈状态。"

    if certainty_late > certainty_early and certainty_gain > 0.05 and certainty_mono >= 0.55:
        certainty_reading = "Q_certainty 已出现较清楚的后层成形趋势，可以视作候选动力学状态。"
    elif certainty_late > certainty_early:
        certainty_reading = "Q_certainty 有后层增强，但动力学还偏弱，尚未达到闭式更新律。"
    else:
        certainty_reading = "Q_certainty 还没有稳定后层成形轨迹，当前更像行为层变量。"

    frontier_score = mean([
        min(max(discourse_large_mean, 0.0), 1.0),
        min(max(1.0 - discourse_large_spread, 0.0), 1.0),
        min(max(certainty_acc, 0.0), 1.0),
        min(max((certainty_mono + 0.5 * (1.0 if certainty_gain > 0 else 0.0)), 0.0), 1.0),
    ])

    if frontier_score >= 0.72:
        verdict = "当前理论已经从“状态命名阶段”进入“弱动力学候选阶段”。"
    elif frontier_score >= 0.60:
        verdict = "当前理论仍处在“稳定状态变量筛选阶段”，还没有真正进入动力学闭式阶段。"
    else:
        verdict = "当前理论距离稳定状态变量与动力学闭环仍较远。"

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage590_frontier_state_dynamics_assessment",
        "title": "前沿状态变量与动力学评估",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "sources": {
            "stage587": str(STAGE587_PATH),
            "stage588": str(STAGE588_PATH),
            "stage589": str(STAGE589_PATH),
        },
        "metrics": {
            "discourse_prev_natural_mean": discourse_prev_natural,
            "discourse_large_mean": discourse_large_mean,
            "discourse_large_spread": discourse_large_spread,
            "discourse_delta": discourse_delta,
            "certainty_accuracy_mean": certainty_acc,
            "certainty_early_mean": certainty_early,
            "certainty_mid_mean": certainty_mid,
            "certainty_late_mean": certainty_late,
            "certainty_late_gain_mean": certainty_gain,
            "certainty_monotonic_ratio_mean": certainty_mono,
            "frontier_score": frontier_score,
        },
        "readings": {
            "discourse": discourse_reading,
            "certainty": certainty_reading,
            "verdict": verdict,
        },
        "core_answer": f"{discourse_reading}{certainty_reading}{verdict}",
    }

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# stage590 前沿状态变量与动力学评估",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
        "## 指标",
        f"- discourse_prev_natural_mean: `{discourse_prev_natural:.4f}`",
        f"- discourse_large_mean: `{discourse_large_mean:.4f}`",
        f"- discourse_large_spread: `{discourse_large_spread:.4f}`",
        f"- discourse_delta: `{discourse_delta:.4f}`",
        f"- certainty_accuracy_mean: `{certainty_acc:.4f}`",
        f"- certainty_early_mean: `{certainty_early:.4f}`",
        f"- certainty_mid_mean: `{certainty_mid:.4f}`",
        f"- certainty_late_mean: `{certainty_late:.4f}`",
        f"- certainty_late_gain_mean: `{certainty_gain:.4f}`",
        f"- certainty_monotonic_ratio_mean: `{certainty_mono:.4f}`",
        f"- frontier_score: `{frontier_score:.4f}`",
    ]
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(OUTPUT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
