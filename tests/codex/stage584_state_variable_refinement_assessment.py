#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE582_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage582_discourse_chain_substate_empirical_20260409"
    / "summary.json"
)
STAGE583_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage583_certainty_state_dynamics_empirical_20260409"
    / "summary.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage584_state_variable_refinement_assessment_20260409"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    stage582 = load_json(STAGE582_PATH)
    stage583 = load_json(STAGE583_PATH)

    recent_values = []
    long_values = []
    commitment_values = []
    guarantee_values = []
    opposite_values = []

    for row in stage582["model_rows"]:
        recent_values.append(float(row["experiment_rows"]["discourse_recent_subject"]["accuracy"]))
        long_values.append(float(row["experiment_rows"]["discourse_long_chain"]["accuracy"]))

    for row in stage583["model_rows"]:
        commitment_values.append(float(row["experiment_rows"]["certainty_commitment"]["accuracy"]))
        guarantee_values.append(float(row["experiment_rows"]["certainty_guarantee"]["accuracy"]))
        opposite_values.append(float(row["experiment_rows"]["certainty_opposite_compatibility"]["accuracy"]))

    discourse_recent_mean = mean(recent_values)
    discourse_long_mean = mean(long_values)
    discourse_gap = discourse_long_mean - discourse_recent_mean

    commitment_mean = mean(commitment_values)
    guarantee_mean = mean(guarantee_values)
    opposite_mean = mean(opposite_values)
    certainty_spread = max(commitment_mean, guarantee_mean, opposite_mean) - min(
        commitment_mean, guarantee_mean, opposite_mean
    )

    if discourse_gap <= -0.15:
        discourse_reading = "P_discourse 更像至少包含短链延续和长链回收两个子状态。"
        discourse_candidate = "P_discourse_recent,l + P_discourse_reactivation,l"
    else:
        discourse_reading = "当前样本下 P_discourse 暂未显示出足够强的内部分裂。"
        discourse_candidate = "P_discourse,l"

    if certainty_spread >= 0.20:
        certainty_reading = "Q_certainty 内部子任务差异明显，确定性判断可能还需继续拆分。"
        certainty_candidate = "Q_commitment,l + Q_entailment,l + Q_compatibility,l"
    else:
        certainty_reading = "当前样本下 Q_certainty 可以暂时视作较统一的判断状态。"
        certainty_candidate = "Q_certainty,l"

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage584_state_variable_refinement_assessment",
        "title": "状态变量细化评估",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "sources": {
            "stage582": str(STAGE582_PATH),
            "stage583": str(STAGE583_PATH),
        },
        "means": {
            "discourse_recent_mean": discourse_recent_mean,
            "discourse_long_mean": discourse_long_mean,
            "discourse_gap": discourse_gap,
            "commitment_mean": commitment_mean,
            "guarantee_mean": guarantee_mean,
            "opposite_mean": opposite_mean,
            "certainty_spread": certainty_spread,
        },
        "readings": {
            "discourse": discourse_reading,
            "certainty": certainty_reading,
        },
        "candidate_refinement": {
            "P_discourse": discourse_candidate,
            "Q_certainty": certainty_candidate,
        },
        "core_answer": (
            f"{discourse_reading}{certainty_reading}"
            " 这说明统一状态方程还需要继续压缩粗粒度状态变量。"
        ),
    }

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# stage584 状态变量细化评估",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
        "## 平均结果",
        f"- discourse_recent_mean: `{discourse_recent_mean:.4f}`",
        f"- discourse_long_mean: `{discourse_long_mean:.4f}`",
        f"- discourse_gap: `{discourse_gap:.4f}`",
        f"- commitment_mean: `{commitment_mean:.4f}`",
        f"- guarantee_mean: `{guarantee_mean:.4f}`",
        f"- opposite_mean: `{opposite_mean:.4f}`",
        f"- certainty_spread: `{certainty_spread:.4f}`",
        "",
        "## 候选细化",
        f"- P_discourse: `{discourse_candidate}`",
        f"- Q_certainty: `{certainty_candidate}`",
    ]
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(OUTPUT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
