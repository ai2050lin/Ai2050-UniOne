#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE576_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage576_reference_modifier_substate_empirical_20260409"
    / "summary.json"
)
STAGE578_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage578_personal_coreference_discourse_empirical_20260409"
    / "summary.json"
)
STAGE579_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage579_epistemic_uncertainty_empirical_20260409"
    / "summary.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage580_targeted_bottleneck_assessment_20260409"
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
    stage576 = load_json(STAGE576_PATH)
    stage578 = load_json(STAGE578_PATH)
    stage579 = load_json(STAGE579_PATH)

    by_key_576 = {row["model_key"]: row for row in stage576["model_rows"]}
    by_key_578 = {row["model_key"]: row for row in stage578["model_rows"]}
    by_key_579 = {row["model_key"]: row for row in stage579["model_rows"]}

    personal_simple = []
    personal_discourse = []
    epistemic_scope = []
    epistemic_uncertainty = []
    comparison_rows = []

    for model_key in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        row576 = by_key_576[model_key]
        row578 = by_key_578[model_key]
        row579 = by_key_579[model_key]

        simple_personal = float(row576["experiment_rows"]["pronoun_personal"]["accuracy"])
        discourse_personal = float(row578["experiment_rows"]["pronoun_personal_discourse"]["accuracy"])
        simple_epistemic = float(row576["experiment_rows"]["adverb_epistemic"]["accuracy"])
        targeted_epistemic = float(row579["epistemic_mean_accuracy"])

        personal_simple.append(simple_personal)
        personal_discourse.append(discourse_personal)
        epistemic_scope.append(simple_epistemic)
        epistemic_uncertainty.append(targeted_epistemic)

        comparison_rows.append(
            {
                "model_key": model_key,
                "model_label": row576["model_label"],
                "simple_personal_accuracy": simple_personal,
                "discourse_personal_accuracy": discourse_personal,
                "personal_gap": discourse_personal - simple_personal,
                "simple_epistemic_accuracy": simple_epistemic,
                "targeted_epistemic_accuracy": targeted_epistemic,
                "epistemic_gap": targeted_epistemic - simple_epistemic,
            }
        )

    personal_gap_mean = mean(personal_discourse) - mean(personal_simple)
    epistemic_gap_mean = mean(epistemic_uncertainty) - mean(epistemic_scope)

    if personal_gap_mean < -0.10:
        personal_reading = "P_personal 必须拆成句内局部绑定和跨句链路两层状态。"
    elif personal_gap_mean > 0.10:
        personal_reading = "篇章级共指没有比简单共指更难，P_personal 可以保持统一，但仍需验证更长链路。"
    else:
        personal_reading = "篇章级共指与简单共指接近，P_personal 可能是统一状态，也可能受样本规模限制。"

    if epistemic_gap_mean > 0.10:
        epistemic_reading = "M_epistemic 更像和 Q_certainty 强耦合的判断状态，而不只是修饰范围。"
    elif epistemic_gap_mean < -0.10:
        epistemic_reading = "M_epistemic 在真实不确定性判断里更弱，当前理论明显高估了它。"
    else:
        epistemic_reading = "M_epistemic 既不是纯修饰，也还没有被完全吸收到 Q_t，需要独立子状态。"

    upgraded_state_equation = {
        "previous": (
            "S_t,l = (O_t,l, A_t,l, R_t,l, P_personal,l, P_reflexive,l, P_demonstrative,l, "
            "M_manner,l, M_epistemic,l, M_degree,l, M_frequency,l, Q_t,l, G_t,l, C_t,l)"
        ),
        "candidate_next": (
            "S_t,l = (O_t,l, A_t,l, R_t,l, P_local,l, P_discourse,l, P_reflexive,l, "
            "P_demonstrative,l, M_manner,l, M_epistemic_scope,l, M_degree,l, M_frequency,l, "
            "Q_certainty,l, Q_reasoning,l, G_t,l, C_t,l)"
        ),
    }

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage580_targeted_bottleneck_assessment",
        "title": "定向瓶颈评估与状态方程升级",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "sources": {
            "stage576": str(STAGE576_PATH),
            "stage578": str(STAGE578_PATH),
            "stage579": str(STAGE579_PATH),
        },
        "comparison_rows": comparison_rows,
        "means": {
            "simple_personal_mean": mean(personal_simple),
            "discourse_personal_mean": mean(personal_discourse),
            "personal_gap_mean": personal_gap_mean,
            "simple_epistemic_mean": mean(epistemic_scope),
            "targeted_epistemic_mean": mean(epistemic_uncertainty),
            "epistemic_gap_mean": epistemic_gap_mean,
        },
        "readings": {
            "personal": personal_reading,
            "epistemic": epistemic_reading,
        },
        "upgraded_state_equation": upgraded_state_equation,
        "core_answer": (
            f"{personal_reading}{epistemic_reading}"
            " 这说明统一语言理论要继续压缩大兜底项，把跨句链路和确定性判断从粗粒度状态里拆出来。"
        ),
    }

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# stage580 定向瓶颈评估与状态方程升级",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
        "## 平均值",
        f"- simple_personal_mean: `{summary['means']['simple_personal_mean']:.4f}`",
        f"- discourse_personal_mean: `{summary['means']['discourse_personal_mean']:.4f}`",
        f"- personal_gap_mean: `{summary['means']['personal_gap_mean']:.4f}`",
        f"- simple_epistemic_mean: `{summary['means']['simple_epistemic_mean']:.4f}`",
        f"- targeted_epistemic_mean: `{summary['means']['targeted_epistemic_mean']:.4f}`",
        f"- epistemic_gap_mean: `{summary['means']['epistemic_gap_mean']:.4f}`",
        "",
        "## 理论升级",
        f"- previous: `{upgraded_state_equation['previous']}`",
        f"- candidate_next: `{upgraded_state_equation['candidate_next']}`",
    ]
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(OUTPUT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
