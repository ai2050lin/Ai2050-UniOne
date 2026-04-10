#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE574_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage574_unified_language_state_update_empirical_20260409"
    / "summary.json"
)
STAGE576_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage576_reference_modifier_substate_empirical_20260409"
    / "summary.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage577_state_substate_upgrade_assessment_20260409"
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
    stage574 = load_json(STAGE574_PATH)
    stage576 = load_json(STAGE576_PATH)

    coarse_pronoun = []
    coarse_adverb = []
    split_pronoun = []
    split_adverb = []
    subtype_means: Dict[str, List[float]] = {
        "pronoun_personal": [],
        "pronoun_reflexive": [],
        "pronoun_demonstrative": [],
        "adverb_manner": [],
        "adverb_epistemic": [],
        "adverb_degree": [],
        "adverb_frequency": [],
    }

    by_key_574 = {row["model_key"]: row for row in stage574["model_rows"]}
    by_key_576 = {row["model_key"]: row for row in stage576["model_rows"]}

    comparison_rows = []
    for model_key in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        row574 = by_key_574[model_key]
        row576 = by_key_576[model_key]
        pronoun_old = float(row574["experiment_rows"]["pronoun_coreference"]["accuracy"])
        adverb_old = float(row574["experiment_rows"]["adverb_scope"]["accuracy"])
        pronoun_new = float(row576["pronoun_split_mean_accuracy"])
        adverb_new = float(row576["adverb_split_mean_accuracy"])
        coarse_pronoun.append(pronoun_old)
        coarse_adverb.append(adverb_old)
        split_pronoun.append(pronoun_new)
        split_adverb.append(adverb_new)
        for key in subtype_means:
            subtype_means[key].append(float(row576["experiment_rows"][key]["accuracy"]))
        comparison_rows.append(
            {
                "model_key": model_key,
                "model_label": row576["model_label"],
                "pronoun_old": pronoun_old,
                "pronoun_new": pronoun_new,
                "pronoun_gain": pronoun_new - pronoun_old,
                "adverb_old": adverb_old,
                "adverb_new": adverb_new,
                "adverb_gain": adverb_new - adverb_old,
            }
        )

    pronoun_gain_mean = mean(split_pronoun) - mean(coarse_pronoun)
    adverb_gain_mean = mean(split_adverb) - mean(coarse_adverb)

    upgraded_state_equation = {
        "old": "S_t,l = (O_t,l, A_t,l, R_t,l, P_t,l, M_t,l, Q_t,l, G_t,l, C_t,l)",
        "new": (
            "S_t,l = (O_t,l, A_t,l, R_t,l, P_personal,l, P_reflexive,l, P_demonstrative,l, "
            "M_manner,l, M_epistemic,l, M_degree,l, M_frequency,l, Q_t,l, G_t,l, C_t,l)"
        ),
    }

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage577_state_substate_upgrade_assessment",
        "title": "状态方程子状态升级评估",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "sources": {
            "stage574": str(STAGE574_PATH),
            "stage576": str(STAGE576_PATH),
        },
        "comparison_rows": comparison_rows,
        "coarse_vs_split": {
            "coarse_pronoun_mean": mean(coarse_pronoun),
            "split_pronoun_mean": mean(split_pronoun),
            "pronoun_gain_mean": pronoun_gain_mean,
            "coarse_adverb_mean": mean(coarse_adverb),
            "split_adverb_mean": mean(split_adverb),
            "adverb_gain_mean": adverb_gain_mean,
        },
        "subtype_means": {key: mean(values) for key, values in subtype_means.items()},
        "upgraded_state_equation": upgraded_state_equation,
        "core_answer": (
            "P_t 需要升级为多子状态，因为拆分后平均表现显著提升；"
            "M_t 也需要升级，但不是因为整体更强，而是因为拆分暴露出方式副词强、认识论副词弱、程度/频率高度模型依赖的结构分化。"
        ),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# stage577 状态方程子状态升级评估",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
        "## 粗粒度 vs 子状态",
        f"- coarse_pronoun_mean: `{summary['coarse_vs_split']['coarse_pronoun_mean']:.4f}`",
        f"- split_pronoun_mean: `{summary['coarse_vs_split']['split_pronoun_mean']:.4f}`",
        f"- pronoun_gain_mean: `{summary['coarse_vs_split']['pronoun_gain_mean']:.4f}`",
        f"- coarse_adverb_mean: `{summary['coarse_vs_split']['coarse_adverb_mean']:.4f}`",
        f"- split_adverb_mean: `{summary['coarse_vs_split']['split_adverb_mean']:.4f}`",
        f"- adverb_gain_mean: `{summary['coarse_vs_split']['adverb_gain_mean']:.4f}`",
        "",
        "## 子状态均值",
    ]
    for key, value in summary["subtype_means"].items():
        lines.append(f"- `{key}`: `{value:.4f}`")
    lines.extend([
        "",
        "## 升级后的状态方程",
        f"- old: `{upgraded_state_equation['old']}`",
        f"- new: `{upgraded_state_equation['new']}`",
    ])
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(OUTPUT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
