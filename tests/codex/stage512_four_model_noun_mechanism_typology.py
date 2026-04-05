#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage512_four_model_noun_mechanism_typology_20260404"
)

SOURCE_SUMMARIES = {
    "qwen3": PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage447_polysemy_family_switch_protocol_20260403"
    / "qwen3"
    / "summary.json",
    "deepseek7b": PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage447_polysemy_family_switch_protocol_20260403"
    / "deepseek7b_cpu"
    / "summary.json",
    "glm4": PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage511_glm4_polysemy_switch_protocol_20260404"
    / "summary.json",
    "gemma4": PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage509_gemma4_polysemy_switch_protocol_20260404"
    / "summary.json",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_model_row(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data["model_results"][0]


def classify_model(aggregate: Dict[str, object]) -> str:
    gap = float(aggregate["mean_ordinary_jaccard"]) - float(aggregate["mean_polysemy_jaccard"])
    causal_margin = float(aggregate["mean_switch_prob_drop"]) - float(aggregate["mean_control_prob_drop"])
    split_rate = float(aggregate["polysemy_split_support_count"]) / max(1, int(aggregate["noun_count"]))
    causal_rate = float(aggregate["switch_causality_support_count"]) / max(1, int(aggregate["noun_count"]))
    if gap <= 0:
        return "默认义偏置型弱切换"
    if split_rate >= 0.75 and causal_rate >= 0.5:
        return "低重合强切换型"
    if split_rate >= 0.75 and causal_margin > 0:
        return "低重合弱因果型"
    if split_rate >= 0.75:
        return "结构分裂型"
    return "过渡型"


def build_row(model_row: Dict[str, object]) -> Dict[str, object]:
    agg = model_row["aggregate"]
    gap = float(agg["mean_ordinary_jaccard"]) - float(agg["mean_polysemy_jaccard"])
    causal_margin = float(agg["mean_switch_prob_drop"]) - float(agg["mean_control_prob_drop"])
    return {
        "model_key": model_row["model_key"],
        "model_name": model_row["model_name"],
        "noun_count": int(agg["noun_count"]),
        "mean_polysemy_jaccard": float(agg["mean_polysemy_jaccard"]),
        "mean_ordinary_jaccard": float(agg["mean_ordinary_jaccard"]),
        "ordinary_vs_polysemy_gap": gap,
        "mean_switch_prob_drop": float(agg["mean_switch_prob_drop"]),
        "mean_control_prob_drop": float(agg["mean_control_prob_drop"]),
        "causal_margin": causal_margin,
        "polysemy_split_support_rate": float(agg["polysemy_split_support_count"]) / max(1, int(agg["noun_count"])),
        "switch_causality_support_rate": float(agg["switch_causality_support_count"]) / max(1, int(agg["noun_count"])),
        "typology": classify_model(agg),
        "best_switch_layers": agg.get("best_switch_layers", {}),
    }


def build_report(rows: List[Dict[str, object]], summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['model_name']}",
                f"- 分型：`{row['typology']}`",
                f"- 平均多义词重合：`{row['mean_polysemy_jaccard']:.4f}`",
                f"- 平均普通上下文重合：`{row['mean_ordinary_jaccard']:.4f}`",
                f"- 结构差值：`{row['ordinary_vs_polysemy_gap']:.4f}`",
                f"- 平均切换轴下降：`{row['mean_switch_prob_drop']:.4f}`",
                f"- 平均控制轴下降：`{row['mean_control_prob_drop']:.4f}`",
                f"- 因果边际：`{row['causal_margin']:.4f}`",
                f"- 低重合支持率：`{row['polysemy_split_support_rate']:.4f}`",
                f"- 切换轴因果支持率：`{row['switch_causality_support_rate']:.4f}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    rows = [build_row(load_model_row(path)) for path in SOURCE_SUMMARIES.values()]
    rows = sorted(rows, key=lambda row: row["ordinary_vs_polysemy_gap"], reverse=True)
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage512_four_model_noun_mechanism_typology",
        "title": "四模型名词机制分型",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "model_rows": rows,
        "core_answer": (
            "四模型名词机制已经出现清晰分型：Qwen3 与 DeepSeek7B 更接近低重合结构分裂型，"
            "GLM4 更像结构分裂但因果控制杆隐蔽型，Gemma4 更像默认义偏置型弱切换。"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "REPORT.md").write_text(build_report(rows, summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
