#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ollama_complete_suite_shared import chinese_ratio, english_ratio, extract_token, run_ollama_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage263_qwen_deepseek_complete_behavior_suite_20260324"
MODEL_SPECS = [
    {"model_tag": "qwen4b", "model_name": "qwen3:4b", "display_name": "Qwen3-4B"},
    {"model_tag": "deepseek14b", "model_name": "deepseek-r1:14b", "display_name": "DeepSeek-R1-14B"},
]

PROBES = [
    {
        "category": "translation",
        "probe_name": "中译英天气",
        "mode": "language",
        "expected_language": "english",
        "prompt": "请把中文翻译为英文，只输出翻译结果：今天天气不错",
    },
    {
        "category": "translation",
        "probe_name": "英译中天气",
        "mode": "language",
        "expected_language": "chinese",
        "prompt": "请把英文翻译为中文，只输出翻译结果：The weather is nice today.",
    },
    {
        "category": "sense",
        "probe_name": "苹果水果义",
        "mode": "token",
        "candidates": ["fruit", "brand"],
        "expected": "fruit",
        "prompt": "只输出 fruit 或 brand：句子“Tom ate the apple after washing it.”里的 apple 更像 fruit 还是 brand？",
    },
    {
        "category": "sense",
        "probe_name": "苹果品牌义",
        "mode": "token",
        "candidates": ["fruit", "brand"],
        "expected": "brand",
        "prompt": "只输出 fruit 或 brand：句子“Apple released a new iPhone and updated it.”里的 Apple 更像 fruit 还是 brand？",
    },
    {
        "category": "competition",
        "probe_name": "水果内部竞争",
        "mode": "token",
        "candidates": ["apple", "pear"],
        "expected": "apple",
        "prompt": "只输出 apple 或 pear：Tom washed the apple beside the pear, sliced the fruit, and ate it. 最后的 it 更可能指 apple 还是 pear？",
    },
    {
        "category": "competition",
        "probe_name": "动物内部竞争",
        "mode": "token",
        "candidates": ["cat", "dog"],
        "expected": "cat",
        "prompt": "只输出 cat 或 dog：The dog stood near the cat, the pet was washed, and it later slept. 最后的 it 更可能指 cat 还是 dog？",
    },
    {
        "category": "long_chain",
        "probe_name": "水果长链回收",
        "mode": "token",
        "candidates": ["yes", "no"],
        "expected": "yes",
        "prompt": "只输出 yes 或 no：在句子“Tom picked the apple, washed it, sliced it, and it became sweet”里，最后的 it 是否指 apple？",
    },
    {
        "category": "long_chain",
        "probe_name": "品牌长链回收",
        "mode": "token",
        "candidates": ["yes", "no"],
        "expected": "yes",
        "prompt": "只输出 yes 或 no：在句子“Apple designed the laptop, updated it, and it became popular”里，最后的 it 是否指 laptop？",
    },
    {
        "category": "task_semantics",
        "probe_name": "图像编辑语义",
        "mode": "token",
        "candidates": ["image_edit", "translate", "refactor"],
        "expected": "image_edit",
        "prompt": "只输出 image_edit、translate 或 refactor：命令“修改左边苹果颜色”属于哪种任务？",
    },
    {
        "category": "task_semantics",
        "probe_name": "代码重构语义",
        "mode": "token",
        "candidates": ["image_edit", "translate", "refactor"],
        "expected": "refactor",
        "prompt": "只输出 image_edit、translate 或 refactor：命令“重构 src/app.py 文件”属于哪种任务？",
    },
]


def score_probe(raw_output: str, probe: dict) -> tuple[float, str]:
    if probe["mode"] == "language":
        e_ratio = english_ratio(raw_output)
        c_ratio = chinese_ratio(raw_output)
        if probe["expected_language"] == "english":
            score = 1.0 if e_ratio > c_ratio and e_ratio > 0.10 else 0.0
            prediction = "english" if score > 0 else "other"
        else:
            score = 1.0 if c_ratio > e_ratio and c_ratio > 0.10 else 0.0
            prediction = "chinese" if score > 0 else "other"
        return score, prediction
    prediction = extract_token(raw_output, probe["candidates"])
    score = 1.0 if prediction == probe["expected"] else 0.0
    return score, prediction


def evaluate_model(model_spec: dict) -> dict:
    rows = []
    for probe in PROBES:
        raw_output = run_ollama_prompt(model_spec["model_name"], probe["prompt"])
        score, prediction = score_probe(raw_output, probe)
        rows.append(
            {
                "category": probe["category"],
                "probe_name": probe["probe_name"],
                "score": score,
                "prediction": prediction,
                "expected": probe.get("expected", probe.get("expected_language")),
                "raw_output_preview": raw_output[:160],
            }
        )
    category_rows = []
    for category in sorted({probe["category"] for probe in PROBES}):
        category_scores = [row["score"] for row in rows if row["category"] == category]
        category_rows.append({"category": category, "score": sum(category_scores) / len(category_scores)})
    direct_score = sum(row["score"] for row in rows) / len(rows)
    strongest = max(category_rows, key=lambda row: row["score"])
    weakest = min(category_rows, key=lambda row: row["score"])
    return {
        "model_tag": model_spec["model_tag"],
        "model_name": model_spec["model_name"],
        "display_name": model_spec["display_name"],
        "probe_count": len(rows),
        "direct_score": direct_score,
        "strongest_category": strongest["category"],
        "strongest_category_score": strongest["score"],
        "weakest_category": weakest["category"],
        "weakest_category_score": weakest["score"],
        "category_rows": category_rows,
        "probe_rows": rows,
    }


def build_summary() -> dict:
    model_rows = [evaluate_model(model_spec) for model_spec in MODEL_SPECS]
    strongest_model = max(model_rows, key=lambda row: row["direct_score"])
    weakest_model = min(model_rows, key=lambda row: row["direct_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage263_qwen_deepseek_complete_behavior_suite",
        "title": "Qwen 与 DeepSeek 完整行为测试套件",
        "status_short": "qwen_deepseek_complete_behavior_ready",
        "model_count": len(model_rows),
        "probe_count_per_model": len(PROBES),
        "strongest_model": strongest_model["display_name"],
        "weakest_model": weakest_model["display_name"],
        "model_rows": model_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage263：Qwen 与 DeepSeek 完整行为测试套件",
        "",
        f"- 模型数量：{summary['model_count']}",
        f"- 每模型探针数：{summary['probe_count_per_model']}",
        f"- 最强模型：{summary['strongest_model']}",
        f"- 最弱模型：{summary['weakest_model']}",
    ]
    for row in summary["model_rows"]:
        lines.extend(
            [
                "",
                f"## {row['display_name']}",
                f"- 直测总分：{row['direct_score']:.4f}",
                f"- 最强类别：{row['strongest_category']} ({row['strongest_category_score']:.4f})",
                f"- 最弱类别：{row['weakest_category']} ({row['weakest_category_score']:.4f})",
            ]
        )
    (output_dir / "STAGE263_QWEN_DEEPSEEK_COMPLETE_BEHAVIOR_SUITE_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8-sig"
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen 与 DeepSeek 完整行为测试套件")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

