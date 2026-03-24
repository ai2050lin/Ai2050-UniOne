#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage256_deepseek14b_multidirection_translation_probe_20260324"
MODEL_NAME = "deepseek-r1:14b"
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

PROMPTS = [
    {"name": "zh_to_en_weather", "prompt": "请把中文翻译为英文，只输出结果：今天天气不错", "target": "english"},
    {"name": "zh_to_en_apple", "prompt": "把下面中文译成英文，只输出结果：苹果很甜", "target": "english"},
    {"name": "en_to_zh_weather", "prompt": "请把英文翻译为中文，只输出结果：The weather is nice today.", "target": "chinese"},
    {"name": "en_to_zh_apple", "prompt": "把下面英文译成中文，只输出结果：The apple is sweet.", "target": "chinese"},
]


def clean_output(text: str) -> str:
    text = ANSI_RE.sub("", text)
    return text.replace("\r", "\n").strip()


def run_prompt(prompt: str) -> str:
    completed = subprocess.run(
        ["ollama", "run", MODEL_NAME, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    combined = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
    cleaned = clean_output(combined)
    if completed.returncode != 0 and not cleaned:
        raise RuntimeError(f"模型调用失败: {completed.returncode}\n{combined[:500]}")
    return cleaned


def strip_thinking(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lower = line.lower().strip()
        if "thinking" in lower or "done thinking" in lower:
            continue
        lines.append(line.strip())
    return "\n".join(line for line in lines if line).strip()


def english_ratio(text: str) -> float:
    if not text:
        return 0.0
    ascii_count = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return ascii_count / max(len(text), 1)


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk_count / max(len(text), 1)


def build_summary() -> dict:
    rows = []
    for item in PROMPTS:
        raw = run_prompt(item["prompt"])
        clean = strip_thinking(raw)
        e_ratio = english_ratio(clean)
        c_ratio = chinese_ratio(clean)
        if item["target"] == "english":
            hit = 1.0 if e_ratio > c_ratio and e_ratio > 0.1 else 0.0
        else:
            hit = 1.0 if c_ratio > e_ratio and c_ratio > 0.1 else 0.0
        rows.append(
            {
                "probe_name": item["name"],
                "target_language": item["target"],
                "english_ratio": e_ratio,
                "chinese_ratio": c_ratio,
                "hit": hit,
                "output_preview": clean[:120],
            }
        )
    correct_count = int(sum(row["hit"] for row in rows))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage256_deepseek14b_multidirection_translation_probe",
        "title": "DeepSeek14B 多方向翻译直测",
        "status_short": "deepseek14b_multidirection_translation_probe_ready",
        "model_name": MODEL_NAME,
        "probe_count": len(rows),
        "correct_count": correct_count,
        "behavior_score": correct_count / len(rows),
        "strongest_probe_name": max(rows, key=lambda row: row["hit"])["probe_name"],
        "weakest_probe_name": min(rows, key=lambda row: row["hit"])["probe_name"],
        "probe_rows": rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    report = [
        "# Stage256：DeepSeek14B 多方向翻译直测",
        "",
        "## 核心结果",
        f"- 模型：{summary['model_name']}",
        f"- 探针数量：{summary['probe_count']}",
        f"- 正确数量：{summary['correct_count']}",
        f"- 行为总分：{summary['behavior_score']:.4f}",
    ]
    (output_dir / "STAGE256_DEEPSEEK14B_MULTIDIRECTION_TRANSLATION_PROBE_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8-sig"
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek14B 多方向翻译直测")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
