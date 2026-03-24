#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage238_deepseek14b_direct_chain_probe_20260324"
MODEL_NAME = "deepseek-r1:14b"

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

PROMPTS = [
    (
        "水果义触发",
        "只输出 fruit 或 brand：In the sentence 'I ate an apple after washing the fruit', "
        "does apple refer to fruit or brand?",
        "fruit",
    ),
    (
        "品牌义触发",
        "只输出 fruit 或 brand：In the sentence 'I bought an Apple laptop and updated the device', "
        "does Apple refer to fruit or brand?",
        "brand",
    ),
    (
        "水果结果链",
        "只输出 yes 或 no：In the sentence 'Tom sliced the apple and it became sweet', "
        "does it refer to apple?",
        "yes",
    ),
    (
        "品牌结果链",
        "只输出 yes 或 no：In the sentence 'Apple released a device and it became popular', "
        "does it refer to Apple device?",
        "yes",
    ),
]


def clean_output(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    # 优先取最后一个像答案的短行，避免把长思维过程当结果。
    for line in reversed(lines):
        lower = line.lower()
        if lower in {"fruit", "brand", "yes", "no"}:
            return lower
    return lines[-1]


def run_prompt(prompt: str) -> str:
    completed = subprocess.run(
        ["ollama", "run", MODEL_NAME, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=240,
    )
    combined = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
    cleaned = clean_output(combined)
    if completed.returncode != 0 and not cleaned:
        raise RuntimeError(f"模型调用失败：{completed.returncode}\n{combined[:500]}")
    return cleaned


def normalize_label(text: str) -> str:
    lower = text.lower()
    if "fruit" in lower:
        return "fruit"
    if "brand" in lower:
        return "brand"
    if re.search(r"\byes\b", lower):
        return "yes"
    if re.search(r"\bno\b", lower):
        return "no"
    if "是" in text:
        return "yes"
    if "否" in text or "不是" in text:
        return "no"
    return "unknown"


def build_summary() -> dict:
    rows = []
    hit_count = 0
    for probe_name, prompt, expected in PROMPTS:
        raw = run_prompt(prompt)
        predicted = normalize_label(raw)
        is_correct = predicted == expected
        if is_correct:
            hit_count += 1
        rows.append(
            {
                "probe_name": probe_name,
                "expected": expected,
                "predicted": predicted,
                "is_correct": is_correct,
                "raw_output": raw,
            }
        )

    score = hit_count / len(rows)
    weakest_probe = next((row["probe_name"] for row in rows if not row["is_correct"]), "无")
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage238_deepseek14b_direct_chain_probe",
        "title": "DeepSeek14B 直测处理链探针",
        "status_short": "deepseek14b_direct_chain_probe_ready",
        "model_name": MODEL_NAME,
        "probe_count": len(rows),
        "direct_chain_score": score,
        "correct_count": hit_count,
        "weakest_probe_name": weakest_probe,
        "top_gap_name": "14B 复杂处理后段仍需更长链复核",
        "probe_rows": rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage238：DeepSeek14B 直测处理链探针",
        "",
        "## 核心结果",
        f"- 模型：{summary['model_name']}",
        f"- 探针数量：{summary['probe_count']}",
        f"- 直测总分：{summary['direct_chain_score']:.4f}",
        f"- 正确数量：{summary['correct_count']}",
        f"- 最弱探针：{summary['weakest_probe_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE238_DEEPSEEK14B_DIRECT_CHAIN_PROBE_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek14B 直测处理链探针")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
