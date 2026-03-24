#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage241_deepseek14b_long_chain_probe_20260324"
MODEL_NAME = "deepseek-r1:14b"
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
SPINNER_ONLY_RE = re.compile(r"^[⠁-⣿\s]+$")

PROBES = [
    {
        "probe_name": "多对象水果归属",
        "prompt": "只输出 apple 或 pear：Tom washed the apple beside the pear, then sliced it and served it sweet. Which object does it most likely refer to at the end?",
        "expected": "apple",
    },
    {
        "probe_name": "多对象品牌归属",
        "prompt": "只输出 apple 或 phone：Apple announced a phone, improved it, and sold it quickly. Which object does it most likely refer to at the end?",
        "expected": "phone",
    },
    {
        "probe_name": "水果长链结果",
        "prompt": "只输出 yes 或 no：In 'Tom picked the apple, washed it, sliced it, and it became sweet', does the final it refer to apple?",
        "expected": "yes",
    },
    {
        "probe_name": "品牌长链结果",
        "prompt": "只输出 yes 或 no：In 'Apple designed the laptop, updated it, and it became popular', does the final it refer to laptop?",
        "expected": "yes",
    },
    {
        "probe_name": "竞争干扰水果链",
        "prompt": "只输出 apple 或 knife：Tom held the knife near the apple, cleaned the fruit, sliced it, and ate it. Which object does the last it most likely refer to?",
        "expected": "apple",
    },
    {
        "probe_name": "竞争干扰品牌链",
        "prompt": "只输出 company 或 device：Apple discussed the device, refined it, shipped it, and praised it. Which object does the last it most likely refer to?",
        "expected": "device",
    },
]


def clean_output(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = text.replace("\r", "\n")
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if SPINNER_ONLY_RE.fullmatch(line):
            continue
        lines.append(line)
    if not lines:
        return ""
    for line in reversed(lines):
        lower = line.lower()
        if lower in {"apple", "pear", "phone", "knife", "company", "device", "yes", "no"}:
            return lower
    return lines[-1].strip().lower()


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
        raise RuntimeError(f"模型调用失败：{completed.returncode}\n{combined[:500]}")
    return cleaned


def normalize_label(text: str) -> str:
    lower = text.lower()
    for token in ["apple", "pear", "phone", "knife", "company", "device", "yes", "no"]:
        if re.search(rf"\b{token}\b", lower):
            return token
    return "unknown"


def build_summary() -> dict:
    rows = []
    hit_count = 0
    for probe in PROBES:
        raw = run_prompt(probe["prompt"])
        predicted = normalize_label(raw)
        ok = predicted == probe["expected"]
        if ok:
            hit_count += 1
        rows.append(
            {
                "probe_name": probe["probe_name"],
                "expected": probe["expected"],
                "predicted": predicted,
                "is_correct": ok,
                "raw_output": raw,
            }
        )
    score = hit_count / len(rows)
    weakest = next((row["probe_name"] for row in rows if not row["is_correct"]), "无")
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage241_deepseek14b_long_chain_probe",
        "title": "DeepSeek14B 长链复杂处理复核",
        "status_short": "deepseek14b_long_chain_probe_ready",
        "model_name": MODEL_NAME,
        "probe_count": len(rows),
        "long_chain_score": score,
        "correct_count": hit_count,
        "weakest_probe_name": weakest,
        "top_gap_name": "14B 长链多对象竞争仍需继续加压",
        "probe_rows": rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage241：DeepSeek14B 长链复杂处理复核",
        "",
        "## 核心结果",
        f"- 模型：{summary['model_name']}",
        f"- 探针数量：{summary['probe_count']}",
        f"- 长链总分：{summary['long_chain_score']:.4f}",
        f"- 正确数量：{summary['correct_count']}",
        f"- 最弱探针：{summary['weakest_probe_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE241_DEEPSEEK14B_LONG_CHAIN_PROBE_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="DeepSeek14B 长链复杂处理复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
