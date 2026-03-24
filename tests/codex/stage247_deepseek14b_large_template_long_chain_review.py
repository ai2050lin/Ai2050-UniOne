#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage247_deepseek14b_large_template_long_chain_review_20260324"
MODEL_NAME = "deepseek-r1:14b"
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
SPINNER_ONLY_RE = re.compile(r"^[⠁-⣿\s]+$")

PROBES = [
    ("模板1 水果链", "只输出 apple 或 pear：Tom washed the apple beside the pear, sliced it, and served it sweet. Which object does the last it refer to?", "apple"),
    ("模板2 水果跨句", "只输出 apple 或 pear：Tom picked the apple near the pear. Later he washed it and served it. Which object does the final it refer to?", "apple"),
    ("模板3 工具干扰", "只输出 apple 或 knife：Tom held the knife near the apple, cut it, and ate it. Which object does the final it refer to?", "apple"),
    ("模板4 品牌设备", "只输出 phone 或 store：Apple sold the phone in the store, repaired it, and promoted it. Which object does the last it refer to?", "phone"),
    ("模板5 品牌跨句", "只输出 laptop 或 store：Apple released the laptop near the store. Later it updated it and sold it. Which object does the final it refer to when talking about the sold thing?", "laptop"),
    ("模板6 品牌修复", "只输出 yes 或 no：In 'Apple announced the laptop, not the store, updated it, and it became popular', does the last it refer to laptop?", "yes"),
    ("模板7 水果修复", "只输出 yes 或 no：In 'Tom sliced the apple, not the pear, washed it, and it became sweet', does the last it refer to apple?", "yes"),
    ("模板8 竞争水果", "只输出 apple 或 pear：Tom moved the pear near the apple, then washed it and sliced it. Which object does the final it refer to?", "apple"),
    ("模板9 竞争品牌", "只输出 device 或 company：Apple improved the device, praised it, shipped it, and compared it with the company store. Which object does the final it refer to?", "device"),
    ("模板10 双重干扰", "只输出 apple 或 knife：Tom put the knife beside the apple, cleaned the fruit, dried it, and ate it. Which object does the final it refer to?", "apple"),
    ("模板11 水果属性", "只输出 yes 或 no：In 'Tom peeled the apple and it became juicy', does the final it refer to apple?", "yes"),
    ("模板12 品牌属性", "只输出 yes 或 no：In 'Apple updated the phone and it became faster', does the final it refer to phone?", "yes"),
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
    valid = {"apple", "pear", "knife", "phone", "store", "laptop", "device", "company", "yes", "no"}
    for line in reversed(lines):
        lower = line.lower()
        for token in valid:
            if lower == token or lower == f"answer: {token}" or lower == f"**answer:** {token}":
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
    for token in ["apple", "pear", "knife", "phone", "store", "laptop", "device", "company", "yes", "no"]:
        if re.search(rf"\b{token}\b", lower):
            return token
    return "unknown"


def build_summary() -> dict:
    rows = []
    hit_count = 0
    for probe_name, prompt, expected in PROBES:
        raw = run_prompt(prompt)
        predicted = normalize_label(raw)
        ok = predicted == expected
        if ok:
            hit_count += 1
        rows.append(
            {
                "probe_name": probe_name,
                "expected": expected,
                "predicted": predicted,
                "is_correct": ok,
                "raw_output": raw[:300],
            }
        )
    score = hit_count / len(rows)
    weakest = next((row["probe_name"] for row in rows if not row["is_correct"]), "无")
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage247_deepseek14b_large_template_long_chain_review",
        "title": "DeepSeek14B 大模板长链复核",
        "status_short": "deepseek14b_large_template_long_chain_review_ready",
        "model_name": MODEL_NAME,
        "probe_count": len(rows),
        "review_score": score,
        "correct_count": hit_count,
        "weakest_probe_name": weakest,
        "top_gap_name": "14B 大模板长链仍需继续扩样验证天然来源保真",
        "probe_rows": rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage247：DeepSeek14B 大模板长链复核",
        "",
        "## 核心结果",
        f"- 模型：{summary['model_name']}",
        f"- 探针数量：{summary['probe_count']}",
        f"- 总分：{summary['review_score']:.4f}",
        f"- 正确数量：{summary['correct_count']}",
        f"- 最弱探针：{summary['weakest_probe_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE247_DEEPSEEK14B_LARGE_TEMPLATE_LONG_CHAIN_REVIEW_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="DeepSeek14B 大模板长链复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
