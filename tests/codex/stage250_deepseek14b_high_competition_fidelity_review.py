#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage250_deepseek14b_high_competition_fidelity_review_20260324"
MODEL_NAME = "deepseek-r1:14b"
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
SPINNER_ONLY_RE = re.compile(r"^[\s⠁-⣿]+$")

PROBES = [
    ("水果竞争1", "只输出 apple 或 pear：Tom moved the pear near the apple, washed the fruit, sliced it, and ate it. Which object does the last it refer to?", "apple"),
    ("水果竞争2", "只输出 banana 或 peach：Tom set the peach beside the banana, peeled the fruit, then ate it. Which object does the last it refer to?", "banana"),
    ("水果竞争3", "只输出 orange 或 lemon：Tom placed the lemon near the orange, washed the fruit, and served it sweet. Which object does the last it refer to?", "orange"),
    ("动物竞争1", "只输出 cat 或 dog：The dog stood near the cat, the pet was washed, and it later slept. Which object does the last it refer to?", "cat"),
    ("动物竞争2", "只输出 lion 或 tiger：The tiger stood near the lion, the animal was fed, and it later roared. Which object does the last it refer to?", "lion"),
    ("动物竞争3", "只输出 horse 或 rabbit：The rabbit stayed near the horse, the animal was brushed, and it later ran. Which object does the last it refer to?", "horse"),
    ("品牌竞争1", "只输出 iphone 或 macbook：Apple displayed the MacBook beside the iPhone, repaired the device, and sold it. Which object does the last it refer to?", "iphone"),
    ("跨类竞争1", "只输出 apple 或 knife：Tom put the knife near the apple, cleaned the fruit, cut it, and ate it. Which object does the last it refer to?", "apple"),
]


def clean_output(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = text.replace("\r", "\n")
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or SPINNER_ONLY_RE.fullmatch(line):
            continue
        lines.append(line)
    if not lines:
        return ""
    valid = ["apple", "pear", "banana", "peach", "orange", "lemon", "cat", "dog", "lion", "tiger", "horse", "rabbit", "iphone", "macbook", "ipad", "knife", "brush"]
    for line in reversed(lines):
        lower = line.lower()
        for token in valid:
            if lower == token or lower == f"answer: {token}" or lower == f"**answer:** {token}" or lower == f"answer: **{token}**":
                return token
        patterns = [
            r"refer[s]? to (?:the )?([a-z]+)",
            r"answer:\s*\**([a-z]+)\**",
            r"\b(it refers to|refers to the)\s+([a-z]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, lower)
            if match:
                groups = [group for group in match.groups() if group]
                for group in reversed(groups):
                    if group in valid:
                        return group
        found = [token for token in valid if re.search(rf"\b{re.escape(token)}\b", lower)]
        if len(found) == 1:
            return found[0]
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
        raise RuntimeError(f"模型调用失败: {completed.returncode}\n{combined[:500]}")
    return cleaned


def build_summary() -> dict:
    probe_rows = []
    correct_count = 0
    for probe_name, prompt, expected in PROBES:
        predicted = run_prompt(prompt)
        ok = predicted == expected
        if ok:
            correct_count += 1
        probe_rows.append(
            {
                "probe_name": probe_name,
                "expected": expected,
                "predicted": predicted,
                "is_correct": ok,
            }
        )
    review_score = correct_count / len(PROBES)
    weakest = next((row["probe_name"] for row in probe_rows if not row["is_correct"]), "无")
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage250_deepseek14b_high_competition_fidelity_review",
        "title": "DeepSeek14B 高竞争来源保真复核",
        "status_short": "deepseek14b_high_competition_fidelity_review_ready",
        "model_name": MODEL_NAME,
        "probe_count": len(PROBES),
        "correct_count": correct_count,
        "review_score": review_score,
        "weakest_probe_name": weakest,
        "top_gap_name": "高竞争同类对象仍是天然来源保真最容易失稳的场景",
        "probe_rows": probe_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    report = [
        "# Stage250：DeepSeek14B 高竞争来源保真复核",
        "",
        "## 核心结果",
        f"- 模型：{summary['model_name']}",
        f"- 探针数量：{summary['probe_count']}",
        f"- 正确数量：{summary['correct_count']}",
        f"- 总分：{summary['review_score']:.4f}",
        f"- 最弱探针：{summary['weakest_probe_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE250_DEEPSEEK14B_HIGH_COMPETITION_FIDELITY_REVIEW_REPORT.md").write_text("\n".join(report), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek14B 高竞争来源保真复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
