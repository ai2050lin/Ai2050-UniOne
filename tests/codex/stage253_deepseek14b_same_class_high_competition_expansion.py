#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage253_deepseek14b_same_class_high_competition_expansion_20260324"
STAGE250_SUMMARY = PROJECT_ROOT / "tests" / "codex_temp" / "stage250_deepseek14b_high_competition_fidelity_review_20260324" / "summary.json"
MODEL_NAME = "deepseek-r1:14b"
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
SPINNER_ONLY_RE = re.compile(r"^[\\s⠋-⣿]+$")

EXPANSION_PROBES = [
    ("水果竞争扩张", "只输出 grape 或 melon：Tom moved the melon beside the grape, chilled the fruit, and later served it fresh. Which object does the last it refer to?", "grape", "fruit"),
    ("动物竞争扩张", "只输出 wolf 或 bear：The bear stayed near the wolf, the animal was tracked, and it later howled. Which object does the last it refer to?", "wolf", "animal"),
    ("品牌竞争扩张1", "只输出 ipad 或 iphone：Apple placed the iPhone beside the iPad, charged the device, and shipped it. Which object does the last it refer to?", "ipad", "brand"),
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
    valid = ["apple", "pear", "banana", "peach", "orange", "lemon", "grape", "melon", "cat", "dog", "lion", "tiger", "horse", "rabbit", "wolf", "bear", "iphone", "macbook", "ipad"]
    for line in reversed(lines):
        lower = line.lower()
        if lower in valid:
            return lower
        answer_match = re.search(r"answer:\s*\**([a-z0-9]+)\**", lower)
        if answer_match and answer_match.group(1) in valid:
            return answer_match.group(1)
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


def load_stage250_rows() -> list[dict]:
    summary = json.loads(STAGE250_SUMMARY.read_text(encoding="utf-8-sig"))
    rows = []
    for row in summary["probe_rows"]:
        probe_name = str(row["probe_name"])
        if probe_name.startswith("水果"):
            family_name = "fruit"
        elif probe_name.startswith("动物"):
            family_name = "animal"
        elif probe_name.startswith("品牌"):
            family_name = "brand"
        else:
            family_name = "other"
        rows.append(
            {
                "probe_name": probe_name,
                "family_name": family_name,
                "expected": row["expected"],
                "predicted": row["predicted"],
                "is_correct": bool(row["is_correct"]),
            }
        )
    return rows


def build_summary() -> dict:
    probe_rows = load_stage250_rows()
    family_counter = {"fruit": [0, 0], "animal": [0, 0], "brand": [0, 0]}
    for row in probe_rows:
        family_name = row["family_name"]
        if family_name in family_counter:
            family_counter[family_name][1] += 1
            if row["is_correct"]:
                family_counter[family_name][0] += 1
    for probe_name, prompt, expected, family_name in EXPANSION_PROBES:
        predicted = run_prompt(prompt)
        ok = predicted == expected
        family_counter[family_name][1] += 1
        if ok:
            family_counter[family_name][0] += 1
        probe_rows.append(
            {
                "probe_name": probe_name,
                "family_name": family_name,
                "expected": expected,
                "predicted": predicted,
                "is_correct": ok,
            }
        )
    correct_count = sum(1 for row in probe_rows if row["is_correct"])
    family_scores = {
        family_name: (right / total if total else 0.0)
        for family_name, (right, total) in family_counter.items()
    }
    weakest_family = min(family_scores.items(), key=lambda item: item[1])[0]
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage253_deepseek14b_same_class_high_competition_expansion",
        "title": "DeepSeek14B 同类高竞争长链扩张",
        "status_short": "deepseek14b_same_class_high_competition_expansion_ready",
        "model_name": MODEL_NAME,
        "probe_count": len(probe_rows),
        "correct_count": correct_count,
        "review_score": correct_count / len(probe_rows),
        "fruit_score": family_scores["fruit"],
        "animal_score": family_scores["animal"],
        "brand_score": family_scores["brand"],
        "weakest_family_name": weakest_family,
        "top_gap_name": "同类高竞争会优先暴露薄差分结构和天然来源保真的上限",
        "probe_rows": probe_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    report = [
        "# Stage253：DeepSeek14B 同类高竞争长链扩张",
        "",
        "## 核心结果",
        f"- 模型：{summary['model_name']}",
        f"- 探针数量：{summary['probe_count']}",
        f"- 正确数量：{summary['correct_count']}",
        f"- 总分：{summary['review_score']:.4f}",
        f"- 水果得分：{summary['fruit_score']:.4f}",
        f"- 动物得分：{summary['animal_score']:.4f}",
        f"- 品牌得分：{summary['brand_score']:.4f}",
        f"- 最弱家族：{summary['weakest_family_name']}",
        f"- 头号发现：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE253_DEEPSEEK14B_SAME_CLASS_HIGH_COMPETITION_EXPANSION_REPORT.md").write_text("\n".join(report), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek14B 同类高竞争长链扩张")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
