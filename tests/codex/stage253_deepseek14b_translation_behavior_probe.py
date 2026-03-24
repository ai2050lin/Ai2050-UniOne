#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage253_deepseek14b_translation_behavior_probe_20260324"
MODEL_NAME = "deepseek-r1:14b"
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

PROMPTS = {
    "plain": "请直接续写这句话，不要翻译：今天天气不错",
    "translate_en": "请把中文翻译为英文，只输出翻译结果：今天天气不错",
}


def clean_output(text: str) -> str:
    text = ANSI_RE.sub("", text)
    return text.replace("\r", "\n").strip()


def extract_english_phrase(text: str) -> str:
    quote_patterns = [
        r"[“\"]([^”\"\n]*[A-Za-z][^”\"\n]*)[”\"]",
        r"'([^'\n]*[A-Za-z][^'\n]*)'",
    ]
    for pattern in quote_patterns:
        matches = [m.strip() for m in re.findall(pattern, text) if "thinking" not in m.lower()]
        if matches:
            return max(matches, key=len).strip()
    line_candidates = []
    for line in text.splitlines():
        line = line.strip()
        if "thinking" in line.lower():
            continue
        if re.search(r"[A-Za-z]{3,}", line):
            line_candidates.append(line)
    if line_candidates:
        return max(line_candidates, key=len)
    return ""


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


def ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    ascii_count = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return ascii_count / max(len(text), 1)


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk_count / max(len(text), 1)


def build_summary() -> dict:
    plain_out = run_prompt(PROMPTS["plain"])
    trans_out = run_prompt(PROMPTS["translate_en"])
    plain_english_phrase = extract_english_phrase(plain_out)
    trans_english_phrase = extract_english_phrase(trans_out)
    plain_ascii = ascii_ratio(plain_english_phrase)
    trans_ascii = ascii_ratio(trans_english_phrase)
    plain_cjk = cjk_ratio(plain_english_phrase)
    trans_cjk = cjk_ratio(trans_english_phrase)
    behavior_score = (
        min(1.0, trans_ascii + (1.0 if trans_english_phrase else 0.0))
        + max(0.0, 1.0 - plain_ascii)
    ) / 2.0
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage253_deepseek14b_translation_behavior_probe",
        "title": "DeepSeek14B 翻译行为探针",
        "status_short": "deepseek14b_translation_behavior_probe_ready",
        "model_name": MODEL_NAME,
        "plain_output": plain_out[:400],
        "translate_output": trans_out[:400],
        "plain_english_phrase": plain_english_phrase,
        "translate_english_phrase": trans_english_phrase,
        "plain_ascii_ratio": plain_ascii,
        "translate_ascii_ratio": trans_ascii,
        "plain_cjk_ratio": plain_cjk,
        "translate_cjk_ratio": trans_cjk,
        "behavior_score": behavior_score,
        "top_gap_name": "翻译命令是否把输出语言稳定切到英文仍需更多模板复核",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    report = [
        "# Stage253：DeepSeek14B 翻译行为探针",
        "",
        "## 核心结果",
        f"- 模型：{summary['model_name']}",
        f"- 普通续写英文比例：{summary['plain_ascii_ratio']:.4f}",
        f"- 翻译命令英文比例：{summary['translate_ascii_ratio']:.4f}",
        f"- 普通续写中文比例：{summary['plain_cjk_ratio']:.4f}",
        f"- 翻译命令中文比例：{summary['translate_cjk_ratio']:.4f}",
        f"- 行为总分：{summary['behavior_score']:.4f}",
    ]
    (output_dir / "STAGE253_DEEPSEEK14B_TRANSLATION_BEHAVIOR_PROBE_REPORT.md").write_text("\n".join(report), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek14B 翻译行为探针")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
