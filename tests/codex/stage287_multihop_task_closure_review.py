#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage287_multihop_task_closure_review_20260324"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
if OLLAMA_HOST.startswith(":"):
    OLLAMA_HOST = f"http://127.0.0.1{OLLAMA_HOST}"
elif not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = f"http://{OLLAMA_HOST}"
OLLAMA_URL = f"{OLLAMA_HOST}/api/generate"

MODEL_SPECS = [
    {"model_tag": "qwen4b", "display_name": "Qwen3-4B", "ollama_name": "qwen3:4b"},
    {"model_tag": "deepseek14b", "display_name": "DeepSeek-R1-14B", "ollama_name": "deepseek-r1:14b"},
]

PROMPTS = [
    {
        "name": "translation_chain",
        "prompt": "你是翻译器。只输出最终结果，不要解释。请把中文翻译为英文，再翻译为法文：今天天气不错。",
        "expected_type": "french_only",
    },
    {
        "name": "translation_chain_apple",
        "prompt": "你是翻译器。只输出最终结果，不要解释。请把中文翻译为英文，再翻译为法文：苹果很甜。",
        "expected_type": "french_only",
    },
    {
        "name": "refactor_chain",
        "prompt": "你是代码助手。只输出最终代码，不要解释。请重构下面的 Python 函数，提取重复逻辑并重命名变量：\n\ndef work(a,b):\n    x=a+b\n    y=a+b\n    return x+y\n",
        "expected_type": "code_only",
    },
    {
        "name": "modify_chain",
        "prompt": "你是文本助手。只输出最终结果，不要解释。请先把下面这句话改成更正式的中文，再翻译为英文：今天天气不错。",
        "expected_type": "english_only",
    },
]

FRENCH_HINTS = [" le ", " la ", " les ", " une ", " un ", " des ", " est ", " aujourd", " beau", " douce", " pomme"]
ENGLISH_HINTS = [" the ", " is ", " today", " weather", " apple", " sweet", " code", " return "]


def ask_ollama(model_name: str, prompt: str) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["response"].strip()


def classify_output(text: str) -> str:
    lower = f" {text.lower()} "
    if "def " in lower or "return " in lower:
        return "code_only"
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "chinese_mixed"
    french_hits = sum(1 for hint in FRENCH_HINTS if hint in lower) + len(re.findall(r"[àâçéèêëîïôûùüÿœ]", lower))
    english_hits = sum(1 for hint in ENGLISH_HINTS if hint in lower)
    if french_hits > english_hits:
        return "french_only"
    return "english_only"


def clean_output(text: str) -> bool:
    lower = text.lower()
    noisy_markers = ["解释", "说明", "think", "<think>", "英文：", "法文：", "结果："]
    return not any(marker in lower for marker in noisy_markers)


def run_model(spec: dict) -> dict:
    rows = []
    correct = 0
    clean = 0
    for item in PROMPTS:
        output = ask_ollama(spec["ollama_name"], item["prompt"])
        final_type = classify_output(output)
        matched = final_type == item["expected_type"]
        is_clean = clean_output(output)
        correct += int(matched)
        clean += int(is_clean)
        rows.append(
            {
                "prompt_name": item["name"],
                "expected_type": item["expected_type"],
                "final_type": final_type,
                "matched": matched,
                "clean_output": is_clean,
                "output": output,
            }
        )
    return {
        "model_tag": spec["model_tag"],
        "display_name": spec["display_name"],
        "closure_score": correct / len(PROMPTS),
        "clean_ratio": clean / len(PROMPTS),
        "prompt_rows": rows,
    }


def build_summary() -> dict:
    rows = [run_model(spec) for spec in MODEL_SPECS]
    strongest = max(rows, key=lambda item: item["closure_score"])
    weakest = min(rows, key=lambda item: item["closure_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage287_multihop_task_closure_review",
        "title": "多跳任务后段闭合复核",
        "status_short": "multihop_task_closure_review_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "model_rows": rows,
        "top_gap_name": "如果多跳任务前段理解正确但后段把中间结果和最终结果一起吐出来，那么更像后段闭合和输出约束问题，而不是前段任务切换错误",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="多跳任务后段闭合复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
