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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage284_multihop_translation_command_probe_20260324"
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
        "name": "single_en_weather",
        "prompt": "你是翻译器。只输出最终结果，不要解释。请把中文翻译为英文：今天天气不错。",
        "expected_final_language": "english",
    },
    {
        "name": "chain_en_fr_weather",
        "prompt": "你是翻译器。只输出最终结果，不要解释。请把中文翻译为英文，再翻译为法文：今天天气不错。",
        "expected_final_language": "french",
    },
    {
        "name": "chain_en_fr_apple",
        "prompt": "你是翻译器。只输出最终结果，不要解释。请把中文翻译为英文，再翻译为法文：苹果很甜。",
        "expected_final_language": "french",
    },
    {
        "name": "literal_command_text",
        "prompt": "你是翻译器。只输出最终结果，不要解释。请翻译“把中文翻译为英文，再翻译为法文”。",
        "expected_final_language": "english",
    },
]

FRENCH_HINTS = [" le ", " la ", " les ", " une ", " un ", " des ", " est ", " aujourd", " météo", " pomme", " douce", " beau"]
ENGLISH_HINTS = [" the ", " is ", " today", " weather", " apple", " sweet", " translate", " french", " english"]


def ask_ollama(model_name: str, prompt: str) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    return data["response"].strip()


def classify_language(text: str) -> str:
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "chinese"
    lower = f" {text.lower()} "
    french_hits = sum(1 for hint in FRENCH_HINTS if hint in lower) + len(re.findall(r"[àâçéèêëîïôûùüÿœ]", lower))
    english_hits = sum(1 for hint in ENGLISH_HINTS if hint in lower)
    if french_hits > english_hits:
        return "french"
    return "english"


def is_clean_output(text: str) -> bool:
    lower = text.lower()
    noisy_markers = ["解释", "说明", "think", "<think>", "translation:", "英文：", "法文：", "结果："]
    return not any(marker in lower for marker in noisy_markers)


def run_model(spec: dict) -> dict:
    prompt_rows = []
    correct = 0
    for item in PROMPTS:
        output = ask_ollama(spec["ollama_name"], item["prompt"])
        final_language = classify_language(output)
        clean = is_clean_output(output)
        matched = final_language == item["expected_final_language"]
        if matched:
            correct += 1
        prompt_rows.append(
            {
                "prompt_name": item["name"],
                "expected_final_language": item["expected_final_language"],
                "final_language": final_language,
                "clean_output": clean,
                "matched": matched,
                "output": output,
            }
        )
    score = correct / len(PROMPTS)
    clean_ratio = sum(1 for row in prompt_rows if row["clean_output"]) / len(PROMPTS)
    return {
        "model_tag": spec["model_tag"],
        "display_name": spec["display_name"],
        "behavior_score": float(score),
        "clean_ratio": float(clean_ratio),
        "prompt_rows": prompt_rows,
    }


def build_summary() -> dict:
    model_rows = [run_model(spec) for spec in MODEL_SPECS]
    strongest = max(model_rows, key=lambda item: item["behavior_score"])
    weakest = min(model_rows, key=lambda item: item["behavior_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage284_multihop_translation_command_probe",
        "title": "多跳翻译命令探针",
        "status_short": "multihop_translation_command_probe_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "model_rows": model_rows,
        "top_gap_name": "如果多跳翻译命令最后落不到法文，而只停在英文或解释层，那么更像任务路径或目标语言读出控制问题；如果前段命令理解正确但后段最终语言漂移，更接近天然来源保真和后段闭合问题",
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
    parser = argparse.ArgumentParser(description="多跳翻译命令探针")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
