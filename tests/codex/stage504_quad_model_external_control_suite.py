#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from typing import Dict, List

import torch

from multimodel_language_shared import free_model, load_model_bundle
from ollama_complete_suite_shared import run_ollama_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage504_quad_model_external_control_suite_20260404"
)

MODEL_SPECS = [
    {"model_key": "qwen3", "display_name": "Qwen3-4B", "backend": "ollama", "model_name": "qwen3:4b"},
    {"model_key": "deepseek7b", "display_name": "DeepSeek-R1-7B", "backend": "ollama", "model_name": "deepseek-r1:7b"},
    {"model_key": "glm4", "display_name": "GLM-4-9B", "backend": "hf", "model_name": "glm4:9b"},
    {"model_key": "gemma4", "display_name": "Gemma4-e2b", "backend": "ollama", "model_name": "gemma4:e2b"},
]

PROBES: List[Dict[str, object]] = [
    {
        "category": "polysemy",
        "probe_name": "苹果水果义",
        "prompt": "只输出 A/B/C/D 中一个字母，不要解释。\n句子：“Tom washed the apple and ate it.”\n这里的 apple 更像：\nA. 水果\nB. 品牌\nC. 地点\nD. 动作",
        "expected": "A",
    },
    {
        "category": "polysemy",
        "probe_name": "苹果品牌义",
        "prompt": "只输出 A/B/C/D 中一个字母，不要解释。\n句子：“Apple released a new iPhone and updated it.”\n这里的 Apple 更像：\nA. 水果\nB. 品牌公司\nC. 地点\nD. 动作",
        "expected": "B",
    },
    {
        "category": "polysemy",
        "probe_name": "python 编程义",
        "prompt": "只输出 A/B/C/D 中一个字母，不要解释。\n句子：“She writes backend services in Python every day.”\n这里的 Python 更像：\nA. 水果\nB. 动物\nC. 编程语言\nD. 城市",
        "expected": "C",
    },
    {
        "category": "pattern_completion",
        "probe_name": "苹后接果",
        "prompt": "只输出 A/B/C/D 中一个字母，不要解释。\n短语前缀是“新鲜的苹”，最可能补成：\nA. 果\nB. 菜\nC. 子\nD. 园",
        "expected": "A",
    },
    {
        "category": "pattern_completion",
        "probe_name": "自后接己",
        "prompt": "只输出 A/B/C/D 中一个字母，不要解释。\n短语前缀是“保护自”，最可能补成：\nA. 己\nB. 人\nC. 上\nD. 下",
        "expected": "A",
    },
    {
        "category": "pattern_completion",
        "probe_name": "因后接为",
        "prompt": "只输出 A/B/C/D 中一个字母，不要解释。\n连接词前缀是“因”，最可能补成：\nA. 为\nB. 此\nC. 果\nD. 子",
        "expected": "A",
    },
    {
        "category": "long_route",
        "probe_name": "长距离责任归属",
        "prompt": "只输出 A/B/C/D 中一个字母，不要解释。\n张三昨天在例会上批评了李四，因为李四连续三周没有提交周报。晚上经理单独谈话时，最终承认错误的是：\nA. 张三\nB. 李四\nC. 经理\nD. 周报",
        "expected": "B",
    },
    {
        "category": "long_route",
        "probe_name": "长距离逻辑链",
        "prompt": "只输出 A/B/C/D 中一个字母，不要解释。\n如果设备过热就会报警；只要报警，工程师就会停机检查。现在设备已经过热，所以工程师接下来会：\nA. 停机检查\nB. 继续运行\nC. 离开现场\nD. 更换办公室",
        "expected": "A",
    },
    {
        "category": "concept_hierarchy",
        "probe_name": "苹果父类归属",
        "prompt": "只输出 A/B/C/D 中一个字母，不要解释。\n苹果更属于：\nA. 水果\nB. 颜色\nC. 味道\nD. 公司职位",
        "expected": "A",
    },
    {
        "category": "attribute_binding",
        "probe_name": "苹果颜色绑定",
        "prompt": "只输出 A/B/C/D 中一个字母，不要解释。\n如果问题是“苹果常见是什么颜色”，以下最合适的是：\nA. 红色\nB. 方形\nC. 嘈杂\nD. 缓慢",
        "expected": "A",
    },
]

CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_choice_letter(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        matches = CHOICE_RE.findall(line.upper())
        if matches:
            return matches[-1].upper()
    matches = CHOICE_RE.findall(text.upper())
    if matches:
        return matches[-1].upper()
    return "UNKNOWN"


def generate_hf_text(model, tokenizer, prompt: str, *, max_new_tokens: int = 8) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_model_probe(model_spec: dict, probe: dict, *, model_bundle=None) -> dict:
    if model_spec["backend"] == "ollama":
        raw_output = run_ollama_prompt(str(model_spec["model_name"]), str(probe["prompt"]), timeout=700)
    elif model_spec["backend"] == "hf":
        if model_bundle is None:
            raise RuntimeError("HF 后端缺少已加载模型")
        model, tokenizer = model_bundle
        raw_output = generate_hf_text(model, tokenizer, str(probe["prompt"]))
    else:
        raise ValueError(f"未知后端: {model_spec['backend']}")

    prediction = extract_choice_letter(raw_output)
    score = 1.0 if prediction == probe["expected"] else 0.0
    return {
        "category": probe["category"],
        "probe_name": probe["probe_name"],
        "expected": probe["expected"],
        "prediction": prediction,
        "score": score,
        "raw_output_preview": raw_output[:200],
    }


def summarize_rows(rows: List[dict]) -> dict:
    category_summary = {}
    for category in sorted({row["category"] for row in rows}):
        cat_rows = [row for row in rows if row["category"] == category]
        category_summary[category] = {
            "probe_count": len(cat_rows),
            "accuracy": round(sum(row["score"] for row in cat_rows) / max(len(cat_rows), 1), 4),
        }
    return {
        "overall_accuracy": round(sum(row["score"] for row in rows) / max(len(rows), 1), 4),
        "category_summary": category_summary,
    }


def run_model(model_spec: dict) -> dict:
    model_bundle = None
    try:
        if model_spec["backend"] == "hf":
            model_bundle = load_model_bundle(str(model_spec["model_key"]), prefer_cuda=True)
        rows = [run_model_probe(model_spec, probe, model_bundle=model_bundle) for probe in PROBES]
        return {
            "display_name": model_spec["display_name"],
            "backend": model_spec["backend"],
            "probe_rows": rows,
            "summary": summarize_rows(rows),
        }
    finally:
        if model_bundle is not None:
            free_model(model_bundle[0])


def build_report(summary: dict) -> str:
    lines = ["# stage504 四模型统一外部行为强控制协议", ""]
    lines.append("- 说明：本轮将 Gemma4 加入统一测试，但 Gemma4 当前是 GGUF/Ollama 形态，因此属于外部行为协议，不属于层内挂钩协议。")
    lines.append("")
    for model_key, model_row in summary["models"].items():
        lines.append(f"## {model_key}")
        lines.append("")
        lines.append(f"- 显示名：`{model_row['display_name']}`")
        lines.append(f"- 后端：`{model_row['backend']}`")
        lines.append(f"- 总准确率：`{model_row['summary']['overall_accuracy']}`")
        for category, cat_row in model_row["summary"]["category_summary"].items():
            lines.append(f"- `{category}`：`{cat_row['accuracy']}` ({cat_row['probe_count']} probes)")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    summary = {
        "stage": "stage504_quad_model_external_control_suite",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "probe_count": len(PROBES),
        "models": {},
    }
    for model_spec in MODEL_SPECS:
        model_key = str(model_spec["model_key"])
        summary["models"][model_key] = run_model(model_spec)
    overall_rank = sorted(
        (
            (model_key, row["summary"]["overall_accuracy"])
            for model_key, row in summary["models"].items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    summary["overall_ranking"] = overall_rank
    summary["elapsed_seconds"] = round(time.time() - started, 3)
    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "REPORT.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
