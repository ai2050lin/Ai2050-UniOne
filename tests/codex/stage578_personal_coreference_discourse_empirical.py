#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from multimodel_language_shared import MODEL_SPECS, candidate_score_map, free_model, load_model_bundle
from qwen3_language_shared import move_batch_to_model_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage578_personal_coreference_discourse_empirical_20260409"
)
MODELS_IN_ORDER = ["qwen3", "deepseek7b", "glm4", "gemma4"]

EXPERIMENTS = {
    "pronoun_personal_discourse": [
        {
            "prompt": (
                "John handed the report to David after David printed it. "
                "Later he filed the document in the archive. The word 'he' refers to"
            ),
            "candidates": [" David", " John"],
            "correct": " David",
        },
        {
            "prompt": (
                "David handed the report to John after John printed it. "
                "Later he filed the document in the archive. The word 'he' refers to"
            ),
            "candidates": [" John", " David"],
            "correct": " John",
        },
        {
            "prompt": (
                "John briefed David before David met the client. "
                "After the meeting, he wrote a summary for the team. The word 'he' refers to"
            ),
            "candidates": [" David", " John"],
            "correct": " David",
        },
        {
            "prompt": (
                "David briefed John before John met the client. "
                "After the meeting, he wrote a summary for the team. The word 'he' refers to"
            ),
            "candidates": [" John", " David"],
            "correct": " John",
        },
        {
            "prompt": (
                "Tom called Mark when Mark arrived at the station. "
                "A few minutes later he boarded the train. The word 'he' refers to"
            ),
            "candidates": [" Mark", " Tom"],
            "correct": " Mark",
        },
        {
            "prompt": (
                "Mark called Tom when Tom arrived at the station. "
                "A few minutes later he boarded the train. The word 'he' refers to"
            ),
            "candidates": [" Tom", " Mark"],
            "correct": " Tom",
        },
    ]
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_model_bundle_safe(model_key: str):
    if model_key == "gemma4":
        model_path = MODEL_SPECS[model_key]["model_path"]
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            local_files_only=True,
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="cpu",
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        return model, tokenizer
    if model_key in {"deepseek7b", "glm4"}:
        model_path = MODEL_SPECS[model_key]["model_path"]
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
            use_fast=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="cpu",
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        return model, tokenizer
    return load_model_bundle(model_key, prefer_cuda=True)


def safe_token_id(tokenizer, token_text: str) -> int | None:
    ids = tokenizer(token_text, add_special_tokens=False)["input_ids"]
    if len(ids) == 1:
        return int(ids[0])
    return None


def get_final_norm(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    return None


def layer_margin(model, tokenizer, prompt: str, correct: str, wrong: str) -> List[float]:
    correct_id = safe_token_id(tokenizer, correct)
    wrong_id = safe_token_id(tokenizer, wrong)
    if correct_id is None or wrong_id is None:
        return []
    encoded = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=256)
    encoded = move_batch_to_model_device(model, encoded)
    with torch.inference_mode():
        out = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
    head = model.get_output_embeddings()
    norm = get_final_norm(model)
    margins: List[float] = []
    for layer_hidden in out.hidden_states:
        vec = layer_hidden[:, -1, :].clone().detach()
        if norm is not None:
            vec = norm(vec)
        logits = head(vec.clone()).float()[0]
        margins.append(float((logits[correct_id] - logits[wrong_id]).item()))
    return margins


def analyze_case(model, tokenizer, case: Dict[str, object]) -> Dict[str, object]:
    score_map = candidate_score_map(model, tokenizer, str(case["prompt"]), case["candidates"])
    ranked = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
    wrong = next(candidate for candidate in case["candidates"] if candidate != case["correct"])
    return {
        "prompt": case["prompt"],
        "scores": score_map,
        "best_candidate": ranked[0][0],
        "correct_candidate": case["correct"],
        "is_correct": ranked[0][0] == case["correct"],
        "margin_top1_top2": float(ranked[0][1] - ranked[1][1]) if len(ranked) >= 2 else 0.0,
        "layer_margins": layer_margin(model, tokenizer, str(case["prompt"]), str(case["correct"]), str(wrong)),
    }


def merge_existing_rows(summary_path: Path, new_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    if not summary_path.exists():
        return list(new_rows)
    try:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return list(new_rows)
    merged: Dict[str, Dict[str, object]] = {}
    for row in existing.get("model_rows", []):
        merged[str(row.get("model_key"))] = row
    for row in new_rows:
        merged[str(row.get("model_key"))] = row
    return [merged[key] for key in MODELS_IN_ORDER if key in merged]


def analyze_model(model_key: str) -> Dict[str, object]:
    started = time.time()
    print(f"[stage578] loading {model_key} ...", flush=True)
    model, tokenizer = load_model_bundle_safe(model_key)
    try:
        case_rows = [analyze_case(model, tokenizer, case) for case in EXPERIMENTS["pronoun_personal_discourse"]]
        accuracy = float(sum(1 for row in case_rows if row["is_correct"]) / max(len(case_rows), 1))
        mean_margin = float(sum(row["margin_top1_top2"] for row in case_rows) / max(len(case_rows), 1))
        row = {
            "model_key": model_key,
            "model_label": MODEL_SPECS[model_key]["label"],
            "elapsed_seconds": round(time.time() - started, 3),
            "experiment_rows": {
                "pronoun_personal_discourse": {
                    "accuracy": accuracy,
                    "mean_margin_top1_top2": mean_margin,
                    "case_rows": case_rows,
                }
            },
            "core_reading": (
                "如果篇章级人称共指显著弱于句内简单共指，说明 P_personal 不能只被理解成局部绑定，"
                "而需要单独引入跨句链路状态。"
            ),
        }
        print(f"[stage578] finished {model_key} in {time.time() - started:.2f}s", flush=True)
        return row
    finally:
        free_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_core_answer(model_rows: Sequence[Dict[str, object]]) -> str:
    valid_rows = [row for row in model_rows if "error" not in row]
    if not valid_rows:
        return "篇章级人称共指实验未成功产出有效结果。"
    accuracies = [float(row["experiment_rows"]["pronoun_personal_discourse"]["accuracy"]) for row in valid_rows]
    mean_accuracy = sum(accuracies) / len(accuracies)
    if mean_accuracy >= 0.85:
        return "篇章级人称共指在四模型上整体较稳，P_personal 可以保留为强状态，但仍需和局部绑定区分。"
    if mean_accuracy >= 0.65:
        return "篇章级人称共指只有中等稳定度，P_personal 更像“局部绑定 + 跨句链路”混合状态，不能继续粗粒度处理。"
    return "篇章级人称共指整体偏弱，说明当前理论对 P_personal 的刻画还远不够，需要正式拆出跨句链路状态。"


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# stage578 篇章级人称共指链实测",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
    ]
    for row in summary["model_rows"]:
        if "error" in row:
            lines.append(f"## {row['model_label']}")
            lines.append(f"- 运行失败：{row['error']}")
            lines.append("")
            continue
        exp_row = row["experiment_rows"]["pronoun_personal_discourse"]
        lines.append(f"## {row['model_label']}")
        lines.append(f"- 准确率：`{exp_row['accuracy']:.4f}`")
        lines.append(f"- 平均 top1-top2 间隔：`{exp_row['mean_margin_top1_top2']:.4f}`")
        for case in exp_row["case_rows"][:2]:
            margins = case.get("layer_margins", [])
            if margins:
                lines.append(f"- 层间 margin 首层/末层：`{margins[0]:.4f}` / `{margins[-1]:.4f}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="篇章级人称共指链实测")
    parser.add_argument("--models", nargs="+", default=MODELS_IN_ORDER, help="按顺序运行的模型键")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    model_rows = []
    for model_key in args.models:
        try:
            model_rows.append(analyze_model(model_key))
        except Exception as exc:
            print(f"[stage578] {model_key} failed: {exc!r}", flush=True)
            model_rows.append({"model_key": model_key, "model_label": MODEL_SPECS[model_key]["label"], "error": repr(exc)})

    summary_path = OUTPUT_DIR / "summary.json"
    merged_rows = merge_existing_rows(summary_path, model_rows)
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage578_personal_coreference_discourse_empirical",
        "title": "篇章级人称共指链实测",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "models_in_order": MODELS_IN_ORDER,
        "model_rows": merged_rows,
        "core_answer": build_core_answer(merged_rows),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(OUTPUT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
