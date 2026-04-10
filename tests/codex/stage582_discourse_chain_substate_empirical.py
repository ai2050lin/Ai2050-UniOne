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
    / "stage582_discourse_chain_substate_empirical_20260409"
)
MODELS_IN_ORDER = ["qwen3", "deepseek7b", "glm4", "gemma4"]

EXPERIMENTS = {
    "discourse_recent_subject": [
        {
            "prompt": (
                "John called David about the schedule. "
                "David later emailed Mark with the final draft. "
                "After that, he printed the attachment. The word 'he' refers to"
            ),
            "candidates": [" David", " John", " Mark"],
            "correct": " David",
        },
        {
            "prompt": (
                "David called John about the schedule. "
                "John later emailed Mark with the final draft. "
                "After that, he printed the attachment. The word 'he' refers to"
            ),
            "candidates": [" John", " David", " Mark"],
            "correct": " John",
        },
        {
            "prompt": (
                "Tom met Alex near the station. "
                "Alex then phoned Brian about the tickets. "
                "Soon after, he checked the platform number. The word 'he' refers to"
            ),
            "candidates": [" Alex", " Tom", " Brian"],
            "correct": " Alex",
        },
        {
            "prompt": (
                "Alex met Tom near the station. "
                "Tom then phoned Brian about the tickets. "
                "Soon after, he checked the platform number. The word 'he' refers to"
            ),
            "candidates": [" Tom", " Alex", " Brian"],
            "correct": " Tom",
        },
    ],
    "discourse_long_chain": [
        {
            "prompt": (
                "John briefed David before the client call. "
                "David updated Mark after the call ended. "
                "Later John revised the budget sheet. "
                "He then archived the final version. The word 'he' refers to"
            ),
            "candidates": [" John", " David", " Mark"],
            "correct": " John",
        },
        {
            "prompt": (
                "David briefed John before the client call. "
                "John updated Mark after the call ended. "
                "Later David revised the budget sheet. "
                "He then archived the final version. The word 'he' refers to"
            ),
            "candidates": [" David", " John", " Mark"],
            "correct": " David",
        },
        {
            "prompt": (
                "Tom showed Alex the repair notes. "
                "Alex sent Brian the corrected file. "
                "Later Tom checked the printer again. "
                "He then signed the cover page. The word 'he' refers to"
            ),
            "candidates": [" Tom", " Alex", " Brian"],
            "correct": " Tom",
        },
        {
            "prompt": (
                "Alex showed Tom the repair notes. "
                "Tom sent Brian the corrected file. "
                "Later Alex checked the printer again. "
                "He then signed the cover page. The word 'he' refers to"
            ),
            "candidates": [" Alex", " Tom", " Brian"],
            "correct": " Alex",
        },
    ],
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
    print(f"[stage582] loading {model_key} ...", flush=True)
    model, tokenizer = load_model_bundle_safe(model_key)
    try:
        experiment_rows = {}
        for exp_name, cases in EXPERIMENTS.items():
            case_rows = [analyze_case(model, tokenizer, case) for case in cases]
            accuracy = float(sum(1 for row in case_rows if row["is_correct"]) / max(len(case_rows), 1))
            mean_margin = float(sum(row["margin_top1_top2"] for row in case_rows) / max(len(case_rows), 1))
            experiment_rows[exp_name] = {
                "accuracy": accuracy,
                "mean_margin_top1_top2": mean_margin,
                "case_rows": case_rows,
            }
        row = {
            "model_key": model_key,
            "model_label": MODEL_SPECS[model_key]["label"],
            "elapsed_seconds": round(time.time() - started, 3),
            "experiment_rows": experiment_rows,
            "discourse_mean_accuracy": float(
                (experiment_rows["discourse_recent_subject"]["accuracy"] + experiment_rows["discourse_long_chain"]["accuracy"])
                / 2.0
            ),
        }
        print(f"[stage582] finished {model_key} in {time.time() - started:.2f}s", flush=True)
        return row
    finally:
        free_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_core_answer(model_rows: Sequence[Dict[str, object]]) -> str:
    valid_rows = [row for row in model_rows if "error" not in row]
    if not valid_rows:
        return "跨句链路细分实验没有得到有效结果。"
    recent_mean = sum(float(row["experiment_rows"]["discourse_recent_subject"]["accuracy"]) for row in valid_rows) / len(valid_rows)
    long_mean = sum(float(row["experiment_rows"]["discourse_long_chain"]["accuracy"]) for row in valid_rows) / len(valid_rows)
    gap = long_mean - recent_mean
    if gap <= -0.15:
        return "长链回收明显弱于短链延续，P_discourse 很可能还需要继续拆分。"
    if gap >= 0.15:
        return "长链回收没有更难，当前样本下 P_discourse 尚未表现出明显内部分裂。"
    return "短链延续和长链回收接近，P_discourse 是否继续拆分仍需更大样本。"


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# stage582 跨句链路子状态实测",
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
        lines.append(f"## {row['model_label']}")
        for exp_name, exp_row in row["experiment_rows"].items():
            lines.append(f"- `{exp_name}` 准确率：`{exp_row['accuracy']:.4f}`")
            lines.append(f"- `{exp_name}` 平均 top1-top2 间隔：`{exp_row['mean_margin_top1_top2']:.4f}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="跨句链路子状态实测")
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
            print(f"[stage582] {model_key} failed: {exc!r}", flush=True)
            model_rows.append({"model_key": model_key, "model_label": MODEL_SPECS[model_key]["label"], "error": repr(exc)})

    summary_path = OUTPUT_DIR / "summary.json"
    merged_rows = merge_existing_rows(summary_path, model_rows)
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage582_discourse_chain_substate_empirical",
        "title": "跨句链路子状态实测",
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
