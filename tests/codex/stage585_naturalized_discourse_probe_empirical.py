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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage585_naturalized_discourse_probe_empirical_20260409"
MODELS_IN_ORDER = ["qwen3", "deepseek7b", "glm4", "gemma4"]

EXPERIMENTS = {
    "narrative_recent_actor": [
        {
            "prompt": (
                "John spent the morning reviewing the budget. "
                "By lunch, David had already sent the updated file to the team. "
                "That evening, he called the manager to confirm the numbers. The person was"
            ),
            "candidates": [" David", " John"],
            "correct": " David",
        },
        {
            "prompt": (
                "David spent the morning reviewing the budget. "
                "By lunch, John had already sent the updated file to the team. "
                "That evening, he called the manager to confirm the numbers. The person was"
            ),
            "candidates": [" John", " David"],
            "correct": " John",
        },
        {
            "prompt": (
                "Tom arrived first at the station. "
                "A few minutes later, Alex bought the tickets for the trip. "
                "Soon after, he checked the departure board. The person was"
            ),
            "candidates": [" Alex", " Tom"],
            "correct": " Alex",
        },
        {
            "prompt": (
                "Alex arrived first at the station. "
                "A few minutes later, Tom bought the tickets for the trip. "
                "Soon after, he checked the departure board. The person was"
            ),
            "candidates": [" Tom", " Alex"],
            "correct": " Tom",
        },
    ],
    "narrative_reactivated_actor": [
        {
            "prompt": (
                "John explained the plan to David before the meeting. "
                "Afterward, David sent Mark the revised notes. "
                "Before leaving, John locked the cabinet and took the key. "
                "Later, he placed it in the archive box. The person was"
            ),
            "candidates": [" John", " David"],
            "correct": " John",
        },
        {
            "prompt": (
                "David explained the plan to John before the meeting. "
                "Afterward, John sent Mark the revised notes. "
                "Before leaving, David locked the cabinet and took the key. "
                "Later, he placed it in the archive box. The person was"
            ),
            "candidates": [" David", " John"],
            "correct": " David",
        },
        {
            "prompt": (
                "Tom showed the draft to Alex in the afternoon. "
                "Later, Alex emailed Brian the corrected version. "
                "Near sunset, Tom signed the cover page and packed the folder. "
                "After dinner, he dropped it at the office. The person was"
            ),
            "candidates": [" Tom", " Alex"],
            "correct": " Tom",
        },
        {
            "prompt": (
                "Alex showed the draft to Tom in the afternoon. "
                "Later, Tom emailed Brian the corrected version. "
                "Near sunset, Alex signed the cover page and packed the folder. "
                "After dinner, he dropped it at the office. The person was"
            ),
            "candidates": [" Alex", " Tom"],
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
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True, use_fast=False)
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
    print(f"[stage585] loading {model_key} ...", flush=True)
    model, tokenizer = load_model_bundle_safe(model_key)
    try:
        experiment_rows = {}
        for exp_name, cases in EXPERIMENTS.items():
            case_rows = [analyze_case(model, tokenizer, case) for case in cases]
            accuracy = float(sum(1 for row in case_rows if row["is_correct"]) / max(len(case_rows), 1))
            mean_margin = float(sum(row["margin_top1_top2"] for row in case_rows) / max(len(case_rows), 1))
            experiment_rows[exp_name] = {"accuracy": accuracy, "mean_margin_top1_top2": mean_margin, "case_rows": case_rows}
        row = {
            "model_key": model_key,
            "model_label": MODEL_SPECS[model_key]["label"],
            "elapsed_seconds": round(time.time() - started, 3),
            "experiment_rows": experiment_rows,
            "natural_discourse_mean_accuracy": float(
                (experiment_rows["narrative_recent_actor"]["accuracy"] + experiment_rows["narrative_reactivated_actor"]["accuracy"]) / 2.0
            ),
        }
        print(f"[stage585] finished {model_key} in {time.time() - started:.2f}s", flush=True)
        return row
    finally:
        free_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_core_answer(model_rows: Sequence[Dict[str, object]]) -> str:
    valid_rows = [row for row in model_rows if "error" not in row]
    if not valid_rows:
        return "自然叙事跨句链路实验没有得到有效结果。"
    recent_mean = sum(float(row["experiment_rows"]["narrative_recent_actor"]["accuracy"]) for row in valid_rows) / len(valid_rows)
    react_mean = sum(float(row["experiment_rows"]["narrative_reactivated_actor"]["accuracy"]) for row in valid_rows) / len(valid_rows)
    if abs(react_mean - recent_mean) >= 0.20:
        return "自然叙事口径下，跨句链路仍表现出内部差异，P_discourse 的测量仍需继续打磨。"
    return "自然叙事口径下，P_discourse 两类子任务接近，至少没有出现更强的内部裂解证据。"


def build_report(summary: Dict[str, object]) -> str:
    lines = ["# stage585 自然叙事跨句链路实测", "", "## 核心结论", summary["core_answer"], ""]
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
    parser = argparse.ArgumentParser(description="自然叙事跨句链路实测")
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
            print(f"[stage585] {model_key} failed: {exc!r}", flush=True)
            model_rows.append({"model_key": model_key, "model_label": MODEL_SPECS[model_key]["label"], "error": repr(exc)})
    summary_path = OUTPUT_DIR / "summary.json"
    merged_rows = merge_existing_rows(summary_path, model_rows)
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage585_naturalized_discourse_probe_empirical",
        "title": "自然叙事跨句链路实测",
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
