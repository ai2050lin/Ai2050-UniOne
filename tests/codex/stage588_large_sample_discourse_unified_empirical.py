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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage588_large_sample_discourse_unified_empirical_20260410"
MODELS_IN_ORDER = ["qwen3", "deepseek7b", "glm4", "gemma4"]

EXPERIMENTS = {
    "recent_chain": [
        {
            "prompt": (
                "Lena reviewed the budget with Mia after breakfast. "
                "Mia then sent Oliver the corrected spreadsheet before noon. "
                "Soon after, she printed the final page. The person was"
            ),
            "candidates": [" Mia", " Lena", " Oliver"],
            "correct": " Mia",
        },
        {
            "prompt": (
                "Mia reviewed the budget with Lena after breakfast. "
                "Lena then sent Oliver the corrected spreadsheet before noon. "
                "Soon after, she printed the final page. The person was"
            ),
            "candidates": [" Lena", " Mia", " Oliver"],
            "correct": " Lena",
        },
        {
            "prompt": (
                "Ethan met Noah at the storage room before lunch. "
                "Noah later called Lucas about the delayed shipment. "
                "Soon after, he updated the checklist. The person was"
            ),
            "candidates": [" Noah", " Ethan", " Lucas"],
            "correct": " Noah",
        },
        {
            "prompt": (
                "Noah met Ethan at the storage room before lunch. "
                "Ethan later called Lucas about the delayed shipment. "
                "Soon after, he updated the checklist. The person was"
            ),
            "candidates": [" Ethan", " Noah", " Lucas"],
            "correct": " Ethan",
        },
        {
            "prompt": (
                "Sara welcomed Emma near the library. "
                "Emma then texted Chloe about the room number. "
                "A moment later, she checked the reservation desk. The person was"
            ),
            "candidates": [" Emma", " Sara", " Chloe"],
            "correct": " Emma",
        },
        {
            "prompt": (
                "Emma welcomed Sara near the library. "
                "Sara then texted Chloe about the room number. "
                "A moment later, she checked the reservation desk. The person was"
            ),
            "candidates": [" Sara", " Emma", " Chloe"],
            "correct": " Sara",
        },
    ],
    "reactivated_chain": [
        {
            "prompt": (
                "Lena outlined the schedule for Mia before the call. "
                "Later, Mia forwarded Oliver the revised agenda. "
                "Before leaving, Lena signed the cover sheet and closed the folder. "
                "That night, she placed it in the archive cabinet. The person was"
            ),
            "candidates": [" Lena", " Mia", " Oliver"],
            "correct": " Lena",
        },
        {
            "prompt": (
                "Mia outlined the schedule for Lena before the call. "
                "Later, Lena forwarded Oliver the revised agenda. "
                "Before leaving, Mia signed the cover sheet and closed the folder. "
                "That night, she placed it in the archive cabinet. The person was"
            ),
            "candidates": [" Mia", " Lena", " Oliver"],
            "correct": " Mia",
        },
        {
            "prompt": (
                "Ethan explained the repair plan to Noah in the morning. "
                "After lunch, Noah emailed Lucas the updated checklist. "
                "At sunset, Ethan unlocked the cabinet and took the binder. "
                "Later, he returned it to the front office. The person was"
            ),
            "candidates": [" Ethan", " Noah", " Lucas"],
            "correct": " Ethan",
        },
        {
            "prompt": (
                "Noah explained the repair plan to Ethan in the morning. "
                "After lunch, Ethan emailed Lucas the updated checklist. "
                "At sunset, Noah unlocked the cabinet and took the binder. "
                "Later, he returned it to the front office. The person was"
            ),
            "candidates": [" Noah", " Ethan", " Lucas"],
            "correct": " Noah",
        },
        {
            "prompt": (
                "Sara described the seating chart to Emma before dinner. "
                "Afterward, Emma sent Chloe the corrected invitation list. "
                "Near closing time, Sara sealed the envelope and took the receipt. "
                "On the way out, she handed it to the clerk. The person was"
            ),
            "candidates": [" Sara", " Emma", " Chloe"],
            "correct": " Sara",
        },
        {
            "prompt": (
                "Emma described the seating chart to Sara before dinner. "
                "Afterward, Sara sent Chloe the corrected invitation list. "
                "Near closing time, Emma sealed the envelope and took the receipt. "
                "On the way out, she handed it to the clerk. The person was"
            ),
            "candidates": [" Emma", " Sara", " Chloe"],
            "correct": " Emma",
        },
    ],
    "natural_story_chain": [
        {
            "prompt": (
                "Lena and Mia spent the afternoon preparing the exhibition room. "
                "Mia labeled the final crate while Oliver checked the lighting. "
                "When the visitors finally left, Lena carried the guestbook downstairs. "
                "Before going home, she locked it inside the office drawer. The person was"
            ),
            "candidates": [" Lena", " Mia", " Oliver"],
            "correct": " Lena",
        },
        {
            "prompt": (
                "Mia and Lena spent the afternoon preparing the exhibition room. "
                "Lena labeled the final crate while Oliver checked the lighting. "
                "When the visitors finally left, Mia carried the guestbook downstairs. "
                "Before going home, she locked it inside the office drawer. The person was"
            ),
            "candidates": [" Mia", " Lena", " Oliver"],
            "correct": " Mia",
        },
        {
            "prompt": (
                "Noah and Ethan stayed late at the workshop to finish the repair log. "
                "Ethan copied the serial numbers while Lucas sorted the spare parts. "
                "Just before midnight, Noah packed the signed report into a blue folder. "
                "A few minutes later, he slid it under the supervisor's door. The person was"
            ),
            "candidates": [" Noah", " Ethan", " Lucas"],
            "correct": " Noah",
        },
        {
            "prompt": (
                "Ethan and Noah stayed late at the workshop to finish the repair log. "
                "Noah copied the serial numbers while Lucas sorted the spare parts. "
                "Just before midnight, Ethan packed the signed report into a blue folder. "
                "A few minutes later, he slid it under the supervisor's door. The person was"
            ),
            "candidates": [" Ethan", " Noah", " Lucas"],
            "correct": " Ethan",
        },
        {
            "prompt": (
                "Emma and Sara arrived early to prepare the reading hall for the event. "
                "Sara arranged the chairs while Chloe tested the microphone near the stage. "
                "After the audience left, Emma collected the marked program sheets. "
                "Before the doors were shut, she stacked them behind the reception desk. The person was"
            ),
            "candidates": [" Emma", " Sara", " Chloe"],
            "correct": " Emma",
        },
        {
            "prompt": (
                "Sara and Emma arrived early to prepare the reading hall for the event. "
                "Emma arranged the chairs while Chloe tested the microphone near the stage. "
                "After the audience left, Sara collected the marked program sheets. "
                "Before the doors were shut, she stacked them behind the reception desk. The person was"
            ),
            "candidates": [" Sara", " Emma", " Chloe"],
            "correct": " Sara",
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
    encoded = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=320)
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
    margins = layer_margin(model, tokenizer, str(case["prompt"]), str(case["correct"]), str(wrong))
    return {
        "prompt": case["prompt"],
        "scores": score_map,
        "best_candidate": ranked[0][0],
        "correct_candidate": case["correct"],
        "is_correct": ranked[0][0] == case["correct"],
        "margin_top1_top2": float(ranked[0][1] - ranked[1][1]) if len(ranked) >= 2 else 0.0,
        "final_layer_margin": float(margins[-1]) if margins else 0.0,
        "layer_margins": margins,
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
    print(f"[stage588] loading {model_key} ...", flush=True)
    model, tokenizer = load_model_bundle_safe(model_key)
    try:
        experiment_rows = {}
        for exp_name, cases in EXPERIMENTS.items():
            case_rows = [analyze_case(model, tokenizer, case) for case in cases]
            accuracy = float(sum(1 for row in case_rows if row["is_correct"]) / max(len(case_rows), 1))
            mean_margin = float(sum(row["margin_top1_top2"] for row in case_rows) / max(len(case_rows), 1))
            mean_final_margin = float(sum(row["final_layer_margin"] for row in case_rows) / max(len(case_rows), 1))
            experiment_rows[exp_name] = {
                "accuracy": accuracy,
                "mean_margin_top1_top2": mean_margin,
                "mean_final_layer_margin": mean_final_margin,
                "case_count": len(case_rows),
                "case_rows": case_rows,
            }
        accuracies = [float(exp["accuracy"]) for exp in experiment_rows.values()]
        row = {
            "model_key": model_key,
            "model_label": MODEL_SPECS[model_key]["label"],
            "elapsed_seconds": round(time.time() - started, 3),
            "experiment_rows": experiment_rows,
            "large_sample_discourse_mean_accuracy": float(sum(accuracies) / len(accuracies)),
            "subtask_spread": float(max(accuracies) - min(accuracies)),
        }
        print(f"[stage588] finished {model_key} in {time.time() - started:.2f}s", flush=True)
        return row
    finally:
        free_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_core_answer(model_rows: Sequence[Dict[str, object]]) -> str:
    valid_rows = [row for row in model_rows if "error" not in row]
    if not valid_rows:
        return "大样本跨句链路实验没有得到有效结果。"
    overall_mean = sum(float(row["large_sample_discourse_mean_accuracy"]) for row in valid_rows) / len(valid_rows)
    spread_mean = sum(float(row["subtask_spread"]) for row in valid_rows) / len(valid_rows)
    if spread_mean >= 0.18:
        return "更大样本下，不同跨句链路子任务仍分裂明显，P_discourse 暂时还不够稳定。"
    if overall_mean < 0.60:
        return "更大样本下，P_discourse 整体仍偏弱，但内部没有继续裂解，当前更像弱而统一的瓶颈状态。"
    return "更大样本下，P_discourse 整体保持中等稳定，暂不支持继续细拆。"


def build_report(summary: Dict[str, object]) -> str:
    lines = ["# stage588 大样本跨句链路统一实测", "", "## 核心结论", summary["core_answer"], ""]
    for row in summary["model_rows"]:
        if "error" in row:
            lines.append(f"## {row['model_label']}")
            lines.append(f"- 运行失败：{row['error']}")
            lines.append("")
            continue
        lines.append(f"## {row['model_label']}")
        lines.append(f"- 大样本跨句链路平均准确率：`{row['large_sample_discourse_mean_accuracy']:.4f}`")
        lines.append(f"- 子任务波动：`{row['subtask_spread']:.4f}`")
        for exp_name, exp_row in row["experiment_rows"].items():
            lines.append(f"- `{exp_name}` 准确率：`{exp_row['accuracy']:.4f}`")
            lines.append(f"- `{exp_name}` 平均 top1-top2 间隔：`{exp_row['mean_margin_top1_top2']:.4f}`")
            lines.append(f"- `{exp_name}` 平均末层 margin：`{exp_row['mean_final_layer_margin']:.4f}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="大样本跨句链路统一实测")
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
            print(f"[stage588] {model_key} failed: {exc!r}", flush=True)
            model_rows.append({"model_key": model_key, "model_label": MODEL_SPECS[model_key]["label"], "error": repr(exc)})
    summary_path = OUTPUT_DIR / "summary.json"
    merged_rows = merge_existing_rows(summary_path, model_rows)
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage588_large_sample_discourse_unified_empirical",
        "title": "大样本跨句链路统一实测",
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
