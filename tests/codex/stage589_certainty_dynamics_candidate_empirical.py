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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage589_certainty_dynamics_candidate_empirical_20260410"
MODELS_IN_ORDER = ["qwen3", "deepseek7b", "glm4", "gemma4"]

EXPERIMENTS = {
    "certainty_commitment": [
        {
            "prompt": "Sentence: 'The report definitely reached the director.' The speaker shows",
            "candidates": [" certainty", " uncertainty", " contradiction"],
            "correct": " certainty",
        },
        {
            "prompt": "Sentence: 'The report probably reached the director.' The speaker shows",
            "candidates": [" uncertainty", " certainty", " contradiction"],
            "correct": " uncertainty",
        },
        {
            "prompt": "Sentence: 'The report cannot have reached the director.' The speaker shows",
            "candidates": [" contradiction", " certainty", " uncertainty"],
            "correct": " contradiction",
        },
        {
            "prompt": "Sentence: 'The report may have reached the director.' The speaker shows",
            "candidates": [" uncertainty", " certainty", " contradiction"],
            "correct": " uncertainty",
        },
    ],
    "certainty_consequence": [
        {
            "prompt": "Text: 'The report definitely reached the director.' Is the delivery settled, open, or impossible?",
            "candidates": [" settled", " open", " impossible"],
            "correct": " settled",
        },
        {
            "prompt": "Text: 'The report probably reached the director.' Is the delivery settled, open, or impossible?",
            "candidates": [" open", " settled", " impossible"],
            "correct": " open",
        },
        {
            "prompt": "Text: 'The report may have reached the director.' Is the delivery settled, open, or impossible?",
            "candidates": [" open", " settled", " impossible"],
            "correct": " open",
        },
        {
            "prompt": "Text: 'The report cannot have reached the director.' Is the delivery settled, open, or impossible?",
            "candidates": [" impossible", " open", " settled"],
            "correct": " impossible",
        },
    ],
    "certainty_counterclaim": [
        {
            "prompt": "Text: 'The report definitely reached the director.' Is the opposite claim 'The report did not reach the director' compatible?",
            "candidates": [" no", " yes", " maybe"],
            "correct": " no",
        },
        {
            "prompt": "Text: 'The report probably reached the director.' Is the opposite claim 'The report did not reach the director' compatible?",
            "candidates": [" yes", " no", " maybe"],
            "correct": " yes",
        },
        {
            "prompt": "Text: 'The report may have reached the director.' Is the opposite claim 'The report did not reach the director' compatible?",
            "candidates": [" yes", " no", " maybe"],
            "correct": " yes",
        },
        {
            "prompt": "Text: 'The report cannot have reached the director.' Is the opposite claim 'The report reached the director' compatible?",
            "candidates": [" no", " yes", " maybe"],
            "correct": " no",
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


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def thirds(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"early": 0.0, "mid": 0.0, "late": 0.0}
    n = len(values)
    first_end = max(1, n // 3)
    second_end = max(first_end + 1, (2 * n) // 3)
    return {
        "early": mean(values[:first_end]),
        "mid": mean(values[first_end:second_end]),
        "late": mean(values[second_end:]),
    }


def monotonic_gain_ratio(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    gains = 0
    for left, right in zip(values[:-1], values[1:]):
        if right >= left:
            gains += 1
    return float(gains / (len(values) - 1))


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
    margins = layer_margin(model, tokenizer, str(case["prompt"]), str(case["correct"]), str(wrong))
    segment_means = thirds(margins)
    return {
        "prompt": case["prompt"],
        "scores": score_map,
        "best_candidate": ranked[0][0],
        "correct_candidate": case["correct"],
        "is_correct": ranked[0][0] == case["correct"],
        "margin_top1_top2": float(ranked[0][1] - ranked[1][1]) if len(ranked) >= 2 else 0.0,
        "layer_margins": margins,
        "early_mean_margin": float(segment_means["early"]),
        "mid_mean_margin": float(segment_means["mid"]),
        "late_mean_margin": float(segment_means["late"]),
        "late_gain": float(segment_means["late"] - segment_means["early"]),
        "monotonic_gain_ratio": float(monotonic_gain_ratio(margins)),
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
    print(f"[stage589] loading {model_key} ...", flush=True)
    model, tokenizer = load_model_bundle_safe(model_key)
    try:
        experiment_rows = {}
        all_case_rows: List[Dict[str, object]] = []
        for exp_name, cases in EXPERIMENTS.items():
            case_rows = [analyze_case(model, tokenizer, case) for case in cases]
            all_case_rows.extend(case_rows)
            experiment_rows[exp_name] = {
                "accuracy": float(sum(1 for row in case_rows if row["is_correct"]) / max(len(case_rows), 1)),
                "mean_margin_top1_top2": float(sum(row["margin_top1_top2"] for row in case_rows) / max(len(case_rows), 1)),
                "early_mean_margin": float(sum(row["early_mean_margin"] for row in case_rows) / max(len(case_rows), 1)),
                "mid_mean_margin": float(sum(row["mid_mean_margin"] for row in case_rows) / max(len(case_rows), 1)),
                "late_mean_margin": float(sum(row["late_mean_margin"] for row in case_rows) / max(len(case_rows), 1)),
                "late_gain_mean": float(sum(row["late_gain"] for row in case_rows) / max(len(case_rows), 1)),
                "monotonic_gain_ratio_mean": float(sum(row["monotonic_gain_ratio"] for row in case_rows) / max(len(case_rows), 1)),
                "case_rows": case_rows,
            }
        row = {
            "model_key": model_key,
            "model_label": MODEL_SPECS[model_key]["label"],
            "elapsed_seconds": round(time.time() - started, 3),
            "experiment_rows": experiment_rows,
            "dynamics_means": {
                "accuracy_mean": float(sum(exp["accuracy"] for exp in experiment_rows.values()) / len(experiment_rows)),
                "early_mean": float(sum(row["early_mean_margin"] for row in all_case_rows) / max(len(all_case_rows), 1)),
                "mid_mean": float(sum(row["mid_mean_margin"] for row in all_case_rows) / max(len(all_case_rows), 1)),
                "late_mean": float(sum(row["late_mean_margin"] for row in all_case_rows) / max(len(all_case_rows), 1)),
                "late_gain_mean": float(sum(row["late_gain"] for row in all_case_rows) / max(len(all_case_rows), 1)),
                "monotonic_gain_ratio_mean": float(sum(row["monotonic_gain_ratio"] for row in all_case_rows) / max(len(all_case_rows), 1)),
            },
        }
        print(f"[stage589] finished {model_key} in {time.time() - started:.2f}s", flush=True)
        return row
    finally:
        free_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_core_answer(model_rows: Sequence[Dict[str, object]]) -> str:
    valid_rows = [row for row in model_rows if "error" not in row]
    if not valid_rows:
        return "确定性动力学候选实验没有得到有效结果。"
    early_mean = mean([float(row["dynamics_means"]["early_mean"]) for row in valid_rows])
    late_mean = mean([float(row["dynamics_means"]["late_mean"]) for row in valid_rows])
    gain_mean = mean([float(row["dynamics_means"]["late_gain_mean"]) for row in valid_rows])
    monotonic_mean = mean([float(row["dynamics_means"]["monotonic_gain_ratio_mean"]) for row in valid_rows])
    if late_mean > early_mean and gain_mean > 0.05 and monotonic_mean >= 0.55:
        return "Q_certainty 在层间表现出后层成形趋势，已经开始接近真正的候选状态动力学。"
    if late_mean > early_mean:
        return "Q_certainty 存在一定后层增强，但单调性还不够强，当前更像弱动力学候选。"
    return "Q_certainty 目前还没有表现出稳定的后层成形轨迹，距离闭式动力学仍远。"


def build_report(summary: Dict[str, object]) -> str:
    lines = ["# stage589 确定性动力学候选实测", "", "## 核心结论", summary["core_answer"], ""]
    for row in summary["model_rows"]:
        if "error" in row:
            lines.append(f"## {row['model_label']}")
            lines.append(f"- 运行失败：{row['error']}")
            lines.append("")
            continue
        lines.append(f"## {row['model_label']}")
        lines.append(f"- 平均准确率：`{row['dynamics_means']['accuracy_mean']:.4f}`")
        lines.append(f"- early_mean: `{row['dynamics_means']['early_mean']:.4f}`")
        lines.append(f"- mid_mean: `{row['dynamics_means']['mid_mean']:.4f}`")
        lines.append(f"- late_mean: `{row['dynamics_means']['late_mean']:.4f}`")
        lines.append(f"- late_gain_mean: `{row['dynamics_means']['late_gain_mean']:.4f}`")
        lines.append(f"- monotonic_gain_ratio_mean: `{row['dynamics_means']['monotonic_gain_ratio_mean']:.4f}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="确定性动力学候选实测")
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
            print(f"[stage589] {model_key} failed: {exc!r}", flush=True)
            model_rows.append({"model_key": model_key, "model_label": MODEL_SPECS[model_key]["label"], "error": repr(exc)})
    summary_path = OUTPUT_DIR / "summary.json"
    merged_rows = merge_existing_rows(summary_path, model_rows)
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage589_certainty_dynamics_candidate_empirical",
        "title": "确定性动力学候选实测",
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
