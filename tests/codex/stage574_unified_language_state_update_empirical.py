#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import gc
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from multimodel_language_shared import (
    MODEL_SPECS,
    candidate_score_map,
    free_model,
    load_model_bundle,
)
from qwen3_language_shared import move_batch_to_model_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage574_unified_language_state_update_empirical_20260409"
)

MODELS_IN_ORDER = ["qwen3", "deepseek7b", "glm4", "gemma4"]
EXPERIMENTS = {
    "pronoun_coreference": [
        {
            "prompt": "Sentence: 'Mary thanked John because he helped with the project.' The word 'he' refers to",
            "candidates": [" John", " Mary"],
            "correct": " John",
        },
        {
            "prompt": "Sentence: 'John thanked Mary because she helped with the project.' The word 'she' refers to",
            "candidates": [" Mary", " John"],
            "correct": " Mary",
        },
    ],
    "preposition_relation": [
        {
            "prompt": "Sentence: 'The apple is under the table.' The relation between apple and table is",
            "candidates": [" under", " on", " near"],
            "correct": " under",
        },
        {
            "prompt": "Sentence: 'The key is in the box.' The relation between key and box is",
            "candidates": [" in", " under", " over"],
            "correct": " in",
        },
    ],
    "adverb_scope": [
        {
            "prompt": "Sentence: 'The boy quickly opened the door.' The word 'quickly' mainly modifies the",
            "candidates": [" action", " door", " boy"],
            "correct": " action",
        },
        {
            "prompt": "Sentence: 'The boy probably opened the door.' The word 'probably' mainly modifies the",
            "candidates": [" statement", " door", " boy"],
            "correct": " statement",
        },
    ],
    "logic_reasoning": [
        {
            "prompt": "All fruits are edible. Apple is a fruit. Therefore apple is",
            "candidates": [" edible", " metal", " green"],
            "correct": " edible",
        },
        {
            "prompt": "No red fruit is green. This apple is red. Therefore this apple is",
            "candidates": [" green", " red", " not"],
            "correct": " not",
        },
        {
            "prompt": "If the key is in the box, then the box is on the shelf. The key is in the box. Therefore the key is on the",
            "candidates": [" shelf", " floor", " table"],
            "correct": " shelf",
        },
    ],
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_model_bundle_safe(model_key: str):
    if model_key == "gemma4":
        model_path = MODEL_SPECS[model_key]["model_path"]
        processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
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
    if model_key not in {"deepseek7b", "glm4"}:
        return load_model_bundle(model_key, prefer_cuda=True)
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


def logic_layer_margin(model, tokenizer, prompt: str, correct: str, wrong: str) -> List[float]:
    correct_id = safe_token_id(tokenizer, correct)
    wrong_id = safe_token_id(tokenizer, wrong)
    if correct_id is None or wrong_id is None:
        return []
    encoded = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=192)
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
    return {
        "prompt": case["prompt"],
        "scores": score_map,
        "best_candidate": ranked[0][0],
        "correct_candidate": case["correct"],
        "is_correct": ranked[0][0] == case["correct"],
        "margin_top1_top2": float(ranked[0][1] - ranked[1][1]) if len(ranked) >= 2 else 0.0,
    }


def analyze_model(model_key: str) -> Dict[str, object]:
    started = time.time()
    print(f"[stage574] loading {model_key} ...", flush=True)
    model, tokenizer = load_model_bundle_safe(model_key)
    try:
        experiment_rows = {}
        for exp_name, cases in EXPERIMENTS.items():
            rows = [analyze_case(model, tokenizer, case) for case in cases]
            accuracy = float(sum(1 for row in rows if row["is_correct"]) / max(len(rows), 1))
            experiment_rows[exp_name] = {
                "accuracy": accuracy,
                "case_rows": rows,
            }

        logic_margins = []
        for case in EXPERIMENTS["logic_reasoning"]:
            wrong = next(c for c in case["candidates"] if c != case["correct"])
            logic_margins.append(
                {
                    "prompt": case["prompt"],
                    "correct": case["correct"],
                    "wrong": wrong,
                    "layer_margins": logic_layer_margin(model, tokenizer, case["prompt"], case["correct"], wrong),
                }
            )
        experiment_rows["logic_reasoning"]["layerwise_margin_rows"] = logic_margins

        state_update_reading = {
            "pronoun_coreference": "P_t / Q_t",
            "preposition_relation": "R_t / G_t",
            "adverb_scope": "M_t / Q_t",
            "logic_reasoning": "Q_t / R_t / C_t",
        }
        result = {
            "model_key": model_key,
            "model_label": MODEL_SPECS[model_key]["label"],
            "elapsed_seconds": round(time.time() - started, 3),
            "experiment_rows": experiment_rows,
            "state_update_reading": state_update_reading,
            "core_reading": (
                "如果四类最小实验能被同一套对象/关系/指代/修饰/推理状态词汇表解释，"
                "且逻辑 margin 在层间逐步升高，就说明统一状态更新方程开始具备实证支撑。"
            ),
        }
        print(f"[stage574] finished {model_key} in {time.time() - started:.2f}s", flush=True)
        return result
    finally:
        free_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# stage574 统一语言状态更新实测",
        "",
        "## 总结",
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
            if exp_name == "logic_reasoning":
                for margin_row in exp_row.get("layerwise_margin_rows", [])[:2]:
                    margins = margin_row["layer_margins"]
                    if margins:
                        lines.append(
                            f"  - 逻辑 margin 首层/末层：`{margins[0]:.4f}` / `{margins[-1]:.4f}`"
                        )
        lines.append("")
    return "\n".join(lines) + "\n"


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
    rows = [merged[key] for key in MODELS_IN_ORDER if key in merged]
    for key, row in merged.items():
        if key not in MODELS_IN_ORDER:
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一语言状态更新实测")
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS_IN_ORDER,
        help="按顺序运行的模型键",
    )
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
            print(f"[stage574] {model_key} failed: {exc!r}", flush=True)
            model_rows.append(
                {
                    "model_key": model_key,
                    "model_label": MODEL_SPECS[model_key]["label"],
                    "error": repr(exc),
                }
            )

    success_rows = [row for row in model_rows if "error" not in row]
    summary_path = OUTPUT_DIR / "summary.json"
    merged_rows = merge_existing_rows(summary_path, model_rows)
    success_rows = [row for row in merged_rows if "error" not in row]
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage574_unified_language_state_update_empirical",
        "title": "统一语言状态更新实测",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "models_in_order": MODELS_IN_ORDER,
        "model_rows": merged_rows,
        "core_answer": (
            "本实验用四类最小实验检验统一状态更新方程是否具备实证支撑。"
            "如果代词、介词、副词和逻辑推理都能在同一套状态变量上被解释，"
            "且逻辑结论不是只在末层突然出现，那么语言理论就开始从概念编码走向计算理论。"
        ),
        "support_count": len(success_rows),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(OUTPUT_DIR), "support_count": len(success_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
