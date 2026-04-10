#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from multimodel_language_shared import MODEL_SPECS, candidate_score_map, free_model, load_model_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage576_reference_modifier_substate_empirical_20260409"
)
MODELS_IN_ORDER = ["qwen3", "deepseek7b", "glm4", "gemma4"]

EXPERIMENTS = {
    "pronoun_personal": [
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
    "pronoun_reflexive": [
        {
            "prompt": "Sentence: 'John saw himself in the mirror.' The word 'himself' refers to",
            "candidates": [" John", " mirror"],
            "correct": " John",
        },
        {
            "prompt": "Sentence: 'Mary blamed herself for the mistake.' The word 'herself' refers to",
            "candidates": [" Mary", " mistake"],
            "correct": " Mary",
        },
    ],
    "pronoun_demonstrative": [
        {
            "prompt": "Sentence: 'This apple is ripe.' The word 'This' mainly points to the",
            "candidates": [" object", " action", " relation"],
            "correct": " object",
        },
        {
            "prompt": "Sentence: 'That chair is broken.' The word 'That' mainly points to the",
            "candidates": [" object", " action", " relation"],
            "correct": " object",
        },
    ],
    "adverb_manner": [
        {
            "prompt": "Sentence: 'The boy quickly opened the door.' The word 'quickly' mainly modifies the",
            "candidates": [" action", " door", " boy"],
            "correct": " action",
        },
        {
            "prompt": "Sentence: 'The nurse carefully wrapped the bandage.' The word 'carefully' mainly modifies the",
            "candidates": [" action", " bandage", " nurse"],
            "correct": " action",
        },
    ],
    "adverb_epistemic": [
        {
            "prompt": "Sentence: 'The boy probably opened the door.' The word 'probably' mainly modifies the",
            "candidates": [" statement", " door", " boy"],
            "correct": " statement",
        },
        {
            "prompt": "Sentence: 'They ربما finished the work.' Replace 'ربما' with its meaning: it mainly modifies the",
            "candidates": [" statement", " work", " people"],
            "correct": " statement",
        },
    ],
    "adverb_degree": [
        {
            "prompt": "Sentence: 'The apple is very ripe.' The word 'very' mainly modifies the",
            "candidates": [" property", " apple", " statement"],
            "correct": " property",
        },
        {
            "prompt": "Sentence: 'The water is extremely cold.' The word 'extremely' mainly modifies the",
            "candidates": [" property", " water", " speaker"],
            "correct": " property",
        },
    ],
    "adverb_frequency": [
        {
            "prompt": "Sentence: 'The lights usually turn on at sunset.' The word 'usually' mainly modifies the",
            "candidates": [" event frequency", " lights", " place"],
            "correct": " event frequency",
        },
        {
            "prompt": "Sentence: 'We often meet near the library.' The word 'often' mainly modifies the",
            "candidates": [" event frequency", " library", " speaker"],
            "correct": " event frequency",
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


def merge_existing_rows(summary_path: Path, new_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
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
    return rows


def analyze_model(model_key: str) -> Dict[str, object]:
    started = time.time()
    print(f"[stage576] loading {model_key} ...", flush=True)
    model, tokenizer = load_model_bundle_safe(model_key)
    try:
        experiment_rows = {}
        for exp_name, cases in EXPERIMENTS.items():
            case_rows = [analyze_case(model, tokenizer, case) for case in cases]
            accuracy = float(sum(1 for row in case_rows if row["is_correct"]) / max(len(case_rows), 1))
            experiment_rows[exp_name] = {
                "accuracy": accuracy,
                "case_rows": case_rows,
            }

        pronoun_split_gain = float(
            (
                experiment_rows["pronoun_personal"]["accuracy"]
                + experiment_rows["pronoun_reflexive"]["accuracy"]
                + experiment_rows["pronoun_demonstrative"]["accuracy"]
            )
            / 3.0
        )
        adverb_split_gain = float(
            (
                experiment_rows["adverb_manner"]["accuracy"]
                + experiment_rows["adverb_epistemic"]["accuracy"]
                + experiment_rows["adverb_degree"]["accuracy"]
                + experiment_rows["adverb_frequency"]["accuracy"]
            )
            / 4.0
        )

        row = {
            "model_key": model_key,
            "model_label": MODEL_SPECS[model_key]["label"],
            "elapsed_seconds": round(time.time() - started, 3),
            "experiment_rows": experiment_rows,
            "pronoun_split_mean_accuracy": pronoun_split_gain,
            "adverb_split_mean_accuracy": adverb_split_gain,
            "reading": (
                "如果拆分后的子状态显著优于粗粒度 P_t / M_t，"
                "就说明当前统一理论需要从‘一个指代项 + 一个修饰项’升级为多子状态结构。"
            ),
        }
        print(f"[stage576] finished {model_key} in {time.time() - started:.2f}s", flush=True)
        return row
    finally:
        free_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# stage576 指代/修饰子状态拆分实测",
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
        lines.append(f"- pronoun_split_mean_accuracy: `{row['pronoun_split_mean_accuracy']:.4f}`")
        lines.append(f"- adverb_split_mean_accuracy: `{row['adverb_split_mean_accuracy']:.4f}`")
        for exp_name, exp_row in row["experiment_rows"].items():
            lines.append(f"- `{exp_name}`: `{exp_row['accuracy']:.4f}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="指代/修饰子状态拆分实测")
    parser.add_argument("--models", nargs="+", default=MODELS_IN_ORDER, help="按顺序运行的模型键")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    model_rows: List[Dict[str, object]] = []
    for model_key in args.models:
        try:
            model_rows.append(analyze_model(model_key))
        except Exception as exc:
            print(f"[stage576] {model_key} failed: {exc!r}", flush=True)
            model_rows.append(
                {
                    "model_key": model_key,
                    "model_label": MODEL_SPECS[model_key]["label"],
                    "error": repr(exc),
                }
            )

    summary_path = OUTPUT_DIR / "summary.json"
    merged_rows = merge_existing_rows(summary_path, model_rows)
    success_rows = [row for row in merged_rows if "error" not in row]
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage576_reference_modifier_substate_empirical",
        "title": "指代/修饰子状态拆分实测",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "models_in_order": MODELS_IN_ORDER,
        "model_rows": merged_rows,
        "core_answer": (
            "本实验检验把 P_t 拆成人称/反身/指示，把 M_t 拆成方式/认识论/程度/频率后，"
            "是否能更稳定地承载代词与副词机制。若成立，统一状态更新方程应升级为多子状态版本。"
        ),
        "support_count": len(success_rows),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(OUTPUT_DIR), "support_count": len(success_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
