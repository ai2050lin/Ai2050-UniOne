#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import gc
import json
import math
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
    / "stage573_fruit_minimal_causal_encoding_empirical_20260409"
)

MODELS_IN_ORDER = ["qwen3", "deepseek7b", "glm4", "gemma4"]
PROMPT_TEMPLATES = [
    "This is an {word}.",
    "I saw a {word} today.",
]
NOUN_GROUPS = {
    "fruit": ["apple", "banana", "pear", "orange"],
    "animal": ["cat", "dog", "horse", "rabbit"],
    "object": ["car", "chair", "table", "boat"],
}
ATTRIBUTE_CASES = {
    "red": [("apple", "red apple"), ("car", "red car"), ("chair", "red chair")],
    "sweet": [("apple", "sweet apple"), ("banana", "sweet banana"), ("tea", "sweet tea")],
}
PROBE_PROMPTS = [
    ("An apple is a type of", [" fruit", " animal", " object"]),
    ("A banana is a type of", [" fruit", " animal", " object"]),
    ("A pear is usually", [" sweet", " metal", " loud"]),
]


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


def l2_norm(vec: torch.Tensor) -> float:
    return float(torch.linalg.norm(vec).item())


def safe_cos(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    denom = torch.linalg.norm(vec_a) * torch.linalg.norm(vec_b)
    if float(denom.item()) < 1e-8:
        return 0.0
    return float(torch.dot(vec_a, vec_b).item() / denom.item())


def mean_tensor(rows: Sequence[torch.Tensor]) -> torch.Tensor:
    if not rows:
        raise ValueError("empty tensor list")
    return torch.stack([row.float().cpu() for row in rows], dim=0).mean(dim=0)


def encode_hidden(model, tokenizer, text: str) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=128)
    encoded = move_batch_to_model_device(model, encoded)
    with torch.inference_mode():
        out = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden_states = [layer[0, -1, :].detach().float().cpu() for layer in out.hidden_states]
    return {
        "last": hidden_states[-1],
        "layers": torch.stack(hidden_states, dim=0),
    }


def concept_mean(model, tokenizer, word: str) -> Dict[str, torch.Tensor]:
    rows_last = []
    rows_layers = []
    for template in PROMPT_TEMPLATES:
        rec = encode_hidden(model, tokenizer, template.format(word=word))
        rows_last.append(rec["last"])
        rows_layers.append(rec["layers"])
    return {
        "last": mean_tensor(rows_last),
        "layers": mean_tensor(rows_layers),
    }


def pairwise_mean_cos(vectors: Sequence[torch.Tensor]) -> float:
    scores: List[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            scores.append(safe_cos(vectors[i], vectors[j]))
    return float(sum(scores) / max(len(scores), 1))


def analyze_model(model_key: str) -> Dict[str, object]:
    started = time.time()
    print(f"[stage573] loading {model_key} ...", flush=True)
    model, tokenizer = load_model_bundle_safe(model_key)
    try:
        concept_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        all_nouns = sorted({w for group in NOUN_GROUPS.values() for w in group})
        extra_words = sorted({noun for pairs in ATTRIBUTE_CASES.values() for noun, _ in pairs} | {"tea"})
        for word in sorted(set(all_nouns + extra_words)):
            concept_cache[word] = concept_mean(model, tokenizer, word)

        global_mean = mean_tensor([concept_cache[word]["last"] for word in all_nouns])
        fruit_mean = mean_tensor([concept_cache[word]["last"] for word in NOUN_GROUPS["fruit"]])
        animal_mean = mean_tensor([concept_cache[word]["last"] for word in NOUN_GROUPS["animal"]])
        object_mean = mean_tensor([concept_cache[word]["last"] for word in NOUN_GROUPS["object"]])
        b_fruit = fruit_mean - global_mean
        b_animal = animal_mean - global_mean
        b_object = object_mean - global_mean

        fruit_rows = {}
        for word in NOUN_GROUPS["fruit"]:
            base = concept_cache[word]["last"]
            offset = base - global_mean - b_fruit
            fruit_rows[word] = {
                "concept_offset_norm": l2_norm(offset),
                "concept_offset_ratio": float(l2_norm(offset) / max(l2_norm(base - global_mean), 1e-8)),
                "cos_to_fruit_backbone": safe_cos(base, b_fruit),
                "cos_to_animal_backbone": safe_cos(base, b_animal),
                "cos_to_object_backbone": safe_cos(base, b_object),
                "family_margin_vs_animal": safe_cos(base, fruit_mean) - safe_cos(base, animal_mean),
                "family_margin_vs_object": safe_cos(base, fruit_mean) - safe_cos(base, object_mean),
            }

        attribute_rows = {}
        for attr_name, pairs in ATTRIBUTE_CASES.items():
            deltas = []
            combo_norms = []
            residual_norms = []
            combo_scores = []
            for noun, combo in pairs:
                combo_vec = concept_mean(model, tokenizer, combo)["last"]
                noun_vec = concept_cache[noun]["last"]
                delta = combo_vec - noun_vec
                deltas.append(delta)
                combo_norms.append(l2_norm(combo_vec))
            attr_channel = mean_tensor(deltas)
            for (noun, combo), delta in zip(pairs, deltas):
                combo_vec = concept_mean(model, tokenizer, combo)["last"]
                residual = combo_vec - concept_cache[noun]["last"] - attr_channel
                residual_norms.append(l2_norm(residual))
                combo_scores.append(
                    {
                        "noun": noun,
                        "combo": combo,
                        "delta_norm": l2_norm(delta),
                        "binding_residual_ratio": float(l2_norm(residual) / max(l2_norm(combo_vec), 1e-8)),
                        "cos_delta_to_attr_channel": safe_cos(delta, attr_channel),
                    }
                )
            attribute_rows[attr_name] = {
                "attr_channel_norm": l2_norm(attr_channel),
                "shared_delta_cos_mean": pairwise_mean_cos(deltas),
                "mean_binding_residual_ratio": float(sum(residual_norms[i] / max(combo_norms[i], 1e-8) for i in range(len(combo_norms))) / max(len(combo_norms), 1)),
                "pair_rows": combo_scores,
            }

        layerwise_apple_fruit = []
        layerwise_apple_animal = []
        for layer_idx in range(concept_cache["apple"]["layers"].shape[0]):
            apple_l = concept_cache["apple"]["layers"][layer_idx]
            fruit_l = mean_tensor([concept_cache[word]["layers"][layer_idx] for word in NOUN_GROUPS["fruit"]])
            animal_l = mean_tensor([concept_cache[word]["layers"][layer_idx] for word in NOUN_GROUPS["animal"]])
            layerwise_apple_fruit.append(safe_cos(apple_l, fruit_l))
            layerwise_apple_animal.append(safe_cos(apple_l, animal_l))

        probe_rows = []
        for prompt, candidates in PROBE_PROMPTS:
            score_map = candidate_score_map(model, tokenizer, prompt, candidates)
            ranked = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
            probe_rows.append(
                {
                    "prompt": prompt,
                    "scores": score_map,
                    "best_candidate": ranked[0][0],
                    "margin_top1_top2": float(ranked[0][1] - ranked[1][1]) if len(ranked) >= 2 else 0.0,
                }
            )

        result = {
            "model_key": model_key,
            "model_label": MODEL_SPECS[model_key]["label"],
            "elapsed_seconds": round(time.time() - started, 3),
            "global_norm": l2_norm(global_mean),
            "fruit_backbone_norm": l2_norm(b_fruit),
            "animal_backbone_norm": l2_norm(b_animal),
            "object_backbone_norm": l2_norm(b_object),
            "fruit_rows": fruit_rows,
            "attribute_rows": attribute_rows,
            "layerwise_apple_fruit_cos": [float(x) for x in layerwise_apple_fruit],
            "layerwise_apple_animal_cos": [float(x) for x in layerwise_apple_animal],
            "probe_rows": probe_rows,
            "core_reading": (
                "苹果若显著更靠近水果骨干、概念偏置相对较小、属性 delta 在多对象上可复用且组合残差非零，"
                "则支持“水果骨干 + 概念偏置 + 属性通道 + 绑定桥接”这套最小因果结构。"
            ),
        }
        print(f"[stage573] finished {model_key} in {time.time() - started:.2f}s", flush=True)
        return result
    finally:
        free_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# stage573 水果最小因果编码结构实测",
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
        lines.append(f"- 水果骨干范数：`{row['fruit_backbone_norm']:.4f}`")
        lines.append(
            f"- apple 对水果/动物/物体骨干余弦：`{row['fruit_rows']['apple']['cos_to_fruit_backbone']:.4f}` / "
            f"`{row['fruit_rows']['apple']['cos_to_animal_backbone']:.4f}` / "
            f"`{row['fruit_rows']['apple']['cos_to_object_backbone']:.4f}`"
        )
        lines.append(
            f"- apple 概念偏置比例：`{row['fruit_rows']['apple']['concept_offset_ratio']:.4f}`"
        )
        for attr_name, attr_row in row["attribute_rows"].items():
            lines.append(
                f"- `{attr_name}` 属性共享余弦/绑定残差：`{attr_row['shared_delta_cos_mean']:.4f}` / "
                f"`{attr_row['mean_binding_residual_ratio']:.4f}`"
            )
        lines.append(
            f"- apple 末层 fruit-animal 分离：`{row['layerwise_apple_fruit_cos'][-1] - row['layerwise_apple_animal_cos'][-1]:.4f}`"
        )
        for probe in row["probe_rows"]:
            lines.append(
                f"- `{probe['prompt']}` → `{probe['best_candidate']}` (top1-top2=`{probe['margin_top1_top2']:.4f}`)"
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
    parser = argparse.ArgumentParser(description="水果最小因果编码结构实测")
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
            print(f"[stage573] {model_key} failed: {exc!r}", flush=True)
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
        "experiment_id": "stage573_fruit_minimal_causal_encoding_empirical",
        "title": "水果最小因果编码结构实测",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "models_in_order": MODELS_IN_ORDER,
        "model_rows": merged_rows,
        "core_answer": (
            "本实验用四模型串行检验水果骨干、概念偏置、属性通道与绑定残差四个可观测量。"
            "如果苹果在结构上稳定表现为“更靠近水果骨干 + 偏置较小 + 属性可迁移 + 绑定残差非零”，"
            "则最小因果编码结构得到第一轮实证支持。"
        ),
        "support_count": len(success_rows),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(OUTPUT_DIR), "support_count": len(success_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
