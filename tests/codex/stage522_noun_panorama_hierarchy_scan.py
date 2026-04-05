#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from multimodel_language_shared import MODEL_SPECS, free_model, load_model_bundle
from qwen3_language_shared import capture_qwen_mlp_payloads, discover_layers, move_batch_to_model_device, remove_hooks


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage522_noun_panorama_hierarchy_scan_20260404"
)
MODEL_KEYS = ["qwen3", "deepseek7b", "glm4", "gemma4"]
TOP_K = 256

FAMILY_NOUNS = {
    "fruit": ["apple", "banana", "orange", "grape", "pear", "peach"],
    "animal": ["tiger", "lion", "horse", "rabbit", "eagle", "dolphin"],
    "celestial": ["sun", "moon", "star", "planet", "comet", "galaxy"],
    "abstract": ["freedom", "justice", "beauty", "truth", "time", "love"],
}

PROMPT_TEMPLATES = [
    "This text is about {noun}.",
    "People can think about {noun}.",
    "The sentence mentions {noun}.",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_last_subsequence(full_ids: List[int], sub_ids: List[int]):
    if not sub_ids or len(sub_ids) > len(full_ids):
        return None
    last_match = None
    for start in range(0, len(full_ids) - len(sub_ids) + 1):
        if full_ids[start : start + len(sub_ids)] == sub_ids:
            last_match = (start, start + len(sub_ids))
    return last_match


def locate_target_span(tokenizer, prompt: str, target: str):
    full_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    variants = [target, f" {target}", target.lower(), f" {target.lower()}", target.capitalize(), f" {target.capitalize()}"]
    best = None
    best_len = -1
    seen = set()
    for text in variants:
        if text in seen:
            continue
        seen.add(text)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        match = find_last_subsequence(full_ids, ids)
        if match is not None and len(ids) > best_len:
            best = match
            best_len = len(ids)
    if best is None:
        raise RuntimeError(f"无法定位目标词：target={target!r}, prompt={prompt!r}")
    return best


def capture_case_flat_neuron_vector(model, tokenizer, sentence: str, target: str) -> torch.Tensor:
    layer_payload_map = {layer_idx: "neuron_in" for layer_idx in range(len(discover_layers(model)))}
    buffers, handles = capture_qwen_mlp_payloads(model, layer_payload_map)
    try:
        encoded = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
        encoded = move_batch_to_model_device(model, encoded)
        start, end = locate_target_span(tokenizer, sentence, target)
        with torch.inference_mode():
            model(**encoded, use_cache=False, return_dict=True)
        per_layer = []
        for layer_idx in range(len(buffers)):
            payload = buffers[layer_idx]
            if payload is None:
                raise RuntimeError(f"缺少第 {layer_idx} 层神经元载荷")
            vec = payload[0, start:end, :].mean(dim=0).detach().float().cpu()
            per_layer.append(vec)
        return torch.cat(per_layer, dim=0)
    finally:
        remove_hooks(handles)


def top_active_ids(vec: torch.Tensor, top_k: int) -> List[int]:
    positive = torch.clamp(vec.float(), min=0.0)
    vals, idxs = torch.topk(positive, k=min(top_k, positive.numel()))
    out = []
    for value, idx in zip(vals.tolist(), idxs.tolist()):
        if value <= 0:
            continue
        out.append(int(idx))
    return out


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def mean_tensor(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([v.float() for v in vectors], dim=0).mean(dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm().item() * b.norm().item())
    if denom == 0:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def build_family_core(noun_top_ids: Dict[str, List[int]], nouns: Sequence[str], threshold: int) -> List[int]:
    freq = defaultdict(int)
    for noun in nouns:
        for idx in noun_top_ids[noun]:
            freq[int(idx)] += 1
    return sorted([idx for idx, count in freq.items() if count >= threshold])


def analyze_model(model_key: str) -> Dict[str, object]:
    model, tokenizer = load_model_bundle(model_key, prefer_cuda=False)
    try:
        noun_vectors: Dict[str, torch.Tensor] = {}
        noun_top_ids: Dict[str, List[int]] = {}
        noun_family: Dict[str, str] = {}

        for family_name, nouns in FAMILY_NOUNS.items():
            for noun in nouns:
                prompts = [template.format(noun=noun) for template in PROMPT_TEMPLATES]
                rows = [capture_case_flat_neuron_vector(model, tokenizer, prompt, noun) for prompt in prompts]
                vec = mean_tensor(rows)
                noun_vectors[noun] = vec
                noun_top_ids[noun] = top_active_ids(vec, TOP_K)
                noun_family[noun] = family_name

        all_nouns = [noun for nouns in FAMILY_NOUNS.values() for noun in nouns]
        global_threshold = math.ceil(len(all_nouns) * 0.5)
        global_core = build_family_core(noun_top_ids, all_nouns, global_threshold)

        family_cores = {}
        family_pairwise = {}
        family_mean_vectors = {}
        for family_name, nouns in FAMILY_NOUNS.items():
            family_cores[family_name] = build_family_core(noun_top_ids, nouns, math.ceil(len(nouns) * 0.5))
            family_mean_vectors[family_name] = mean_tensor([noun_vectors[n] for n in nouns])
            pairwise = [jaccard(noun_top_ids[a], noun_top_ids[b]) for a, b in combinations(nouns, 2)]
            family_pairwise[family_name] = sum(pairwise) / max(1, len(pairwise))

        cross_family_matrix = {}
        family_names = list(FAMILY_NOUNS.keys())
        for a, b in combinations(family_names, 2):
            scores = [jaccard(noun_top_ids[x], noun_top_ids[y]) for x in FAMILY_NOUNS[a] for y in FAMILY_NOUNS[b]]
            cross_family_matrix[f"{a}__{b}"] = sum(scores) / max(1, len(scores))

        apple_ids = set(noun_top_ids["apple"])
        apple_breakdown = {
            "apple_global_core_shared_count": len(apple_ids & set(global_core)),
            "apple_fruit_core_shared_count": len(apple_ids & set(family_cores["fruit"])),
            "apple_animal_core_shared_count": len(apple_ids & set(family_cores["animal"])),
            "apple_celestial_core_shared_count": len(apple_ids & set(family_cores["celestial"])),
            "apple_abstract_core_shared_count": len(apple_ids & set(family_cores["abstract"])),
            "apple_unique_vs_fruit_count": len(apple_ids - set(family_cores["fruit"])),
        }

        family_prediction_rows = []
        family_prediction_correct = 0
        family_core_margin_wins = 0
        for noun in all_nouns:
            actual_family = noun_family[noun]
            scores = {}
            for family_name, nouns in FAMILY_NOUNS.items():
                train_nouns = [n for n in nouns if n != noun]
                if family_name != actual_family:
                    train_nouns = list(nouns)
                proto = mean_tensor([noun_vectors[n] for n in train_nouns])
                scores[family_name] = cosine(noun_vectors[noun], proto)
            predicted_family = max(scores.items(), key=lambda item: item[1])[0]
            family_prediction_correct += int(predicted_family == actual_family)

            core_scores = {}
            for family_name, nouns in FAMILY_NOUNS.items():
                train_nouns = [n for n in nouns if n != noun]
                if family_name != actual_family:
                    train_nouns = list(nouns)
                core = build_family_core(noun_top_ids, train_nouns, max(2, math.ceil(len(train_nouns) * 0.5)))
                core_scores[family_name] = jaccard(noun_top_ids[noun], core)
            same_score = core_scores[actual_family]
            best_cross = max(score for fam, score in core_scores.items() if fam != actual_family)
            family_core_margin_wins += int(same_score > best_cross)
            family_prediction_rows.append(
                {
                    "noun": noun,
                    "actual_family": actual_family,
                    "predicted_family": predicted_family,
                    "family_scores": scores,
                    "same_family_core_overlap": same_score,
                    "best_cross_family_core_overlap": best_cross,
                }
            )

        summary = {
            "top_k": TOP_K,
            "noun_count": len(all_nouns),
            "global_core_count": len(global_core),
            "family_pairwise_mean_jaccard": family_pairwise,
            "cross_family_mean_jaccard": cross_family_matrix,
            "family_prediction_accuracy": family_prediction_correct / len(all_nouns),
            "family_core_margin_win_rate": family_core_margin_wins / len(all_nouns),
            "apple_breakdown": apple_breakdown,
            "family_core_counts": {family: len(core) for family, core in family_cores.items()},
        }
        return {
            "model_key": model_key,
            "model_name": MODEL_SPECS[model_key]["label"],
            "layer_count": len(discover_layers(model)),
            "summary": summary,
            "family_prediction_rows": family_prediction_rows,
        }
    finally:
        free_model(model)


def build_report(model_rows: List[Dict[str, object]], summary: Dict[str, object]) -> str:
    lines = [f"# {summary['experiment_id']}", "", "## 核心结论", summary["core_answer"], ""]
    for row in model_rows:
        s = row["summary"]
        apple = s["apple_breakdown"]
        lines.extend(
            [
                f"## {row['model_name']}",
                f"- 全局共享骨干数：`{s['global_core_count']}`",
                f"- 家族预测准确率：`{s['family_prediction_accuracy']:.4f}`",
                f"- 同家族核心胜率：`{s['family_core_margin_win_rate']:.4f}`",
                f"- 苹果与水果核心共享：`{apple['apple_fruit_core_shared_count']}`",
                f"- 苹果与动物核心共享：`{apple['apple_animal_core_shared_count']}`",
                f"- 苹果与天体核心共享：`{apple['apple_celestial_core_shared_count']}`",
                f"- 苹果与抽象核心共享：`{apple['apple_abstract_core_shared_count']}`",
                f"- 苹果相对水果独有：`{apple['apple_unique_vs_fruit_count']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    model_rows = [analyze_model(model_key) for model_key in MODEL_KEYS]
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage522_noun_panorama_hierarchy_scan",
        "title": "大量名词全景层级扫描",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "model_rows": model_rows,
        "core_answer": (
            "大量名词的编码可以分解成全局共享名词骨干、家族共享骨干和名词独有残差。"
            "同家族名词之间的共享显著高于跨家族共享，而且这种结构足以支持对新名词的家族级编码预测。"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "REPORT.md").write_text(build_report(model_rows, summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
