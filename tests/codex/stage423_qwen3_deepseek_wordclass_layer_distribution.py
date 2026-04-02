#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3_language_shared import (
    PROJECT_ROOT,
    QWEN3_MODEL_PATH,
    capture_qwen_mlp_payloads,
    discover_layers,
    move_batch_to_model_device,
    remove_hooks,
)
from stage140_deepseek_language_validation_suite import DEEPSEEK_MODEL_PATH


OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage423_qwen3_deepseek_wordclass_layer_distribution_20260330"
)

QWEN_ROWS_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage139_qwen3_language_validation_suite_20260323"
    / "qwen_word_rows.jsonl"
)
DEEPSEEK_ROWS_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage140_deepseek_language_validation_suite_20260323"
    / "deepseek_word_rows.jsonl"
)

CLEAN_NOUNS_PATH = PROJECT_ROOT / "tests" / "codex" / "deepseek7b_nouns_english_520_clean.csv"

MODEL_SPECS = {
    "qwen3": {
        "model_name": "Qwen/Qwen3-4B",
        "model_path": QWEN3_MODEL_PATH,
        "rows_path": QWEN_ROWS_PATH,
    },
    "deepseek7b": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "model_path": DEEPSEEK_MODEL_PATH,
        "rows_path": DEEPSEEK_ROWS_PATH,
    },
}

WORD_CLASSES = ["noun", "adjective", "verb", "adverb", "pronoun", "preposition"]
TOP_FRACTION = 0.01
EPS = 1e-8

DEFAULT_TARGET_LIMITS = {
    "noun": 384,
    "adjective": 256,
    "verb": 256,
    "adverb": 192,
    "pronoun": 48,
    "preposition": 64,
}

DEFAULT_CONTROL_LIMITS = {
    "noun": 480,
    "adjective": 480,
    "verb": 480,
    "adverb": 480,
    "pronoun": 480,
    "preposition": 480,
}

PRONOUN_WORDS = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "this",
    "that",
    "these",
    "those",
    "who",
    "whom",
    "whose",
    "which",
    "what",
    "someone",
    "somebody",
    "something",
    "anyone",
    "anybody",
    "anything",
    "everyone",
    "everybody",
    "everything",
    "nobody",
    "nothing",
    "another",
    "other",
    "others",
    "each",
    "either",
    "neither",
}

PREPOSITION_WORDS = {
    "about",
    "above",
    "across",
    "after",
    "against",
    "along",
    "among",
    "around",
    "as",
    "at",
    "before",
    "behind",
    "below",
    "beneath",
    "beside",
    "between",
    "beyond",
    "by",
    "despite",
    "down",
    "during",
    "except",
    "for",
    "from",
    "in",
    "inside",
    "into",
    "like",
    "near",
    "of",
    "off",
    "on",
    "onto",
    "over",
    "past",
    "per",
    "since",
    "through",
    "throughout",
    "to",
    "toward",
    "towards",
    "under",
    "underneath",
    "until",
    "up",
    "upon",
    "via",
    "with",
    "within",
    "without",
}

ADVERB_ALLOWLIST = {
    "not",
    "never",
    "always",
    "often",
    "sometimes",
    "maybe",
    "perhaps",
    "quite",
    "rather",
    "very",
    "almost",
    "already",
    "still",
    "then",
    "thus",
    "therefore",
    "however",
    "instead",
    "together",
    "apart",
    "else",
}

ADJECTIVE_BLOCKLIST = {
    "another",
    "other",
    "either",
    "neither",
    "every",
    "some",
    "any",
    "many",
    "much",
    "few",
    "little",
}


def set_offline_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_word_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def load_qwen_like_model(
    model_path: Path,
    *,
    prefer_cuda: bool,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    set_offline_env()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    want_cuda = bool(prefer_cuda and torch.cuda.is_available())
    load_kwargs = {
        "pretrained_model_name_or_path": str(model_path),
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "dtype": torch.bfloat16,
    }
    if want_cuda:
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "cpu"
        load_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()
    return model, tokenizer


def normalize_word(word: str) -> str:
    return word.strip().lower()


def load_clean_word_column(path: Path) -> set[str]:
    words: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            head = normalize_word(row[0])
            if not head or head.startswith("#"):
                continue
            words.add(head)
    return words


CLEAN_NOUN_WORDS = load_clean_word_column(CLEAN_NOUNS_PATH)


def is_ascii_alpha_word(word: str) -> bool:
    return word.isascii() and word.isalpha()


def is_repetitive_noise(word: str) -> bool:
    if len(word) >= 4 and len(set(word)) <= 2:
        return True
    if len(word) >= 5 and word[:2] * (len(word) // 2) == word[: 2 * (len(word) // 2)]:
        return True
    return False


def classify_word(row: Dict[str, object]) -> str | None:
    word = normalize_word(str(row["word"]))
    lexical_type = str(row.get("lexical_type", ""))
    if word in PRONOUN_WORDS:
        return "pronoun"
    if word in PREPOSITION_WORDS:
        return "preposition"
    if lexical_type in {"noun", "adjective", "verb", "adverb"}:
        return lexical_type
    return None


def candidate_ok(row: Dict[str, object], class_name: str) -> bool:
    word = normalize_word(str(row["word"]))
    lexical_type = str(row.get("lexical_type", ""))
    score = float(row.get("lexical_type_score", 0.0))
    effective = float(row.get("effective_encoding_score", 0.0))

    if not word:
        return False
    if class_name in {"noun", "adjective", "verb", "adverb"}:
        if lexical_type != class_name:
            return False
    if class_name in {"pronoun", "preposition"} and lexical_type != "function":
        return False
    if not is_ascii_alpha_word(word):
        return False
    if is_repetitive_noise(word):
        return False

    if class_name == "noun":
        return (
            word in CLEAN_NOUN_WORDS
            and len(word) >= 3
            and lexical_type == "noun"
            and score >= 0.05
            and effective >= 0.30
        )
    if class_name == "adjective":
        return len(word) >= 3 and word not in ADJECTIVE_BLOCKLIST and score >= 0.08 and effective >= 0.36
    if class_name == "verb":
        return len(word) >= 3 and score >= 0.08 and effective >= 0.34
    if class_name == "adverb":
        return (word.endswith("ly") or word in ADVERB_ALLOWLIST) and score >= 0.07 and effective >= 0.32
    if class_name == "pronoun":
        return word in PRONOUN_WORDS and effective >= 0.30
    if class_name == "preposition":
        return word in PREPOSITION_WORDS and effective >= 0.30
    return False


def sort_key(row: Dict[str, object]) -> Tuple[float, float, float]:
    return (
        float(row.get("lexical_type_score", 0.0)),
        float(row.get("effective_encoding_score", 0.0)),
        float(row.get("group_score", 0.0)),
    )


def dedupe_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    best_by_word: Dict[str, Dict[str, object]] = {}
    for row in rows:
        word = normalize_word(str(row["word"]))
        prev = best_by_word.get(word)
        if prev is None or sort_key(row) > sort_key(prev):
            best_by_word[word] = row
    return list(best_by_word.values())


def build_target_rows(
    rows: Sequence[Dict[str, object]],
    class_name: str,
    limit: int,
) -> List[Dict[str, object]]:
    selected = []
    for row in rows:
        if classify_word(row) != class_name:
            continue
        if not candidate_ok(row, class_name):
            continue
        selected.append(row)
    deduped = dedupe_rows(selected)
    deduped.sort(key=sort_key, reverse=True)
    return deduped[:limit]


def build_control_rows(
    rows: Sequence[Dict[str, object]],
    target_class: str,
    target_words: Sequence[str],
    limit: int,
) -> List[Dict[str, object]]:
    per_class_limit = max(24, limit // max(1, len(WORD_CLASSES) - 1))
    target_word_set = set(target_words)
    buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        class_name = classify_word(row)
        if class_name is None or class_name == target_class:
            continue
        word = normalize_word(str(row["word"]))
        if word in target_word_set:
            continue
        if not candidate_ok(row, class_name):
            continue
        buckets[class_name].append(row)

    out: List[Dict[str, object]] = []
    for class_name in WORD_CLASSES:
        if class_name == target_class:
            continue
        deduped = dedupe_rows(buckets[class_name])
        deduped.sort(key=sort_key, reverse=True)
        out.extend(deduped[:per_class_limit])
    out = dedupe_rows(out)
    out.sort(key=sort_key, reverse=True)
    return out[:limit]


def mean_token_activation(layer_tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(layer_tensor.dtype)
    lengths = attention_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(layer_tensor.dtype)
    return (layer_tensor * mask).sum(dim=1) / lengths


def capture_all_layers(model) -> Tuple[Dict[int, torch.Tensor | None], List[object]]:
    layer_count = len(discover_layers(model))
    layer_payload_map = {layer_idx: "neuron_in" for layer_idx in range(layer_count)}
    return capture_qwen_mlp_payloads(model, layer_payload_map)


def init_stats(layer_count: int, neuron_count: int) -> Dict[str, object]:
    shape = (layer_count, neuron_count)
    return {
        "target_count": 0,
        "control_count": 0,
        "target_sum": torch.zeros(shape, dtype=torch.float64),
        "target_sumsq": torch.zeros(shape, dtype=torch.float64),
        "target_pos": torch.zeros(shape, dtype=torch.float64),
        "control_sum": torch.zeros(shape, dtype=torch.float64),
        "control_sumsq": torch.zeros(shape, dtype=torch.float64),
        "control_pos": torch.zeros(shape, dtype=torch.float64),
    }


def update_stats(
    stats: Dict[str, object],
    sample_tensor: torch.Tensor,
    *,
    is_target: bool,
) -> None:
    prefix = "target" if is_target else "control"
    stats[f"{prefix}_sum"] += sample_tensor.sum(dim=0)
    stats[f"{prefix}_sumsq"] += (sample_tensor * sample_tensor).sum(dim=0)
    stats[f"{prefix}_pos"] += (sample_tensor > 0).to(torch.float64).sum(dim=0)
    stats[f"{prefix}_count"] += int(sample_tensor.shape[0])


def process_batch(
    model,
    tokenizer,
    buffers: Dict[int, torch.Tensor | None],
    batch_rows: Sequence[Dict[str, object]],
    *,
    stats: Dict[str, object],
    is_target: bool,
) -> None:
    batch_texts = [str(row["word"]) for row in batch_rows]
    encoded = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    attention_mask_cpu = encoded["attention_mask"].cpu()
    encoded = move_batch_to_model_device(model, encoded)
    with torch.inference_mode():
        model(**encoded, use_cache=False, return_dict=True)

    per_layer_rows: List[torch.Tensor] = []
    for layer_idx in range(len(buffers)):
        layer_tensor = buffers[layer_idx]
        if layer_tensor is None:
            raise RuntimeError(f"第 {layer_idx} 层激活捕获失败，存在空缓冲")
        per_layer_rows.append(mean_token_activation(layer_tensor, attention_mask_cpu).to(torch.float64))
    sample_tensor = torch.stack(per_layer_rows, dim=1)
    update_stats(stats, sample_tensor, is_target=is_target)


def run_scan(
    model,
    tokenizer,
    target_rows: Sequence[Dict[str, object]],
    control_rows: Sequence[Dict[str, object]],
    *,
    batch_size: int,
) -> Dict[str, object]:
    buffers, handles = capture_all_layers(model)
    first_layer = discover_layers(model)[0]
    if hasattr(first_layer.mlp, "gate_proj"):
        neuron_count = int(first_layer.mlp.gate_proj.out_features)
    else:
        raise RuntimeError("未识别到 Qwen 类 MLP gate_proj 结构")
    stats = init_stats(len(buffers), neuron_count)

    try:
        for start in range(0, len(target_rows), batch_size):
            process_batch(
                model,
                tokenizer,
                buffers,
                target_rows[start : start + batch_size],
                stats=stats,
                is_target=True,
            )
        for start in range(0, len(control_rows), batch_size):
            process_batch(
                model,
                tokenizer,
                buffers,
                control_rows[start : start + batch_size],
                stats=stats,
                is_target=False,
            )
    finally:
        remove_hooks(handles)

    return stats


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_layer_summary(
    stats: Dict[str, object],
    *,
    top_fraction: float,
) -> Dict[str, object]:
    target_count = float(stats["target_count"])
    control_count = float(stats["control_count"])

    target_mean = stats["target_sum"] / max(target_count, 1.0)
    control_mean = stats["control_sum"] / max(control_count, 1.0)
    target_var = stats["target_sumsq"] / max(target_count, 1.0) - target_mean * target_mean
    control_var = stats["control_sumsq"] / max(control_count, 1.0) - control_mean * control_mean
    pooled_std = torch.sqrt(((target_var.clamp_min(0.0) + control_var.clamp_min(0.0)) / 2.0).clamp_min(EPS))

    effect = (target_mean - control_mean) / pooled_std
    pos_gap = stats["target_pos"] / max(target_count, 1.0) - stats["control_pos"] / max(control_count, 1.0)
    diff = target_mean - control_mean

    score = (
        0.70 * torch.clamp(effect / 3.0, 0.0, 1.0)
        + 0.30 * torch.clamp(pos_gap / 0.50, 0.0, 1.0)
    )
    active_mask = (diff > 0) & (score > 0)
    active_scores = score[active_mask]
    if active_scores.numel() == 0:
        threshold = 0.0
        effective_mask = active_mask
    else:
        quantile = max(0.0, min(1.0, 1.0 - top_fraction))
        threshold = float(torch.quantile(active_scores, quantile).item())
        effective_mask = active_mask & (score >= threshold)

    layer_rows = []
    score_mass_total = float((score * effective_mask.to(score.dtype)).sum().item()) + EPS
    count_total = int(effective_mask.to(torch.int64).sum().item())

    flat_score = score.flatten()
    top_k = min(20, flat_score.numel())
    top_values, top_indices = torch.topk(flat_score, k=top_k)
    neuron_count = score.shape[1]
    top_neurons = []
    for rank, flat_idx in enumerate(top_indices.tolist(), start=1):
        layer_idx = flat_idx // neuron_count
        neuron_idx = flat_idx % neuron_count
        top_neurons.append(
            {
                "rank": rank,
                "layer_index": int(layer_idx),
                "neuron_index": int(neuron_idx),
                "score": float(score[layer_idx, neuron_idx].item()),
                "effect_size": float(effect[layer_idx, neuron_idx].item()),
                "positive_rate_gap": float(pos_gap[layer_idx, neuron_idx].item()),
                "target_mean_activation": float(target_mean[layer_idx, neuron_idx].item()),
                "control_mean_activation": float(control_mean[layer_idx, neuron_idx].item()),
                "is_effective": bool(effective_mask[layer_idx, neuron_idx].item()),
            }
        )

    for layer_idx in range(score.shape[0]):
        layer_score = score[layer_idx]
        layer_effective = effective_mask[layer_idx]
        effective_count = int(layer_effective.to(torch.int64).sum().item())
        effective_fraction = effective_count / max(1, score.shape[1])
        effective_score_sum = float((layer_score * layer_effective.to(layer_score.dtype)).sum().item())
        top_values, _ = torch.topk(layer_score, k=min(32, layer_score.numel()))
        layer_rows.append(
            {
                "layer_index": int(layer_idx),
                "effective_count": effective_count,
                "effective_fraction": float(effective_fraction),
                "effective_score_sum": effective_score_sum,
                "effective_score_mass_share": float(effective_score_sum / score_mass_total),
                "effective_count_share": float(effective_count / max(1, count_total)),
                "mean_score": float(layer_score.mean().item()),
                "top32_mean_score": float(top_values.mean().item()),
                "max_score": float(layer_score.max().item()),
                "mean_effect_size": float(effect[layer_idx].mean().item()),
                "positive_effect_rate": float((effect[layer_idx] > 0).to(torch.float64).mean().item()),
                "positive_pos_gap_rate": float((pos_gap[layer_idx] > 0).to(torch.float64).mean().item()),
            }
        )

    layer_rows_by_count = sorted(layer_rows, key=lambda row: row["effective_count"], reverse=True)
    layer_rows_by_mass = sorted(layer_rows, key=lambda row: row["effective_score_mass_share"], reverse=True)

    thirds = max(1, score.shape[0] // 3)
    early_end = thirds
    mid_end = min(score.shape[0], 2 * thirds)
    band_ranges = {
        "early": range(0, early_end),
        "middle": range(early_end, mid_end),
        "late": range(mid_end, score.shape[0]),
    }
    band_summary = {}
    for band_name, layer_range in band_ranges.items():
        rows_in_band = [layer_rows[idx] for idx in layer_range]
        band_summary[band_name] = {
            "layer_indices": [row["layer_index"] for row in rows_in_band],
            "effective_count": int(sum(row["effective_count"] for row in rows_in_band)),
            "effective_score_mass_share": float(sum(row["effective_score_mass_share"] for row in rows_in_band)),
        }

    weighted_center = sum(
        row["layer_index"] * row["effective_score_mass_share"] for row in layer_rows
    )

    return {
        "target_count": int(target_count),
        "control_count": int(control_count),
        "layer_count": int(score.shape[0]),
        "neurons_per_layer": int(score.shape[1]),
        "top_fraction": float(top_fraction),
        "effective_score_threshold": float(threshold),
        "effective_neuron_count": int(count_total),
        "weighted_layer_center": float(weighted_center),
        "top_layers_by_count": layer_rows_by_count[:5],
        "top_layers_by_mass": layer_rows_by_mass[:5],
        "band_summary": band_summary,
        "layer_rows": layer_rows,
        "top_neurons": top_neurons,
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 实验设置",
        f"- 时间戳: {summary['timestamp_utc']}",
        f"- 是否使用 CUDA: {summary['used_cuda']}",
        f"- 批大小: {summary['batch_size']}",
        f"- 有效神经元定义: 取正向神经元得分前 {summary['top_fraction'] * 100:.2f}%",
        "",
    ]

    for model_name, model_payload in summary["models"].items():
        lines.extend(
            [
                f"## 模型 {model_name}",
                f"- 模型名称: {model_payload['model_name']}",
                f"- 层数: {model_payload['layer_count']}",
                f"- 每层神经元数: {model_payload['neurons_per_layer']}",
                "",
            ]
        )
        for class_name in WORD_CLASSES:
            class_payload = model_payload["classes"][class_name]
            top_by_mass = class_payload["top_layers_by_mass"][:3]
            top_text = ", ".join(
                f"L{row['layer_index']}({row['effective_score_mass_share']:.3f})" for row in top_by_mass
            )
            lines.extend(
                [
                    f"### {class_name}",
                    f"- 目标样本数: {class_payload['target_count']}",
                    f"- 控制样本数: {class_payload['control_count']}",
                    f"- 有效神经元数: {class_payload['effective_neuron_count']}",
                    f"- 质心层: {class_payload['weighted_layer_center']:.2f}",
                    f"- 主导层: {top_text}",
                    f"- 早/中/后层占比: "
                    f"{class_payload['band_summary']['early']['effective_score_mass_share']:.3f} / "
                    f"{class_payload['band_summary']['middle']['effective_score_mass_share']:.3f} / "
                    f"{class_payload['band_summary']['late']['effective_score_mass_share']:.3f}",
                    "",
                ]
            )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "REPORT.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    report_path.write_text(build_report(summary), encoding="utf-8-sig")


def free_model(model) -> None:
    try:
        del model
    except UnboundLocalError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def analyze_model(
    *,
    model_key: str,
    batch_size: int,
    target_limits: Dict[str, int],
    control_limits: Dict[str, int],
    use_cuda: bool,
) -> Dict[str, object]:
    model_spec = MODEL_SPECS[model_key]
    rows = load_word_rows(model_spec["rows_path"])
    model, tokenizer = load_qwen_like_model(model_spec["model_path"], prefer_cuda=use_cuda)
    try:
        layers = discover_layers(model)
        out = {
            "model_name": model_spec["model_name"],
            "model_path": str(model_spec["model_path"]),
            "layer_count": len(layers),
            "neurons_per_layer": (
                int(layers[0].mlp.gate_proj.out_features)
                if hasattr(layers[0].mlp, "gate_proj")
                else None
            ),
            "classes": {},
        }
        for class_name in WORD_CLASSES:
            target_rows = build_target_rows(rows, class_name, target_limits[class_name])
            if not target_rows:
                raise RuntimeError(f"{model_key} 的 {class_name} 没有可用目标样本")
            control_rows = build_control_rows(
                rows,
                class_name,
                [normalize_word(str(row["word"])) for row in target_rows],
                control_limits[class_name],
            )
            if not control_rows:
                raise RuntimeError(f"{model_key} 的 {class_name} 没有可用控制样本")
            stats = run_scan(
                model,
                tokenizer,
                target_rows,
                control_rows,
                batch_size=batch_size,
            )
            class_summary = build_layer_summary(stats, top_fraction=TOP_FRACTION)
            class_summary["target_words"] = [str(row["word"]) for row in target_rows[:40]]
            class_summary["control_class_counts"] = {
                other_class: sum(1 for row in control_rows if classify_word(row) == other_class)
                for other_class in WORD_CLASSES
                if other_class != class_name
            }
            out["classes"][class_name] = class_summary
        return out
    finally:
        free_model(model)


def build_cross_model_summary(model_payloads: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    comparisons = {}
    for class_name in WORD_CLASSES:
        qwen_center = model_payloads["qwen3"]["classes"][class_name]["weighted_layer_center"]
        deepseek_center = model_payloads["deepseek7b"]["classes"][class_name]["weighted_layer_center"]
        comparisons[class_name] = {
            "qwen3_weighted_layer_center": qwen_center,
            "deepseek7b_weighted_layer_center": deepseek_center,
            "center_shift_deepseek_minus_qwen": deepseek_center - qwen_center,
            "qwen3_top_layers_by_mass": [
                row["layer_index"]
                for row in model_payloads["qwen3"]["classes"][class_name]["top_layers_by_mass"][:5]
            ],
            "deepseek7b_top_layers_by_mass": [
                row["layer_index"]
                for row in model_payloads["deepseek7b"]["classes"][class_name]["top_layers_by_mass"][:5]
            ],
        }
    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3 与 DeepSeek7B 六类词性有效神经元层分布分析")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="输出目录",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="前向批大小",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制不用 CUDA",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    use_cuda = (not args.cpu) and torch.cuda.is_available()

    start_time = time.time()
    model_payloads = {}
    for model_key in ["qwen3", "deepseek7b"]:
        model_payloads[model_key] = analyze_model(
            model_key=model_key,
            batch_size=args.batch_size,
            target_limits=DEFAULT_TARGET_LIMITS,
            control_limits=DEFAULT_CONTROL_LIMITS,
            use_cuda=use_cuda,
        )

    elapsed = time.time() - start_time
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage423_qwen3_deepseek_wordclass_layer_distribution",
        "title": "Qwen3 与 DeepSeek7B 六类词性有效神经元层分布分析",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "used_cuda": use_cuda,
        "batch_size": int(args.batch_size),
        "top_fraction": TOP_FRACTION,
        "target_limits": DEFAULT_TARGET_LIMITS,
        "control_limits": DEFAULT_CONTROL_LIMITS,
        "models": model_payloads,
        "cross_model_summary": build_cross_model_summary(model_payloads),
    }
    write_outputs(summary, output_dir)
    print(
        json.dumps(
            {
                "status_short": "stage423_wordclass_layer_distribution_ready",
                "output_dir": str(output_dir),
                "used_cuda": use_cuda,
                "elapsed_seconds": elapsed,
                "qwen3_noun_top_layers": [
                    row["layer_index"]
                    for row in model_payloads["qwen3"]["classes"]["noun"]["top_layers_by_mass"][:3]
                ],
                "deepseek7b_noun_top_layers": [
                    row["layer_index"]
                    for row in model_payloads["deepseek7b"]["classes"]["noun"]["top_layers_by_mass"][:3]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
