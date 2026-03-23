#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from qwen3_language_shared import (
    PROJECT_ROOT,
    QWEN3_MODEL_PATH,
    capture_qwen_mlp_payloads,
    discover_layers,
    load_qwen3_embedding_weight,
    load_qwen3_model,
    mapped_layer_index,
    move_batch_to_model_device,
    qwen_neuron_dim,
    resolve_anchor_layers,
    remove_hooks,
)
from stage119_gpt2_embedding_full_vocab_scan import (
    build_canonical_inventory,
    build_summary as build_stage119_summary,
    collect_clean_variants,
    fit_group_models,
    fit_lexical_type_models,
    l2_normalize,
    scan_word_rows,
)
from stage121_adverb_gate_bridge_probe import CORE_ADVERB_EXCEPTIONS
from stage122_adverb_context_route_shift_probe import (
    build_case_bundle,
    build_summary as build_stage122_summary,
)
from stage123_route_shift_layer_localization import summarize_layers
from stage130_multisyntax_noun_context_probe import SYNTAX_FAMILIES
from stage133_complex_discourse_noun_propagation import DISCOURSE_FAMILIES
from stage134_noun_verb_joint_propagation import JOINT_FAMILIES
from stage136_anaphora_ellipsis_propagation import REFERENCE_FAMILIES
from stage137_noun_verb_result_chain import CHAIN_FAMILIES, NEUTRAL_RESULT_WORDS, POSITIVE_RESULT_WORDS
from wordclass_neuron_basic_probe_lib import (
    build_neuron_tables,
    build_summary as build_wordclass_summary,
    clamp01,
    top_two_bands,
    unique_target_groups,
)


OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage139_qwen3_language_validation_suite_20260323"
VOCAB_SUMMARY_PATH = OUTPUT_DIR / "qwen_vocab_summary.json"
WORD_ROWS_JSONL_PATH = OUTPUT_DIR / "qwen_word_rows.jsonl"
WORD_ROWS_CSV_PATH = OUTPUT_DIR / "qwen_word_rows.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE139_QWEN3_LANGUAGE_VALIDATION_SUITE_REPORT.md"

NEURON_BATCH_SIZE = 16
PROMPT_BATCH_SIZE = 8
NOUN_PROBE_LIMIT = 4096
CONTROL_PROBE_LIMIT = 4096
NOUN_CONTEXT_LIMITS = {"meso": 160, "macro": 128, "micro": 32}


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_cached_summary(output_dir: Path) -> Dict[str, object] | None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return load_json(summary_path)
    return None


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def l2_unit(vec: torch.Tensor) -> torch.Tensor:
    return vec / torch.linalg.norm(vec).clamp_min(1e-8)


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 1e-12 or y_var <= 1e-12:
        return 0.0
    return cov / math.sqrt(x_var * y_var)


def sign_consistency(xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs:
        return 0.0
    hits = 0
    for x, y in zip(xs, ys):
        if x == 0.0 or y == 0.0 or x * y > 0:
            hits += 1
    return hits / len(xs)


def find_subsequence(sequence: Sequence[int], target: Sequence[int]) -> int | None:
    if not target:
        return None
    for idx in range(len(sequence) - len(target) + 1):
        if list(sequence[idx : idx + len(target)]) == list(target):
            return idx
    return None


def mean_selected(layer_tensor: torch.Tensor, start: int, end: int, neurons: Sequence[int]) -> float:
    neuron_tensor = torch.tensor(list(neurons), dtype=torch.long)
    token_mean = layer_tensor[start:end, :].mean(dim=0)
    return float(token_mean[neuron_tensor].mean().item())


def select_noun_probe_rows(rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    noun_rows = [row for row in rows if row["lexical_type"] == "noun"]
    noun_rows.sort(
        key=lambda row: (
            float(row.get("lexical_type_score", 0.0)),
            float(row.get("effective_encoding_score", 0.0)),
        ),
        reverse=True,
    )
    noun_rows = noun_rows[:NOUN_PROBE_LIMIT]

    control_buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["lexical_type"] != "noun":
            control_buckets[str(row["lexical_type"])].append(row)
    control_rows: List[Dict[str, object]] = []
    per_type = max(64, CONTROL_PROBE_LIMIT // max(1, len(control_buckets)))
    for lexical_type, bucket in sorted(control_buckets.items()):
        bucket.sort(
            key=lambda row: (
                float(row.get("lexical_type_score", 0.0)),
                float(row.get("effective_encoding_score", 0.0)),
            ),
            reverse=True,
        )
        control_rows.extend(bucket[:per_type])
    control_rows = control_rows[:CONTROL_PROBE_LIMIT]
    return noun_rows, control_rows


def balanced_noun_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["lexical_type"] == "noun":
            buckets[str(row["band"])].append(row)
    selected: List[Dict[str, object]] = []
    for band_name, limit in NOUN_CONTEXT_LIMITS.items():
        bucket = buckets.get(band_name, [])
        bucket.sort(
            key=lambda row: (
                float(row.get("lexical_type_score", 0.0)),
                float(row.get("effective_encoding_score", 0.0)),
            ),
            reverse=True,
        )
        selected.extend(bucket[:limit])
    return selected


def select_verbs(rows: Sequence[Dict[str, object]], limit: int = 6) -> List[str]:
    verb_rows = [
        row
        for row in rows
        if row["lexical_type"] == "verb"
        and row["group"] == "macro_action"
        and float(row.get("lexical_type_score", 0.0)) >= 0.55
    ]
    verb_rows.sort(
        key=lambda row: (
            float(row.get("lexical_type_score", 0.0)),
            float(row.get("effective_encoding_score", 0.0)),
        ),
        reverse=True,
    )
    out: List[str] = []
    seen = set()
    for row in verb_rows:
        word = str(row["word"])
        if word not in seen:
            seen.add(word)
            out.append(word)
        if len(out) >= limit:
            break
    return out


def is_core_adverb(row: Dict[str, object]) -> bool:
    if str(row["lexical_type"]) != "adverb":
        return False
    word = str(row["word"]).lower()
    return word.endswith("ly") or word in CORE_ADVERB_EXCEPTIONS


def run_qwen_vocab_scan(output_dir: Path) -> Tuple[Dict[str, object], List[Dict[str, object]], torch.Tensor]:
    if VOCAB_SUMMARY_PATH.exists() and WORD_ROWS_JSONL_PATH.exists():
        summary = load_json(VOCAB_SUMMARY_PATH)
        rows = []
        with WORD_ROWS_JSONL_PATH.open("r", encoding="utf-8-sig") as fh:
            for line in fh:
                rows.append(json.loads(line))
        embed_weight = load_qwen3_embedding_weight()
        return summary, rows, embed_weight

    tokenizer = load_qwen3_model(prefer_cuda=False)[1]
    embed_weight = load_qwen3_embedding_weight()
    variants, skipped_count = collect_clean_variants(tokenizer)
    canonical_rows, matrix, word_to_index = build_canonical_inventory(variants, embed_weight)
    normalized_matrix = l2_normalize(matrix)
    group_list, group_models, seed_coverage = fit_group_models(normalized_matrix, word_to_index)
    lexical_type_list, lexical_type_models, lexical_type_coverage = fit_lexical_type_models(
        normalized_matrix,
        word_to_index,
    )
    rows = scan_word_rows(canonical_rows, matrix, group_models, lexical_type_models)
    summary = build_stage119_summary(
        tokenizer=tokenizer,
        embed_weight=embed_weight,
        skipped_count=skipped_count,
        group_list=group_list,
        seed_coverage=seed_coverage,
        lexical_type_list=lexical_type_list,
        lexical_type_coverage=lexical_type_coverage,
        rows=rows,
    )
    summary["experiment_id"] = "stage139_qwen3_embedding_full_vocab_scan"
    summary["title"] = "Qwen3 词嵌入全词有效编码扫描"
    summary["status_short"] = "qwen3_embedding_vocab_scan_ready"
    summary["model_name"] = "Qwen/Qwen3-4B"
    summary["model_path"] = str(QWEN3_MODEL_PATH)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(VOCAB_SUMMARY_PATH, summary)
    with WORD_ROWS_JSONL_PATH.open("w", encoding="utf-8-sig", newline="") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with WORD_ROWS_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "word",
                "token_id",
                "band",
                "group",
                "group_score",
                "lexical_type",
                "lexical_type_score",
                "effective_encoding_score",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["word"],
                    row["token_id"],
                    row["band"],
                    row["group"],
                    f"{row['group_score']:.6f}",
                    row["lexical_type"],
                    f"{row['lexical_type_score']:.6f}",
                    f"{row['effective_encoding_score']:.6f}",
                ]
            )
    return summary, rows, embed_weight


def select_prototype_indices(
    rows: Sequence[Dict[str, object]],
    *,
    lexical_type: str,
    preferred_band: str | None = None,
    preferred_group: str | None = None,
    min_score: float = 0.15,
    limit: int = 96,
) -> List[int]:
    candidates = [row for row in rows if row["lexical_type"] == lexical_type]
    if preferred_band is not None:
        band_rows = [row for row in candidates if row["band"] == preferred_band]
        if band_rows:
            candidates = band_rows
    if preferred_group is not None:
        group_rows = [row for row in candidates if row["group"] == preferred_group]
        if group_rows:
            candidates = group_rows
    candidates = [row for row in candidates if float(row.get("lexical_type_score", 0.0)) >= min_score]
    candidates.sort(
        key=lambda row: (
            float(row.get("lexical_type_score", 0.0)),
            float(row.get("effective_encoding_score", 0.0)),
        ),
        reverse=True,
    )
    if not candidates:
        raw = [row for row in rows if row["lexical_type"] == lexical_type]
        raw.sort(
            key=lambda row: (
                float(row.get("lexical_type_score", 0.0)),
                float(row.get("effective_encoding_score", 0.0)),
            ),
            reverse=True,
        )
        candidates = raw
    return [int(row["token_id"]) for row in candidates[:limit]]


def build_qwen_route_prototypes(rows: Sequence[Dict[str, object]], embed_weight: torch.Tensor) -> Dict[str, torch.Tensor]:
    spec = {
        "verb": {"lexical_type": "verb", "preferred_band": "macro", "preferred_group": "macro_action"},
        "function": {"lexical_type": "function", "preferred_band": "macro", "preferred_group": None},
        "noun": {"lexical_type": "noun", "preferred_band": "meso", "preferred_group": None},
        "adjective": {"lexical_type": "adjective", "preferred_band": "micro", "preferred_group": None},
    }
    out = {}
    for name, kwargs in spec.items():
        indices = select_prototype_indices(rows, **kwargs)
        mat = embed_weight[indices].float()
        out[name] = l2_unit(mat.mean(dim=0))
    return out


def ids_for_word(tokenizer, word: str) -> List[int]:
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if ids:
        return list(ids)
    return list(tokenizer.encode(word, add_special_tokens=False))


def build_prompt_metrics_qwen(
    outputs,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_idx: int,
    prompt_kind: str,
    prototypes: Dict[str, torch.Tensor],
    verb_ids: Sequence[int],
    adverb_ids: Sequence[int],
    adjective_ids: Sequence[int],
) -> Dict[str, object]:
    valid_len = int(attention_mask[prompt_idx].sum().item())
    seq = input_ids[prompt_idx][:valid_len].tolist()
    verb_pos = find_subsequence(seq, verb_ids)
    if verb_pos is None:
        raise RuntimeError("Qwen 选路偏移探针未找到动词位置")
    modifier_pos = None
    if prompt_kind == "adverb":
        modifier_pos = find_subsequence(seq, adverb_ids)
    elif prompt_kind == "adjective":
        modifier_pos = find_subsequence(seq, adjective_ids)

    verb_route_by_layer: List[float] = []
    last_route_by_layer: List[float] = []
    modifier_attention_by_layer: List[float] = []
    last_pos = len(seq) - 1
    for layer_idx, hidden_state in enumerate(outputs.hidden_states[1:]):
        verb_route_by_layer.append(route_score_from_hidden(hidden_state[prompt_idx, verb_pos, :].detach().float().cpu(), prototypes))
        last_route_by_layer.append(route_score_from_hidden(hidden_state[prompt_idx, last_pos, :].detach().float().cpu(), prototypes))
        if modifier_pos is not None and outputs.attentions is not None and outputs.attentions[layer_idx] is not None:
            attn = outputs.attentions[layer_idx][prompt_idx].detach().float().cpu()
            verb_attn = float(attn[:, verb_pos, modifier_pos].mean().item())
            last_attn = float(attn[:, last_pos, modifier_pos].mean().item())
            modifier_attention_by_layer.append((verb_attn + last_attn) / 2.0)
        elif modifier_pos is not None:
            modifier_attention_by_layer.append(0.0)
    return {
        "kind": prompt_kind,
        "verb_route_by_layer": [float(x) for x in verb_route_by_layer],
        "last_route_by_layer": [float(x) for x in last_route_by_layer],
        "modifier_attention_by_layer": [float(x) for x in modifier_attention_by_layer],
        "verb_route_mean": float(sum(verb_route_by_layer) / len(verb_route_by_layer)),
        "last_route_mean": float(sum(last_route_by_layer) / len(last_route_by_layer)),
        "modifier_attention_mean": float(sum(modifier_attention_by_layer) / max(1, len(modifier_attention_by_layer))),
    }


def analyze_case_qwen(model, tokenizer, prototypes: Dict[str, torch.Tensor], case: Dict[str, str]) -> Dict[str, object]:
    prompts = [case["base_prompt"], case["adverb_prompt"], case["adjective_prompt"]]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids_cpu = encoded["input_ids"].clone()
    attention_mask_cpu = encoded["attention_mask"].clone()
    encoded = move_batch_to_model_device(model, encoded)
    with torch.inference_mode():
        outputs = model(
            **encoded,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
    verb_ids = ids_for_word(tokenizer, case["verb"])
    adverb_ids = ids_for_word(tokenizer, case["adverb"])
    adjective_ids = ids_for_word(tokenizer, case["adjective"])
    prompt_metrics = [
        build_prompt_metrics_qwen(
            outputs=outputs,
            input_ids=input_ids_cpu,
            attention_mask=attention_mask_cpu,
            prompt_idx=prompt_idx,
            prompt_kind=prompt_kind,
            prototypes=prototypes,
            verb_ids=verb_ids,
            adverb_ids=adverb_ids,
            adjective_ids=adjective_ids,
        )
        for prompt_idx, prompt_kind in enumerate(["base", "adverb", "adjective"])
    ]
    base_metrics, adverb_metrics, adjective_metrics = prompt_metrics
    verb_route_delta_by_layer = [float(adv - base) for adv, base in zip(adverb_metrics["verb_route_by_layer"], base_metrics["verb_route_by_layer"])]
    adjective_verb_delta_by_layer = [float(adj - base) for adj, base in zip(adjective_metrics["verb_route_by_layer"], base_metrics["verb_route_by_layer"])]
    verb_route_advantage_by_layer = [float(adv - adj) for adv, adj in zip(adverb_metrics["verb_route_by_layer"], adjective_metrics["verb_route_by_layer"])]
    last_route_advantage_by_layer = [float(adv - adj) for adv, adj in zip(adverb_metrics["last_route_by_layer"], adjective_metrics["last_route_by_layer"])]
    modifier_attention_advantage_by_layer = [
        float(adv - adj) for adv, adj in zip(adverb_metrics["modifier_attention_by_layer"], adjective_metrics["modifier_attention_by_layer"])
    ]
    peak_layer_index = max(range(len(verb_route_advantage_by_layer)), key=lambda idx: verb_route_advantage_by_layer[idx])
    return {
        "verb": case["verb"],
        "adverb": case["adverb"],
        "base_prompt": case["base_prompt"],
        "adverb_prompt": case["adverb_prompt"],
        "adjective_prompt": case["adjective_prompt"],
        "adverb_verb_route_delta": adverb_metrics["verb_route_mean"] - base_metrics["verb_route_mean"],
        "adjective_verb_route_delta": adjective_metrics["verb_route_mean"] - base_metrics["verb_route_mean"],
        "verb_route_advantage": adverb_metrics["verb_route_mean"] - adjective_metrics["verb_route_mean"],
        "adverb_last_route_delta": adverb_metrics["last_route_mean"] - base_metrics["last_route_mean"],
        "adjective_last_route_delta": adjective_metrics["last_route_mean"] - base_metrics["last_route_mean"],
        "adverb_verb_route_peak_delta": max(verb_route_delta_by_layer),
        "adjective_verb_route_peak_delta": max(adjective_verb_delta_by_layer),
        "verb_route_peak_advantage": max(verb_route_advantage_by_layer),
        "adverb_modifier_attention_mean": adverb_metrics["modifier_attention_mean"],
        "adjective_modifier_attention_mean": adjective_metrics["modifier_attention_mean"],
        "modifier_attention_advantage": adverb_metrics["modifier_attention_mean"] - adjective_metrics["modifier_attention_mean"],
        "verb_route_advantage_by_layer": verb_route_advantage_by_layer,
        "last_route_advantage_by_layer": last_route_advantage_by_layer,
        "modifier_attention_advantage_by_layer": modifier_attention_advantage_by_layer,
        "peak_layer_index": int(peak_layer_index),
    }


def run_adverb_bridge(rows: Sequence[Dict[str, object]], embed_weight: torch.Tensor) -> Dict[str, object]:
    prototypes = build_qwen_route_prototypes(rows, embed_weight)
    enriched_rows = []
    for row in rows:
        if row["lexical_type"] not in {"adverb", "verb", "function", "noun", "adjective"}:
            continue
        vec = l2_unit(embed_weight[int(row["token_id"])].float())
        verb_sim = float(torch.dot(vec, prototypes["verb"]).item())
        function_sim = float(torch.dot(vec, prototypes["function"]).item())
        noun_sim = float(torch.dot(vec, prototypes["noun"]).item())
        adjective_sim = float(torch.dot(vec, prototypes["adjective"]).item())
        route_score = (verb_sim + function_sim) / 2.0
        content_score = (noun_sim + adjective_sim) / 2.0
        bridge_margin = route_score - content_score
        balance = 1.0 - abs(verb_sim - function_sim) / (abs(verb_sim) + abs(function_sim) + 1e-8)
        gate_bridge_score = 0.65 * bridge_margin + 0.35 * balance
        enriched_rows.append(
            {
                **row,
                "verb_similarity": verb_sim,
                "function_similarity": function_sim,
                "noun_similarity": noun_sim,
                "adjective_similarity": adjective_sim,
                "route_score": route_score,
                "content_score": content_score,
                "bridge_margin": bridge_margin,
                "action_function_balance": balance,
                "gate_bridge_score": gate_bridge_score,
            }
        )
    by_type: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in enriched_rows:
        by_type[str(row["lexical_type"])].append(row)
    adverb_rows = [row for row in by_type["adverb"] if is_core_adverb(row)]
    by_type["adverb"] = adverb_rows
    type_means = {}
    for lexical_type, type_rows in by_type.items():
        n = max(1, len(type_rows))
        type_means[lexical_type] = {
            "count": len(type_rows),
            "mean_gate_bridge_score": sum(float(row["gate_bridge_score"]) for row in type_rows) / n,
            "mean_bridge_margin": sum(float(row["bridge_margin"]) for row in type_rows) / n,
            "mean_action_function_balance": sum(float(row["action_function_balance"]) for row in type_rows) / n,
        }
    control_mean = (type_means["verb"]["mean_gate_bridge_score"] + type_means["function"]["mean_gate_bridge_score"]) / 2.0
    content_mean = (type_means["noun"]["mean_gate_bridge_score"] + type_means["adjective"]["mean_gate_bridge_score"]) / 2.0
    adverb_mean = type_means["adverb"]["mean_gate_bridge_score"]
    midpoint = (adverb_mean - content_mean) / max(1e-8, control_mean - content_mean)
    adverb_balance = type_means["adverb"]["mean_action_function_balance"]
    bridge_score = 0.60 * clamp01(midpoint) + 0.40 * adverb_balance
    top_gate_adverbs = sorted(adverb_rows, key=lambda row: float(row["gate_bridge_score"]), reverse=True)[:20]
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage139_qwen3_adverb_gate_bridge_probe",
        "title": "Qwen3 副词门控桥探针",
        "status_short": "qwen3_adverb_gate_bridge_ready",
        "source_stage": "stage139_qwen3_embedding_full_vocab_scan",
        "source_output_dir": str(OUTPUT_DIR),
        "core_adverb_count": len(adverb_rows),
        "type_means": type_means,
        "control_gate_mean": float(control_mean),
        "content_gate_mean": float(content_mean),
        "adverb_gate_mean": float(adverb_mean),
        "adverb_midpoint_position": float(midpoint),
        "adverb_action_function_balance_mean": float(adverb_balance),
        "adverb_gate_bridge_score": float(bridge_score),
        "top_gate_adverbs": [
            {
                "word": row["word"],
                "band": row["band"],
                "group": row["group"],
                "bridge_margin": float(row["bridge_margin"]),
                "action_function_balance": float(row["action_function_balance"]),
                "gate_bridge_score": float(row["gate_bridge_score"]),
            }
            for row in top_gate_adverbs
        ],
    }


def run_route_shift(
    model,
    tokenizer,
    rows: Sequence[Dict[str, object]],
    embed_weight: torch.Tensor,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    prototypes = build_qwen_route_prototypes(rows, embed_weight)
    case_rows = [analyze_case_qwen(model, tokenizer, prototypes, case) for case in build_case_bundle()]
    dynamic_summary = build_stage122_summary(case_rows)
    dynamic_summary["experiment_id"] = "stage139_qwen3_adverb_context_route_shift_probe"
    dynamic_summary["title"] = "Qwen3 副词上下文选路偏移探针"
    dynamic_summary["status_short"] = "qwen3_adverb_context_route_shift_ready"
    dynamic_summary["model_name"] = "Qwen/Qwen3-4B"
    dynamic_summary["model_path"] = str(QWEN3_MODEL_PATH)
    layer_summary = summarize_layers(case_rows)
    layer_summary["source_dynamic_score"] = dynamic_summary["adverb_context_route_shift_score"]
    layer_summary["source_case_count"] = dynamic_summary["case_count"]
    return dynamic_summary, layer_summary


def init_probe_stats(layer_count: int, neuron_count: int, group_names: Sequence[str]) -> Dict[str, object]:
    shape = (layer_count, neuron_count)
    group_shape = (len(group_names), layer_count, neuron_count)
    return {
        "target_count": 0,
        "control_count": 0,
        "target_sum": torch.zeros(shape, dtype=torch.float64),
        "target_sumsq": torch.zeros(shape, dtype=torch.float64),
        "target_pos": torch.zeros(shape, dtype=torch.float64),
        "control_sum": torch.zeros(shape, dtype=torch.float64),
        "control_sumsq": torch.zeros(shape, dtype=torch.float64),
        "control_pos": torch.zeros(shape, dtype=torch.float64),
        "group_sum": torch.zeros(group_shape, dtype=torch.float64),
        "group_count": torch.zeros(len(group_names), dtype=torch.float64),
        "band_sum": {},
        "band_count": Counter(),
    }


def update_probe_global(stats: Dict[str, object], sample_tensor: torch.Tensor, is_target: bool) -> None:
    prefix = "target" if is_target else "control"
    stats[f"{prefix}_sum"] += sample_tensor.sum(dim=0)
    stats[f"{prefix}_sumsq"] += (sample_tensor * sample_tensor).sum(dim=0)
    stats[f"{prefix}_pos"] += (sample_tensor > 0).to(torch.float64).sum(dim=0)
    stats[f"{prefix}_count"] += sample_tensor.shape[0]


def update_probe_target(
    stats: Dict[str, object],
    sample_tensor: torch.Tensor,
    batch_rows: Sequence[Dict[str, object]],
    group_index: Dict[str, int],
) -> None:
    for sample_idx, row in enumerate(batch_rows):
        group_name = str(row["group"])
        if group_name in group_index:
            group_idx = group_index[group_name]
            stats["group_sum"][group_idx] += sample_tensor[sample_idx]
            stats["group_count"][group_idx] += 1
        band_name = str(row["band"])
        if band_name not in stats["band_sum"]:
            stats["band_sum"][band_name] = torch.zeros_like(sample_tensor[0])
        stats["band_sum"][band_name] += sample_tensor[sample_idx]
        stats["band_count"][band_name] += 1


def run_wordclass_probe(
    model,
    tokenizer,
    target_rows: Sequence[Dict[str, object]],
    control_rows: Sequence[Dict[str, object]],
    group_names: Sequence[str],
) -> Dict[str, object]:
    layer_indices = list(range(len(discover_layers(model))))
    buffers, handles = capture_qwen_mlp_payloads(model, {idx: "neuron_in" for idx in layer_indices})
    stats = init_probe_stats(len(layer_indices), qwen_neuron_dim(model), group_names)
    group_index = {name: idx for idx, name in enumerate(group_names)}
    try:
        for rows_chunk, is_target in ((target_rows, True), (control_rows, False)):
            for start in range(0, len(rows_chunk), NEURON_BATCH_SIZE):
                batch_rows = rows_chunk[start : start + NEURON_BATCH_SIZE]
                encoded = tokenizer(
                    [str(row["word"]) for row in batch_rows],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                attention_mask = encoded["attention_mask"].cpu()
                encoded = move_batch_to_model_device(model, encoded)
                with torch.inference_mode():
                    model(**encoded, use_cache=False, return_dict=True)
                per_layer_rows = []
                for layer_idx in layer_indices:
                    layer_tensor = buffers[layer_idx]
                    if layer_tensor is None:
                        raise RuntimeError("Qwen 词类探针未捕获到层输出")
                    mask = attention_mask.unsqueeze(-1).to(layer_tensor.dtype)
                    lengths = attention_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(layer_tensor.dtype)
                    mean_tensor = (layer_tensor * mask).sum(dim=1) / lengths
                    per_layer_rows.append(mean_tensor.to(torch.float64))
                sample_tensor = torch.stack(per_layer_rows, dim=1)
                update_probe_global(stats, sample_tensor, is_target=is_target)
                if is_target:
                    update_probe_target(stats, sample_tensor, batch_rows, group_index)
    finally:
        remove_hooks(handles)
    return stats


def run_noun_basic_probe(stage119_summary: Dict[str, object], rows: Sequence[Dict[str, object]], model, tokenizer) -> Dict[str, object]:
    noun_rows, control_rows = select_noun_probe_rows(rows)
    group_names = unique_target_groups(noun_rows)
    primary_band_name, secondary_band_name, band_counts = top_two_bands(noun_rows)
    stats = run_wordclass_probe(model, tokenizer, noun_rows, control_rows, group_names)
    tables = build_neuron_tables(
        stats,
        group_names,
        primary_band_name=primary_band_name,
        secondary_band_name=secondary_band_name,
    )
    summary = build_wordclass_summary(
        experiment_id="stage139_qwen3_noun_neuron_basic_probe",
        title="Qwen3 名词神经元基础探针",
        status_short="qwen3_noun_neuron_basic_ready",
        target_lexical_type="noun",
        stage119_summary=stage119_summary,
        target_rows=noun_rows,
        neuron_tables=tables,
        band_counts=band_counts,
    )
    summary["model_name"] = "Qwen/Qwen3-4B"
    summary["model_path"] = str(QWEN3_MODEL_PATH)
    summary["source_stage"] = "stage139_qwen3_embedding_full_vocab_scan"
    return summary


def build_syntax_inputs(
    tokenizer,
    batch_rows: Sequence[Dict[str, object]],
    family: Dict[str, str],
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[int, int]]]:
    texts = []
    ranges = []
    for row in batch_rows:
        word = str(row["word"])
        text = f"{family['prefix']} {word}{family['suffix']}"
        texts.append(text)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        word_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if not word_ids:
            word_ids = tokenizer.encode(word, add_special_tokens=False)
        start = find_subsequence(token_ids, word_ids)
        if start is None:
            raise RuntimeError("Qwen 多句法探针未定位到名词位置")
        ranges.append((start, start + len(word_ids)))
    encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return encoded, ranges


def run_syntax_family(
    model,
    tokenizer,
    noun_rows: Sequence[Dict[str, object]],
    control_rows: Sequence[Dict[str, object]],
    family: Dict[str, str],
    group_names: Sequence[str],
    primary_band_name: str,
    secondary_band_name: str,
) -> Dict[str, object]:
    layer_indices = list(range(len(discover_layers(model))))
    buffers, handles = capture_qwen_mlp_payloads(model, {idx: "neuron_in" for idx in layer_indices})
    stats = init_probe_stats(len(layer_indices), qwen_neuron_dim(model), group_names)
    group_index = {name: idx for idx, name in enumerate(group_names)}
    try:
        for rows_chunk, is_target in ((noun_rows, True), (control_rows, False)):
            for start in range(0, len(rows_chunk), NEURON_BATCH_SIZE):
                batch_rows = rows_chunk[start : start + NEURON_BATCH_SIZE]
                encoded, ranges = build_syntax_inputs(tokenizer, batch_rows, family)
                encoded = move_batch_to_model_device(model, encoded)
                with torch.inference_mode():
                    model(**encoded, use_cache=False, return_dict=True)
                per_layer_rows = []
                for layer_idx in layer_indices:
                    layer_tensor = buffers[layer_idx]
                    if layer_tensor is None:
                        raise RuntimeError("Qwen 多句法探针未捕获到层输出")
                    token_rows = []
                    for sample_idx, (word_start, word_end) in enumerate(ranges):
                        token_rows.append(layer_tensor[sample_idx, word_start:word_end, :].mean(dim=0))
                    per_layer_rows.append(torch.stack(token_rows, dim=0).to(torch.float64))
                sample_tensor = torch.stack(per_layer_rows, dim=1)
                update_probe_global(stats, sample_tensor, is_target=is_target)
                if is_target:
                    update_probe_target(stats, sample_tensor, batch_rows, group_index)
    finally:
        remove_hooks(handles)

    tables = build_neuron_tables(
        stats,
        group_names,
        primary_band_name=primary_band_name,
        secondary_band_name=secondary_band_name,
    )
    return {
        "family_name": family["name"],
        "dominant_general_layer_index": tables["dominant_general_layer_index"],
        "dominant_general_layer_score": tables["dominant_general_layer_score"],
        "top20_mean_general_rule_score": tables["layer_rows"][tables["dominant_general_layer_index"]]["top20_mean_general_rule_score"],
        "top_general_neurons": tables["top_general_neurons"][:12],
    }


def run_noun_context_probe(
    stage119_summary: Dict[str, object],
    rows: Sequence[Dict[str, object]],
    model,
    tokenizer,
    anchor_layers: Dict[str, int],
) -> Dict[str, object]:
    noun_rows = balanced_noun_rows(rows)
    control_rows = [row for row in rows if row["lexical_type"] != "noun"][: len(noun_rows)]
    group_names = unique_target_groups(noun_rows)
    primary_band_name, secondary_band_name, band_counts = top_two_bands(noun_rows)

    family_rows = []
    neuron_counter: Counter[Tuple[int, int]] = Counter()
    neuron_payloads: Dict[Tuple[int, int], Dict[str, object]] = {}
    early_threshold = mapped_layer_index(len(discover_layers(model)), 2)

    for family in SYNTAX_FAMILIES:
        family_result = run_syntax_family(
            model,
            tokenizer,
            noun_rows,
            control_rows,
            family,
            group_names,
            primary_band_name,
            secondary_band_name,
        )
        family_rows.append(
            {
                "family_name": family_result["family_name"],
                "dominant_general_layer_index": family_result["dominant_general_layer_index"],
                "dominant_general_layer_score": family_result["dominant_general_layer_score"],
                "top20_mean_general_rule_score": family_result["top20_mean_general_rule_score"],
                "top_general_neuron": family_result["top_general_neurons"][0],
            }
        )
        for row in family_result["top_general_neurons"]:
            if int(row["layer_index"]) <= early_threshold:
                key = (int(row["layer_index"]), int(row["neuron_index"]))
                neuron_counter[key] += 1
                neuron_payloads[key] = row

    recurrent_early_neurons = []
    for (layer_idx, neuron_idx), count in neuron_counter.most_common(24):
        row = dict(neuron_payloads[(layer_idx, neuron_idx)])
        row["syntax_hit_count"] = int(count)
        recurrent_early_neurons.append(row)

    early_family_count = sum(1 for row in family_rows if int(row["dominant_general_layer_index"]) <= early_threshold)
    exact_early_count = sum(
        1 for row in family_rows if int(row["dominant_general_layer_index"]) == int(anchor_layers["early_layer"])
    )
    mean_family_score = sum(float(row["top20_mean_general_rule_score"]) for row in family_rows) / max(1, len(family_rows))
    syntax_stability = early_family_count / max(1, len(family_rows))
    recurrent_hit_mean = (
        sum(float(row["syntax_hit_count"]) for row in recurrent_early_neurons[:12]) / max(1, min(12, len(recurrent_early_neurons)))
    )
    multisyntax_score = (
        0.40 * syntax_stability
        + 0.30 * clamp01(mean_family_score / 0.55)
        + 0.20 * clamp01(recurrent_hit_mean / 4.0)
        + 0.10 * clamp01(exact_early_count / max(1, len(family_rows)))
    )

    summary = build_wordclass_summary(
        experiment_id="stage139_qwen3_multisyntax_noun_context_probe",
        title="Qwen3 多句法名词上下文探针",
        status_short="qwen3_multisyntax_noun_context_ready",
        target_lexical_type="noun",
        stage119_summary=stage119_summary,
        target_rows=noun_rows,
        neuron_tables={
            "target_count": len(noun_rows),
            "control_count": len(control_rows),
            "layer_count": len(discover_layers(model)),
            "neurons_per_layer": qwen_neuron_dim(model),
            "target_group_count": len(group_names),
            "primary_band_name": primary_band_name,
            "secondary_band_name": secondary_band_name,
            "dominant_general_layer_index": max(family_rows, key=lambda row: row["top20_mean_general_rule_score"])["dominant_general_layer_index"],
            "dominant_general_layer_score": max(float(row["top20_mean_general_rule_score"]) for row in family_rows),
            "wordclass_neuron_basic_probe_score": 0.0,
            "layer_rows": [],
            "top_general_neurons": list(recurrent_early_neurons),
            "top_primary_band_bias_neurons": [],
            "top_secondary_band_bias_neurons": [],
        },
        band_counts=band_counts,
    )
    summary.update(
        {
            "model_name": "Qwen/Qwen3-4B",
            "model_path": str(QWEN3_MODEL_PATH),
            "source_stage": "stage139_qwen3_embedding_full_vocab_scan",
            "family_count": len(family_rows),
            "family_rows": family_rows,
            "recurrent_early_neurons": recurrent_early_neurons,
            "early_family_count": early_family_count,
            "mapped_early_layer_index": anchor_layers["early_layer"],
            "exact_early_family_count": exact_early_count,
            "mean_family_score": mean_family_score,
            "syntax_stability_rate": syntax_stability,
            "recurrent_early_hit_mean": recurrent_hit_mean,
            "multisyntax_noun_context_score": multisyntax_score,
        }
    )
    return summary


def choose_early_neurons(noun_context_summary: Dict[str, object], anchor_layers: Dict[str, int]) -> List[int]:
    target_layer = int(anchor_layers["early_layer"])
    rows = [row for row in noun_context_summary["recurrent_early_neurons"] if int(row["layer_index"]) == target_layer]
    rows.sort(key=lambda row: (int(row["syntax_hit_count"]), float(row["general_rule_score"])), reverse=True)
    neurons = [int(row["neuron_index"]) for row in rows[:12]]
    if neurons:
        return neurons
    fallback = sorted(
        noun_context_summary["recurrent_early_neurons"],
        key=lambda row: (int(row["syntax_hit_count"]), float(row["general_rule_score"])),
        reverse=True,
    )[:12]
    return [int(row["neuron_index"]) for row in fallback]


def choose_late_neurons(noun_basic_summary: Dict[str, object], anchor_layers: Dict[str, int]) -> List[int]:
    target_layer = int(anchor_layers["late_layer"])
    rows = [row for row in noun_basic_summary["top_general_neurons"] if int(row["layer_index"]) == target_layer]
    rows.sort(key=lambda row: float(row["general_rule_score"]), reverse=True)
    neurons = [int(row["neuron_index"]) for row in rows[:12]]
    if neurons:
        return neurons
    fallback = sorted(
        noun_basic_summary["top_general_neurons"],
        key=lambda row: float(row["general_rule_score"]),
        reverse=True,
    )[:12]
    return [int(row["neuron_index"]) for row in fallback]


def run_complex_discourse_probe(
    model,
    tokenizer,
    rows: Sequence[Dict[str, object]],
    early_neurons: Sequence[int],
    late_neurons: Sequence[int],
    anchor_layers: Dict[str, int],
) -> Dict[str, object]:
    selected_rows = balanced_noun_rows(rows)
    payload_map = {
        int(anchor_layers["early_layer"]): "neuron_in",
        int(anchor_layers["late_layer"]): "neuron_in",
    }
    buffers, handles = capture_qwen_mlp_payloads(model, payload_map)
    family_rows = []
    try:
        for family in DISCOURSE_FAMILIES:
            early_first_values: List[float] = []
            early_last_values: List[float] = []
            late_first_values: List[float] = []
            late_last_values: List[float] = []
            for start in range(0, len(selected_rows), PROMPT_BATCH_SIZE):
                batch_rows = selected_rows[start : start + PROMPT_BATCH_SIZE]
                texts = []
                target_ranges = []
                for row in batch_rows:
                    word = str(row["word"])
                    text = f"{family['prefix']} {word}{family['middle']} {word}{family['suffix']}"
                    texts.append(text)
                    full_ids = tokenizer.encode(text, add_special_tokens=False)
                    word_ids = tokenizer.encode(" " + word, add_special_tokens=False)
                    if not word_ids:
                        word_ids = tokenizer.encode(word, add_special_tokens=False)
                    hits = []
                    for idx in range(len(full_ids) - len(word_ids) + 1):
                        if list(full_ids[idx : idx + len(word_ids)]) == list(word_ids):
                            hits.append(idx)
                    if len(hits) < 2:
                        raise RuntimeError("Qwen 复杂语篇探针未找到两次名词位置")
                    target_ranges.append((hits[0], hits[0] + len(word_ids), hits[-1], hits[-1] + len(word_ids)))
                encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
                encoded = move_batch_to_model_device(model, encoded)
                with torch.inference_mode():
                    model(**encoded, use_cache=False, return_dict=True)
                early_layer = buffers[int(anchor_layers["early_layer"])]
                late_layer = buffers[int(anchor_layers["late_layer"])]
                if early_layer is None or late_layer is None:
                    raise RuntimeError("Qwen 复杂语篇探针未捕获到层输出")
                for sample_idx, (first_start, first_end, last_start, last_end) in enumerate(target_ranges):
                    early_first_values.append(mean_selected(early_layer[sample_idx], first_start, first_end, early_neurons))
                    early_last_values.append(mean_selected(early_layer[sample_idx], last_start, last_end, early_neurons))
                    late_first_values.append(mean_selected(late_layer[sample_idx], first_start, first_end, late_neurons))
                    late_last_values.append(mean_selected(late_layer[sample_idx], last_start, last_end, late_neurons))

            early_corr = correlation(early_first_values, early_last_values)
            late_corr = correlation(late_first_values, late_last_values)
            early_consistency = sign_consistency(early_first_values, early_last_values)
            late_consistency = sign_consistency(late_first_values, late_last_values)
            family_score = (
                0.30 * clamp01((early_corr + 1.0) / 2.0)
                + 0.30 * clamp01((late_corr + 1.0) / 2.0)
                + 0.20 * early_consistency
                + 0.20 * late_consistency
            )
            family_rows.append(
                {
                    "family_name": family["name"],
                    "early_remention_corr": early_corr,
                    "late_remention_corr": late_corr,
                    "early_sign_consistency": early_consistency,
                    "late_sign_consistency": late_consistency,
                    "family_score": family_score,
                }
            )
    finally:
        remove_hooks(handles)

    mean_early = sum(float(row["early_remention_corr"]) for row in family_rows) / len(family_rows)
    mean_late = sum(float(row["late_remention_corr"]) for row in family_rows) / len(family_rows)
    mean_score = sum(float(row["family_score"]) for row in family_rows) / len(family_rows)
    score = (
        0.40 * clamp01((mean_early + 1.0) / 2.0)
        + 0.30 * clamp01((mean_late + 1.0) / 2.0)
        + 0.30 * mean_score
    )
    return {
        "family_count": len(family_rows),
        "mean_early_remention_corr": mean_early,
        "mean_late_remention_corr": mean_late,
        "mean_family_score": mean_score,
        "complex_discourse_noun_propagation_score": score,
        "family_rows": family_rows,
    }


def build_joint_case_rows(noun_rows: Sequence[Dict[str, object]], verbs: Sequence[str]) -> List[Dict[str, object]]:
    cases = []
    for noun_row in noun_rows:
        for verb in verbs:
            cases.append(
                {
                    "word": str(noun_row["word"]),
                    "band": str(noun_row["band"]),
                    "group": str(noun_row["group"]),
                    "verb": verb,
                }
            )
    return cases


def embedding_centroid(tokenizer, embed_weight: torch.Tensor, words: Sequence[str]) -> torch.Tensor:
    vectors = []
    for word in words:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(word, add_special_tokens=False)
        vectors.append(embed_weight[ids].float().mean(dim=0))
    return torch.stack(vectors, dim=0).mean(dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.norm(a).clamp_min(1e-8) * torch.linalg.norm(b).clamp_min(1e-8)
    return float(torch.dot(a, b).item() / float(denom))


def route_score_from_hidden(vec: torch.Tensor, route_prototypes: Dict[str, torch.Tensor]) -> float:
    unit = l2_unit(vec.float())
    route = (torch.dot(unit, route_prototypes["verb"]) + torch.dot(unit, route_prototypes["function"])) / 2.0
    content = (torch.dot(unit, route_prototypes["noun"]) + torch.dot(unit, route_prototypes["adjective"])) / 2.0
    return float((route - content).item())


def run_noun_verb_joint_probe(
    model,
    tokenizer,
    rows: Sequence[Dict[str, object]],
    route_prototypes: Dict[str, torch.Tensor],
    early_neurons: Sequence[int],
    anchor_layers: Dict[str, int],
) -> Dict[str, object]:
    noun_rows = balanced_noun_rows(rows)
    verbs = select_verbs(rows)
    case_rows = build_joint_case_rows(noun_rows, verbs)
    early_layer_idx = int(anchor_layers["early_layer"])
    route_layer_idx = int(anchor_layers["route_layer"])
    buffers, handles = capture_qwen_mlp_payloads(model, {early_layer_idx: "neuron_in"})
    family_rows = []
    try:
        for family in JOINT_FAMILIES:
            noun_anchor_values: List[float] = []
            verb_route_values: List[float] = []
            band_route_values: Dict[str, List[float]] = defaultdict(list)
            sign_hits = 0
            for start in range(0, len(case_rows), PROMPT_BATCH_SIZE):
                batch_rows = case_rows[start : start + PROMPT_BATCH_SIZE]
                texts = []
                target_ranges = []
                for row in batch_rows:
                    word = str(row["word"])
                    verb = str(row["verb"])
                    text = f"{family['prefix']} {word}{family['middle']} {verb}{family['suffix']}"
                    texts.append(text)
                    full_ids = tokenizer.encode(text, add_special_tokens=False)
                    noun_ids = tokenizer.encode(" " + word, add_special_tokens=False) or tokenizer.encode(word, add_special_tokens=False)
                    verb_ids = tokenizer.encode(" " + verb, add_special_tokens=False) or tokenizer.encode(verb, add_special_tokens=False)
                    noun_start = find_subsequence(full_ids, noun_ids)
                    verb_start = find_subsequence(full_ids, verb_ids)
                    if noun_start is None or verb_start is None:
                        raise RuntimeError("Qwen 名词-动词联合探针未定位到目标位置")
                    target_ranges.append((noun_start, noun_start + len(noun_ids), verb_start, verb_start + len(verb_ids)))
                encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
                encoded = move_batch_to_model_device(model, encoded)
                with torch.inference_mode():
                    outputs = model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
                early_layer = buffers[early_layer_idx]
                route_hidden = outputs.hidden_states[route_layer_idx + 1].detach().float().cpu()
                if early_layer is None:
                    raise RuntimeError("Qwen 名词-动词联合探针未捕获到早层神经元输出")
                for sample_idx, (noun_start, noun_end, verb_start, verb_end) in enumerate(target_ranges):
                    noun_anchor = mean_selected(early_layer[sample_idx], noun_start, noun_end, early_neurons)
                    verb_vec = route_hidden[sample_idx, verb_start:verb_end, :].mean(dim=0)
                    verb_route = route_score_from_hidden(verb_vec, route_prototypes)
                    noun_anchor_values.append(noun_anchor)
                    verb_route_values.append(verb_route)
                    band_route_values[str(batch_rows[sample_idx]["band"])].append(verb_route)
                    if noun_anchor == 0.0 or verb_route == 0.0 or noun_anchor * verb_route > 0:
                        sign_hits += 1
            noun_route_corr = correlation(noun_anchor_values, verb_route_values)
            sign_rate = sign_hits / max(1, len(noun_anchor_values))
            macro_mean = sum(band_route_values.get("macro", [0.0])) / max(1, len(band_route_values.get("macro", [])))
            meso_mean = sum(band_route_values.get("meso", [0.0])) / max(1, len(band_route_values.get("meso", [])))
            micro_mean = sum(band_route_values.get("micro", [0.0])) / max(1, len(band_route_values.get("micro", [])))
            route_band_gap = abs(macro_mean - meso_mean)
            family_score = (
                0.45 * clamp01((noun_route_corr + 1.0) / 2.0)
                + 0.30 * sign_rate
                + 0.25 * clamp01(route_band_gap / 0.08)
            )
            family_rows.append(
                {
                    "family_name": family["name"],
                    "noun_route_corr": noun_route_corr,
                    "sign_consistency_rate": sign_rate,
                    "macro_route_mean": macro_mean,
                    "meso_route_mean": meso_mean,
                    "micro_route_mean": micro_mean,
                    "route_band_gap": route_band_gap,
                    "joint_family_score": family_score,
                }
            )
    finally:
        remove_hooks(handles)

    mean_corr = sum(float(row["noun_route_corr"]) for row in family_rows) / len(family_rows)
    mean_sign = sum(float(row["sign_consistency_rate"]) for row in family_rows) / len(family_rows)
    mean_gap = sum(float(row["route_band_gap"]) for row in family_rows) / len(family_rows)
    mean_score = sum(float(row["joint_family_score"]) for row in family_rows) / len(family_rows)
    positive_family_rate = sum(1 for row in family_rows if float(row["noun_route_corr"]) > 0.0) / len(family_rows)
    score = (
        0.45 * clamp01((mean_corr + 1.0) / 2.0)
        + 0.25 * mean_sign
        + 0.15 * clamp01(mean_gap / 0.08)
        + 0.15 * positive_family_rate
    )
    return {
        "family_count": len(family_rows),
        "noun_sample_count": len(noun_rows),
        "verb_count": len(verbs),
        "mean_noun_route_corr": mean_corr,
        "mean_sign_consistency_rate": mean_sign,
        "mean_route_band_gap": mean_gap,
        "positive_family_rate": positive_family_rate,
        "mean_family_score": mean_score,
        "noun_verb_joint_propagation_score": score,
        "verbs": verbs,
        "family_rows": family_rows,
    }


def run_anaphora_probe(
    model,
    tokenizer,
    rows: Sequence[Dict[str, object]],
    early_neurons: Sequence[int],
    late_neurons: Sequence[int],
    anchor_layers: Dict[str, int],
) -> Dict[str, object]:
    noun_rows = balanced_noun_rows(rows)
    payload_map = {int(anchor_layers["early_layer"]): "neuron_in", int(anchor_layers["late_layer"]): "neuron_in"}
    buffers, handles = capture_qwen_mlp_payloads(model, payload_map)
    family_rows = []
    try:
        for family in REFERENCE_FAMILIES:
            noun_source_early: List[float] = []
            pronoun_target_early: List[float] = []
            ellipsis_target_early: List[float] = []
            noun_source_late: List[float] = []
            pronoun_target_late: List[float] = []
            ellipsis_target_late: List[float] = []
            for start in range(0, len(noun_rows), PROMPT_BATCH_SIZE):
                batch_rows = noun_rows[start : start + PROMPT_BATCH_SIZE]
                texts = []
                ranges = []
                kinds = []
                pronoun_ids = tokenizer.encode(" it", add_special_tokens=False)
                that_ids = tokenizer.encode(" that", add_special_tokens=False)
                for row in batch_rows:
                    word = str(row["word"])
                    noun_ids = tokenizer.encode(" " + word, add_special_tokens=False) or tokenizer.encode(word, add_special_tokens=False)
                    prompt_specs = [
                        ("pronoun", f"{family['prefix']} {word}{family['middle']}{family['pronoun_suffix']}{family['tail']}"),
                        ("ellipsis", f"{family['prefix']} {word}{family['middle']}{family['ellipsis_suffix']}{family['tail']}"),
                    ]
                    for kind, text in prompt_specs:
                        full_ids = tokenizer.encode(text, add_special_tokens=False)
                        source_start = find_subsequence(full_ids, noun_ids)
                        if source_start is None:
                            raise RuntimeError("Qwen 回指/省略探针未定位到源名词")
                        target_ids = pronoun_ids if kind == "pronoun" else that_ids
                        target_start = find_subsequence(full_ids, target_ids)
                        if target_start is None:
                            raise RuntimeError("Qwen 回指/省略探针未定位到目标词")
                        texts.append(text)
                        ranges.append((source_start, source_start + len(noun_ids), target_start, target_start + len(target_ids)))
                        kinds.append(kind)
                encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
                encoded = move_batch_to_model_device(model, encoded)
                with torch.inference_mode():
                    model(**encoded, use_cache=False, return_dict=True)
                early_layer = buffers[int(anchor_layers["early_layer"])]
                late_layer = buffers[int(anchor_layers["late_layer"])]
                if early_layer is None or late_layer is None:
                    raise RuntimeError("Qwen 回指/省略探针未捕获到层输出")
                for sample_idx, kind in enumerate(kinds):
                    source_start, source_end, target_start, target_end = ranges[sample_idx]
                    early_source = mean_selected(early_layer[sample_idx], source_start, source_end, early_neurons)
                    early_target = mean_selected(early_layer[sample_idx], target_start, target_end, early_neurons)
                    late_source = mean_selected(late_layer[sample_idx], source_start, source_end, late_neurons)
                    late_target = mean_selected(late_layer[sample_idx], target_start, target_end, late_neurons)
                    noun_source_early.append(early_source)
                    noun_source_late.append(late_source)
                    if kind == "pronoun":
                        pronoun_target_early.append(early_target)
                        pronoun_target_late.append(late_target)
                    else:
                        ellipsis_target_early.append(early_target)
                        ellipsis_target_late.append(late_target)
            noun_pronoun_early_corr = correlation(noun_source_early[: len(pronoun_target_early)], pronoun_target_early)
            noun_ellipsis_early_corr = correlation(noun_source_early[: len(ellipsis_target_early)], ellipsis_target_early)
            noun_pronoun_late_corr = correlation(noun_source_late[: len(pronoun_target_late)], pronoun_target_late)
            noun_ellipsis_late_corr = correlation(noun_source_late[: len(ellipsis_target_late)], ellipsis_target_late)
            pronoun_sign = sign_consistency(noun_source_late[: len(pronoun_target_late)], pronoun_target_late)
            ellipsis_sign = sign_consistency(noun_source_late[: len(ellipsis_target_late)], ellipsis_target_late)
            family_score = (
                0.25 * clamp01((noun_pronoun_late_corr + 1.0) / 2.0)
                + 0.25 * clamp01((noun_ellipsis_late_corr + 1.0) / 2.0)
                + 0.25 * pronoun_sign
                + 0.25 * ellipsis_sign
            )
            family_rows.append(
                {
                    "family_name": family["name"],
                    "noun_pronoun_early_corr": noun_pronoun_early_corr,
                    "noun_ellipsis_early_corr": noun_ellipsis_early_corr,
                    "noun_pronoun_late_corr": noun_pronoun_late_corr,
                    "noun_ellipsis_late_corr": noun_ellipsis_late_corr,
                    "pronoun_sign_consistency_rate": pronoun_sign,
                    "ellipsis_sign_consistency_rate": ellipsis_sign,
                    "family_score": family_score,
                }
            )
    finally:
        remove_hooks(handles)

    mean_pronoun_early = sum(float(row["noun_pronoun_early_corr"]) for row in family_rows) / len(family_rows)
    mean_ellipsis_early = sum(float(row["noun_ellipsis_early_corr"]) for row in family_rows) / len(family_rows)
    mean_pronoun_late = sum(float(row["noun_pronoun_late_corr"]) for row in family_rows) / len(family_rows)
    mean_ellipsis_late = sum(float(row["noun_ellipsis_late_corr"]) for row in family_rows) / len(family_rows)
    score = (
        0.20 * clamp01((mean_pronoun_early + 1.0) / 2.0)
        + 0.20 * clamp01((mean_ellipsis_early + 1.0) / 2.0)
        + 0.30 * clamp01((mean_pronoun_late + 1.0) / 2.0)
        + 0.30 * clamp01((mean_ellipsis_late + 1.0) / 2.0)
    )
    return {
        "family_count": len(family_rows),
        "noun_pronoun_early_corr": mean_pronoun_early,
        "noun_ellipsis_early_corr": mean_ellipsis_early,
        "noun_pronoun_late_corr": mean_pronoun_late,
        "noun_ellipsis_late_corr": mean_ellipsis_late,
        "anaphora_ellipsis_score": score,
        "family_rows": family_rows,
    }


def run_result_chain_probe(
    model,
    tokenizer,
    rows: Sequence[Dict[str, object]],
    embed_weight: torch.Tensor,
    route_prototypes: Dict[str, torch.Tensor],
    early_neurons: Sequence[int],
    anchor_layers: Dict[str, int],
) -> Dict[str, object]:
    noun_rows = balanced_noun_rows(rows)
    verbs = select_verbs(rows)
    case_rows = build_joint_case_rows(noun_rows, verbs)
    early_layer_idx = int(anchor_layers["early_layer"])
    route_layer_idx = int(anchor_layers["route_layer"])
    late_layer_idx = int(anchor_layers["late_layer"])
    positive_centroid = embedding_centroid(tokenizer, embed_weight, POSITIVE_RESULT_WORDS)
    neutral_centroid = embedding_centroid(tokenizer, embed_weight, NEUTRAL_RESULT_WORDS)
    buffers, handles = capture_qwen_mlp_payloads(model, {early_layer_idx: "neuron_in"})
    family_rows = []
    try:
        for family in CHAIN_FAMILIES:
            noun_anchor_values: List[float] = []
            verb_route_values: List[float] = []
            result_values: List[float] = []
            for start in range(0, len(case_rows), PROMPT_BATCH_SIZE):
                batch_rows = case_rows[start : start + PROMPT_BATCH_SIZE]
                texts = []
                target_ranges = []
                result_word = family["result_word"]
                result_ids = tokenizer.encode(" " + result_word, add_special_tokens=False)
                for row in batch_rows:
                    word = str(row["word"])
                    verb = str(row["verb"])
                    text = f"{family['prefix']} {word}{family['middle']} {verb}{family['postverb']} {result_word}{family['tail']}"
                    texts.append(text)
                    full_ids = tokenizer.encode(text, add_special_tokens=False)
                    noun_ids = tokenizer.encode(" " + word, add_special_tokens=False) or tokenizer.encode(word, add_special_tokens=False)
                    verb_ids = tokenizer.encode(" " + verb, add_special_tokens=False) or tokenizer.encode(verb, add_special_tokens=False)
                    noun_start = find_subsequence(full_ids, noun_ids)
                    verb_start = find_subsequence(full_ids, verb_ids)
                    result_start = find_subsequence(full_ids, result_ids)
                    if noun_start is None or verb_start is None or result_start is None:
                        raise RuntimeError("Qwen 结果链探针未定位到目标位置")
                    target_ranges.append((noun_start, noun_start + len(noun_ids), verb_start, verb_start + len(verb_ids), result_start, result_start + len(result_ids)))
                encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
                encoded = move_batch_to_model_device(model, encoded)
                with torch.inference_mode():
                    outputs = model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
                early_layer = buffers[early_layer_idx]
                route_hidden = outputs.hidden_states[route_layer_idx + 1].detach().float().cpu()
                late_hidden = outputs.hidden_states[late_layer_idx + 1].detach().float().cpu()
                if early_layer is None:
                    raise RuntimeError("Qwen 结果链探针未捕获到早层神经元输出")
                for sample_idx, (noun_start, noun_end, verb_start, verb_end, result_start, result_end) in enumerate(target_ranges):
                    noun_anchor = mean_selected(early_layer[sample_idx], noun_start, noun_end, early_neurons)
                    verb_vec = route_hidden[sample_idx, verb_start:verb_end, :].mean(dim=0)
                    result_vec = late_hidden[sample_idx, result_start:result_end, :].mean(dim=0)
                    verb_route = route_score_from_hidden(verb_vec, route_prototypes)
                    result_score = cosine(result_vec, positive_centroid) - cosine(result_vec, neutral_centroid)
                    noun_anchor_values.append(noun_anchor)
                    verb_route_values.append(verb_route)
                    result_values.append(result_score)
            noun_verb_corr = correlation(noun_anchor_values, verb_route_values)
            verb_result_corr = correlation(verb_route_values, result_values)
            noun_result_corr = correlation(noun_anchor_values, result_values)
            family_score = (
                0.35 * clamp01((noun_verb_corr + 1.0) / 2.0)
                + 0.35 * clamp01((verb_result_corr + 1.0) / 2.0)
                + 0.30 * clamp01((noun_result_corr + 1.0) / 2.0)
            )
            family_rows.append(
                {
                    "family_name": family["name"],
                    "noun_verb_corr": noun_verb_corr,
                    "verb_result_corr": verb_result_corr,
                    "noun_result_corr": noun_result_corr,
                    "chain_family_score": family_score,
                }
            )
    finally:
        remove_hooks(handles)

    mean_noun_verb_corr = sum(float(row["noun_verb_corr"]) for row in family_rows) / len(family_rows)
    mean_verb_result_corr = sum(float(row["verb_result_corr"]) for row in family_rows) / len(family_rows)
    mean_noun_result_corr = sum(float(row["noun_result_corr"]) for row in family_rows) / len(family_rows)
    positive_bridge_rate = sum(1 for row in family_rows if float(row["verb_result_corr"]) > 0.0) / len(family_rows)
    mean_family_score = sum(float(row["chain_family_score"]) for row in family_rows) / len(family_rows)
    score = (
        0.35 * clamp01((mean_noun_verb_corr + 1.0) / 2.0)
        + 0.35 * clamp01((mean_verb_result_corr + 1.0) / 2.0)
        + 0.20 * clamp01((mean_noun_result_corr + 1.0) / 2.0)
        + 0.10 * positive_bridge_rate
    )
    return {
        "family_count": len(family_rows),
        "noun_sample_count": len(noun_rows),
        "verb_count": len(verbs),
        "mean_noun_verb_corr": mean_noun_verb_corr,
        "mean_verb_result_corr": mean_verb_result_corr,
        "mean_noun_result_corr": mean_noun_result_corr,
        "positive_bridge_rate": positive_bridge_rate,
        "mean_family_score": mean_family_score,
        "noun_verb_result_chain_score": score,
        "verbs": verbs,
        "family_rows": family_rows,
    }


def run_conditional_field_fit(
    adverb_summary: Dict[str, object],
    route_summary: Dict[str, object],
    joint_summary: Dict[str, object],
    anaphora_summary: Dict[str, object],
    result_summary: Dict[str, object],
) -> Dict[str, object]:
    family134 = {row["family_name"]: row for row in joint_summary["family_rows"]}
    family136 = {row["family_name"]: row for row in anaphora_summary["family_rows"]}
    family137 = {row["family_name"]: row for row in result_summary["family_rows"]}

    aligned_rows = []
    for family_name in sorted(set(family134) & set(family136) & set(family137)):
        row134 = family134[family_name]
        row136 = family136[family_name]
        row137 = family137[family_name]
        q_proxy = clamp01(
            0.35 * ((float(row136["noun_pronoun_late_corr"]) + 1.0) / 2.0)
            + 0.35 * ((float(row136["noun_ellipsis_late_corr"]) + 1.0) / 2.0)
            + 0.30 * float(row136["pronoun_sign_consistency_rate"])
        )
        b_proxy = clamp01(float(row134["route_band_gap"]) / 0.08)
        g_proxy = clamp01(
            0.50 * ((float(row134["noun_route_corr"]) + 1.0) / 2.0)
            + 0.50 * ((float(row137["verb_result_corr"]) + 1.0) / 2.0)
        )
        empirical_target = (
            0.45 * float(row137["chain_family_score"])
            + 0.35 * float(row136["family_score"])
            + 0.20 * float(row134["joint_family_score"])
        )
        aligned_rows.append(
            {
                "family_name": family_name,
                "q_proxy": q_proxy,
                "b_proxy": b_proxy,
                "g_proxy": g_proxy,
                "empirical_target": empirical_target,
            }
        )

    values = [round(i * 0.1, 2) for i in range(11)]
    best = None
    for wq in values:
        for wb in values:
            wg = round(1.0 - wq - wb, 2)
            if wg < 0.0 or wg not in values:
                continue
            preds = [wq * row["q_proxy"] + wb * row["b_proxy"] + wg * row["g_proxy"] for row in aligned_rows]
            targets = [float(row["empirical_target"]) for row in aligned_rows]
            corr = correlation(preds, targets)
            mae = sum(abs(pred - target) for pred, target in zip(preds, targets)) / max(1, len(preds))
            fit_score = 0.65 * clamp01((corr + 1.0) / 2.0) + 0.35 * clamp01(1.0 - mae)
            candidate = (fit_score, corr, -mae, wq, wb, wg)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise RuntimeError("Qwen 条件门控场拟合未找到有效权重")
    fit_score, corr, neg_mae, wq, wb, wg = best
    proxy_means = {
        "q_proxy_mean": sum(float(row["q_proxy"]) for row in aligned_rows) / len(aligned_rows),
        "b_proxy_mean": sum(float(row["b_proxy"]) for row in aligned_rows) / len(aligned_rows),
        "g_proxy_mean": sum(float(row["g_proxy"]) for row in aligned_rows) / len(aligned_rows),
    }
    return {
        "family_count": len(aligned_rows),
        "best_formula": f"field = {wq:.2f}*q + {wb:.2f}*b + {wg:.2f}*g",
        "best_weights": {"q": wq, "b": wb, "g": wg},
        "best_correlation": corr,
        "best_mae": -neg_mae,
        "conditional_gating_field_score": fit_score,
        "strongest_proxy_name": max(proxy_means.items(), key=lambda item: item[1])[0],
        "weakest_proxy_name": min(proxy_means.items(), key=lambda item: item[1])[0],
        "stage121_adverb_gate_bridge_score": adverb_summary["adverb_gate_bridge_score"],
        "stage123_route_shift_layer_localization_score": route_summary["route_shift_layer_localization_score"],
        **proxy_means,
        "family_rows": aligned_rows,
    }


def load_gpt2_baselines() -> Dict[str, object]:
    paths = {
        "stage121": PROJECT_ROOT / "tests" / "codex_temp" / "stage121_adverb_gate_bridge_probe_20260323" / "summary.json",
        "stage123": PROJECT_ROOT / "tests" / "codex_temp" / "stage123_route_shift_layer_localization_20260323" / "summary.json",
        "stage124": PROJECT_ROOT / "tests" / "codex_temp" / "stage124_noun_neuron_basic_probe_20260323" / "summary.json",
        "stage130": PROJECT_ROOT / "tests" / "codex_temp" / "stage130_multisyntax_noun_context_probe_20260323" / "summary.json",
        "stage133": PROJECT_ROOT / "tests" / "codex_temp" / "stage133_complex_discourse_noun_propagation_20260323" / "summary.json",
        "stage134": PROJECT_ROOT / "tests" / "codex_temp" / "stage134_noun_verb_joint_propagation_20260323" / "summary.json",
        "stage136": PROJECT_ROOT / "tests" / "codex_temp" / "stage136_anaphora_ellipsis_propagation_20260323" / "summary.json",
        "stage137": PROJECT_ROOT / "tests" / "codex_temp" / "stage137_noun_verb_result_chain_20260323" / "summary.json",
        "stage138": PROJECT_ROOT / "tests" / "codex_temp" / "stage138_conditional_gating_field_reconstruction_20260323" / "summary.json",
    }
    return {name: load_json(path) for name, path in paths.items() if path.exists()}


def build_transfer_summary(
    anchor_layers: Dict[str, int],
    vocab_summary: Dict[str, object],
    adverb_summary: Dict[str, object],
    dynamic_summary: Dict[str, object],
    route_summary: Dict[str, object],
    noun_basic_summary: Dict[str, object],
    noun_context_summary: Dict[str, object],
    discourse_summary: Dict[str, object],
    joint_summary: Dict[str, object],
    anaphora_summary: Dict[str, object],
    result_summary: Dict[str, object],
    field_summary: Dict[str, object],
) -> Dict[str, object]:
    gpt2 = load_gpt2_baselines()
    checks = {
        "adverb_bridge": adverb_summary["adverb_gate_bridge_score"] >= 0.40,
        "route_shift": dynamic_summary["adverb_context_route_shift_score"] >= 0.40,
        "early_anchor_band": noun_context_summary["syntax_stability_rate"] >= 0.80,
        "discourse_remention": discourse_summary["complex_discourse_noun_propagation_score"] >= 0.70,
        "joint_chain": joint_summary["noun_verb_joint_propagation_score"] >= 0.45,
        "result_chain": result_summary["noun_verb_result_chain_score"] >= 0.45,
        "g_dominant_field": field_summary["best_weights"]["g"] >= field_summary["best_weights"]["q"]
        and field_summary["best_weights"]["g"] >= field_summary["best_weights"]["b"],
    }
    pass_rate = sum(1 for hit in checks.values() if hit) / len(checks)
    if pass_rate >= 0.85:
        verdict = "theory_transfer_strong"
    elif pass_rate >= 0.60:
        verdict = "theory_transfer_partial"
    else:
        verdict = "theory_transfer_weak"
    return {
        "mapped_layers": anchor_layers,
        "theory_check_flags": checks,
        "theory_check_pass_rate": pass_rate,
        "transfer_verdict": verdict,
        "gpt2_reference_snapshot": {
            "stage121_adverb_gate_bridge_score": gpt2.get("stage121", {}).get("adverb_gate_bridge_score"),
            "stage123_route_shift_layer_localization_score": gpt2.get("stage123", {}).get("route_shift_layer_localization_score"),
            "stage124_dominant_general_layer_index": gpt2.get("stage124", {}).get("dominant_general_layer_index"),
            "stage130_syntax_stability_rate": gpt2.get("stage130", {}).get("syntax_stability_rate"),
            "stage133_complex_discourse_noun_propagation_score": gpt2.get("stage133", {}).get("complex_discourse_noun_propagation_score"),
            "stage134_noun_verb_joint_propagation_score": gpt2.get("stage134", {}).get("noun_verb_joint_propagation_score"),
            "stage136_anaphora_ellipsis_score": gpt2.get("stage136", {}).get("anaphora_ellipsis_score"),
            "stage137_noun_verb_result_chain_score": gpt2.get("stage137", {}).get("noun_verb_result_chain_score"),
            "stage138_best_formula": gpt2.get("stage138", {}).get("best_formula"),
        },
        "qwen_core_metrics": {
            "clean_unique_word_count": vocab_summary["clean_unique_word_count"],
            "adverb_gate_bridge_score": adverb_summary["adverb_gate_bridge_score"],
            "adverb_context_route_shift_score": dynamic_summary["adverb_context_route_shift_score"],
            "route_shift_layer_localization_score": route_summary["route_shift_layer_localization_score"],
            "noun_basic_dominant_layer_index": noun_basic_summary["dominant_general_layer_index"],
            "syntax_stability_rate": noun_context_summary["syntax_stability_rate"],
            "complex_discourse_noun_propagation_score": discourse_summary["complex_discourse_noun_propagation_score"],
            "noun_verb_joint_propagation_score": joint_summary["noun_verb_joint_propagation_score"],
            "anaphora_ellipsis_score": anaphora_summary["anaphora_ellipsis_score"],
            "noun_verb_result_chain_score": result_summary["noun_verb_result_chain_score"],
            "conditional_gating_field_score": field_summary["conditional_gating_field_score"],
            "conditional_field_formula": field_summary["best_formula"],
        },
    }


def build_report(summary: Dict[str, object]) -> str:
    transfer = summary["transfer_summary"]
    checks = transfer["theory_check_flags"]
    lines = [
        "# Stage139: Qwen3 语言理论迁移验证套件",
        "",
        "## 总结论",
        f"- 迁移结论: {transfer['transfer_verdict']}",
        f"- 理论检查通过率: {transfer['theory_check_pass_rate']:.4f}",
        f"- 映射层: early=L{transfer['mapped_layers']['early_layer']}, route=L{transfer['mapped_layers']['route_layer']}, late=L{transfer['mapped_layers']['late_layer']}",
        "",
        "## 核心指标",
        f"- 全词扫描有效词数: {summary['vocab_summary']['clean_unique_word_count']}",
        f"- 副词桥分数: {summary['adverb_summary']['adverb_gate_bridge_score']:.4f}",
        f"- 副词动态选路分数: {summary['dynamic_summary']['adverb_context_route_shift_score']:.4f}",
        f"- 选路层定位分数: {summary['route_summary']['route_shift_layer_localization_score']:.4f}",
        f"- 名词静态主导层: L{summary['noun_basic_summary']['dominant_general_layer_index']}",
        f"- 多句法稳定率: {summary['noun_context_summary']['syntax_stability_rate']:.4f}",
        f"- 复杂语篇传播分数: {summary['discourse_summary']['complex_discourse_noun_propagation_score']:.4f}",
        f"- 名词-动词联合分数: {summary['joint_summary']['noun_verb_joint_propagation_score']:.4f}",
        f"- 回指/省略分数: {summary['anaphora_summary']['anaphora_ellipsis_score']:.4f}",
        f"- 名词-动词-结果链分数: {summary['result_summary']['noun_verb_result_chain_score']:.4f}",
        f"- 条件门控场分数: {summary['field_summary']['conditional_gating_field_score']:.4f}",
        f"- 条件门控场最优式: `{summary['field_summary']['best_formula']}`",
        "",
        "## 理论检查",
    ]
    for name, hit in checks.items():
        lines.append(f"- {name}: {'通过' if hit else '未通过'}")
    lines.extend(
        [
            "",
            "## 最严格的硬伤",
            "- 这轮验证仍然是代理量层，不是模型原生状态量的直接读出，因此只能说明理论迁移“可测成立”，还不能说明“第一性原理闭合成立”。",
            "- Qwen3 的 early（早层）/ route（选路层）/ late（后层）是按 GPT-2 比例映射出来的，并不等于已经找到两模型之间的严格同构层。",
            "- 如果名词链持续强于动词结果链，说明“定锚”比“选路闭环”更稳定，理论里的 g（门控路由）仍然偏弱。",
            "- 若 b（上下文偏置）继续最弱，就说明语言场还没有从“局部触发”进入“稳定上下文场”。",
            "",
            "## 下一段任务块",
            "- Stage140：Qwen3 与 GPT-2 的跨模型层同构块，专门学习 early / route / late 的真实对齐，而不是继续用比例映射。",
            "- Stage141：Qwen3 的回指与省略强化块，围绕 r（回返一致性）补专门探针，把代词、省略、跨句回指放到更复杂语篇里。",
            "- Stage142：双模型联合变量反演块，把 GPT-2 与 Qwen3 的 a / q / g / b 代理量压到同一坐标系，验证统一变量是否真能跨模型成立。",
        ]
    )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force:
        cached = load_cached_summary(output_dir)
        if cached is not None:
            return cached

    t0 = time.time()
    vocab_summary, rows, embed_weight = run_qwen_vocab_scan(output_dir)
    adverb_summary = run_adverb_bridge(rows, embed_weight)
    model, tokenizer = load_qwen3_model(prefer_cuda=True)
    anchor_layers = resolve_anchor_layers(model)
    dynamic_summary, route_summary = run_route_shift(model, tokenizer, rows, embed_weight)
    noun_basic_summary = run_noun_basic_probe(vocab_summary, rows, model, tokenizer)
    noun_context_summary = run_noun_context_probe(vocab_summary, rows, model, tokenizer, anchor_layers)
    early_neurons = choose_early_neurons(noun_context_summary, anchor_layers)
    late_neurons = choose_late_neurons(noun_basic_summary, anchor_layers)
    discourse_summary = run_complex_discourse_probe(model, tokenizer, rows, early_neurons, late_neurons, anchor_layers)
    route_prototypes = build_qwen_route_prototypes(rows, embed_weight)
    joint_summary = run_noun_verb_joint_probe(model, tokenizer, rows, route_prototypes, early_neurons, anchor_layers)
    anaphora_summary = run_anaphora_probe(model, tokenizer, rows, early_neurons, late_neurons, anchor_layers)
    result_summary = run_result_chain_probe(model, tokenizer, rows, embed_weight, route_prototypes, early_neurons, anchor_layers)
    field_summary = run_conditional_field_fit(adverb_summary, route_summary, joint_summary, anaphora_summary, result_summary)
    transfer_summary = build_transfer_summary(
        anchor_layers,
        vocab_summary,
        adverb_summary,
        dynamic_summary,
        route_summary,
        noun_basic_summary,
        noun_context_summary,
        discourse_summary,
        joint_summary,
        anaphora_summary,
        result_summary,
        field_summary,
    )
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage139_qwen3_language_validation_suite",
        "title": "Qwen3 语言理论迁移验证套件",
        "status_short": "qwen3_language_validation_ready",
        "model_name": "Qwen/Qwen3-4B",
        "model_path": str(QWEN3_MODEL_PATH),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - t0,
        "vocab_summary": vocab_summary,
        "adverb_summary": adverb_summary,
        "dynamic_summary": dynamic_summary,
        "route_summary": route_summary,
        "noun_basic_summary": noun_basic_summary,
        "noun_context_summary": noun_context_summary,
        "discourse_summary": discourse_summary,
        "joint_summary": joint_summary,
        "anaphora_summary": anaphora_summary,
        "result_summary": result_summary,
        "field_summary": field_summary,
        "transfer_summary": transfer_summary,
    }
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3 语言理论迁移验证套件")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存，强制重跑")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "transfer_verdict": summary["transfer_summary"]["transfer_verdict"],
                "theory_check_pass_rate": summary["transfer_summary"]["theory_check_pass_rate"],
                "conditional_field_formula": summary["field_summary"]["best_formula"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
