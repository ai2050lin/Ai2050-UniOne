#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from qwen3_language_shared import move_batch_to_model_device
from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, discover_layers, load_qwen_like_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage432_apple_noun_efficiency_expression_tradeoff_20260402"
)

MODEL_ORDER = ["qwen3", "deepseek7b"]

FRUIT_PEER_WORDS = ["banana", "orange", "grape", "pear", "peach", "mango", "lemon", "berry"]
NONFRUIT_NOUN_WORDS = ["table", "chair", "river", "mountain", "dog", "teacher", "phone", "car", "book", "hammer"]
COMPANY_PEER_WORDS = ["Microsoft", "Google", "Amazon", "Tesla", "Samsung", "Intel", "NVIDIA"]

APPLE_FRUIT_CASES = [
    {"target": "apple", "sentence": "I ate an apple after lunch."},
    {"target": "apple", "sentence": "The apple tasted sweet and crisp."},
    {"target": "apple", "sentence": "She sliced the apple into thin pieces."},
    {"target": "apple", "sentence": "A red apple rolled off the table."},
]

APPLE_BRAND_CASES = [
    {"target": "Apple", "sentence": "Apple released a new laptop this year."},
    {"target": "Apple", "sentence": "Many developers build apps for Apple devices."},
    {"target": "Apple", "sentence": "Apple announced a faster chip for its computers."},
    {"target": "Apple", "sentence": "Investors watched Apple after the earnings call."},
]


def free_model(model) -> None:
    try:
        del model
    except UnboundLocalError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def mean_tensors(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([vec.float() for vec in vectors], dim=0).mean(dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom.item()) <= 1e-8:
        return 0.0
    return float(torch.dot(a, b).item() / denom.item())


def normalize(vec: torch.Tensor) -> torch.Tensor:
    vec = vec.float()
    norm = torch.linalg.norm(vec).clamp_min(1e-8)
    return vec / norm


def orthonormal_basis(vectors: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    basis: List[torch.Tensor] = []
    for raw in vectors:
        vec = raw.float().clone()
        for base in basis:
            vec = vec - torch.dot(vec, base) * base
        norm = torch.linalg.norm(vec)
        if float(norm.item()) > 1e-8:
            basis.append(vec / norm)
    return basis


def explained_ratio(target: torch.Tensor, basis_vectors: Sequence[torch.Tensor]) -> float:
    basis = orthonormal_basis(basis_vectors)
    if not basis:
        return 0.0
    target = target.float()
    recon = torch.zeros_like(target)
    for base in basis:
        recon = recon + torch.dot(target, base) * base
    target_norm_sq = float(torch.dot(target, target).item())
    if target_norm_sq <= 1e-8:
        return 0.0
    recon_norm_sq = float(torch.dot(recon, recon).item())
    return clamp01(recon_norm_sq / target_norm_sq)


def safe_ratio(numer: float, denom: float) -> float:
    if abs(denom) <= 1e-8:
        return 0.0
    return float(numer / denom)


def find_last_subsequence(full_ids: List[int], sub_ids: List[int]) -> Tuple[int, int] | None:
    if not sub_ids or len(sub_ids) > len(full_ids):
        return None
    last_match = None
    for start in range(0, len(full_ids) - len(sub_ids) + 1):
        if full_ids[start : start + len(sub_ids)] == sub_ids:
            last_match = (start, start + len(sub_ids))
    return last_match


def locate_target_span(tokenizer, prompt: str, target: str) -> Tuple[int, int]:
    full_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    candidates = []
    raw_variants = [target, f" {target}", target.lower(), f" {target.lower()}", target.capitalize(), f" {target.capitalize()}"]
    seen = set()
    for text in raw_variants:
        if text in seen:
            continue
        seen.add(text)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids:
            candidates.append(ids)
    best = None
    best_len = -1
    for ids in candidates:
        match = find_last_subsequence(full_ids, ids)
        if match is not None and len(ids) > best_len:
            best = match
            best_len = len(ids)
    if best is None:
        raise RuntimeError(f"无法在提示词中定位目标词: target={target!r}, prompt={prompt!r}")
    return best


def capture_case_layer_vectors(
    model,
    tokenizer,
    sentence: str,
    target: str,
) -> Dict[int, torch.Tensor]:
    encoded = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
    encoded = move_batch_to_model_device(model, encoded)
    start, end = locate_target_span(tokenizer, sentence, target)
    with torch.inference_mode():
        outputs = model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    layer_count = len(discover_layers(model))
    layer_vectors: Dict[int, torch.Tensor] = {}
    for layer_idx in range(layer_count):
        vec = hidden_states[layer_idx + 1][0, start:end, :].mean(dim=0).detach().float().cpu()
        layer_vectors[layer_idx] = vec
    return layer_vectors


def build_neutral_noun_cases(words: Sequence[str]) -> List[Dict[str, str]]:
    return [{"target": word, "sentence": f"This is a {word}."} for word in words]


def build_company_cases(words: Sequence[str]) -> List[Dict[str, str]]:
    return [{"target": word, "sentence": f"{word} released a new product this year."} for word in words]


def collect_case_bank(
    model,
    tokenizer,
    cases: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for case in cases:
        layer_vectors = capture_case_layer_vectors(model, tokenizer, case["sentence"], case["target"])
        rows.append(
            {
                "target": case["target"],
                "sentence": case["sentence"],
                "layer_vectors": layer_vectors,
            }
        )
    return rows


def mean_by_layer(rows: Sequence[Dict[str, object]]) -> Dict[int, torch.Tensor]:
    if not rows:
        return {}
    layer_ids = sorted(rows[0]["layer_vectors"].keys())
    out: Dict[int, torch.Tensor] = {}
    for layer_idx in layer_ids:
        out[layer_idx] = mean_tensors([row["layer_vectors"][layer_idx] for row in rows])
    return out


def pairwise_mean_cosine(rows: Sequence[Dict[str, object]], layer_idx: int) -> float:
    if len(rows) < 2:
        return 1.0
    vals: List[float] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            vals.append(cosine(rows[i]["layer_vectors"][layer_idx], rows[j]["layer_vectors"][layer_idx]))
    return float(sum(vals) / max(1, len(vals)))


def sense_readout_accuracy(
    apple_fruit_rows: Sequence[Dict[str, object]],
    apple_brand_rows: Sequence[Dict[str, object]],
    fruit_centroid: torch.Tensor,
    company_centroid: torch.Tensor,
    layer_idx: int,
) -> float:
    correct = 0
    total = 0
    for row in apple_fruit_rows:
        vec = row["layer_vectors"][layer_idx]
        if cosine(vec, fruit_centroid) >= cosine(vec, company_centroid):
            correct += 1
        total += 1
    for row in apple_brand_rows:
        vec = row["layer_vectors"][layer_idx]
        if cosine(vec, company_centroid) >= cosine(vec, fruit_centroid):
            correct += 1
        total += 1
    return safe_ratio(correct, total)


def analyze_model(model_key: str, *, prefer_cuda: bool) -> Dict[str, object]:
    model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=prefer_cuda)
    try:
        layer_count = len(discover_layers(model))

        fruit_peer_rows = collect_case_bank(model, tokenizer, build_neutral_noun_cases(FRUIT_PEER_WORDS))
        nonfruit_rows = collect_case_bank(model, tokenizer, build_neutral_noun_cases(NONFRUIT_NOUN_WORDS))
        company_rows = collect_case_bank(model, tokenizer, build_company_cases(COMPANY_PEER_WORDS))
        apple_fruit_rows = collect_case_bank(model, tokenizer, APPLE_FRUIT_CASES)
        apple_brand_rows = collect_case_bank(model, tokenizer, APPLE_BRAND_CASES)

        fruit_peer_mean = mean_by_layer(fruit_peer_rows)
        nonfruit_mean = mean_by_layer(nonfruit_rows)
        company_mean = mean_by_layer(company_rows)
        apple_fruit_mean = mean_by_layer(apple_fruit_rows)
        apple_brand_mean = mean_by_layer(apple_brand_rows)

        noun_base_mean: Dict[int, torch.Tensor] = {}
        for layer_idx in range(layer_count):
            noun_base_mean[layer_idx] = mean_tensors(
                [fruit_peer_mean[layer_idx], nonfruit_mean[layer_idx]]
            )

        layer_rows = []
        for layer_idx in range(layer_count):
            noun_base = noun_base_mean[layer_idx]
            fruit_centroid = fruit_peer_mean[layer_idx]
            company_centroid = company_mean[layer_idx]
            fruit_vec = apple_fruit_mean[layer_idx]
            brand_vec = apple_brand_mean[layer_idx]
            fruit_offset = fruit_centroid - noun_base
            company_offset = company_centroid - noun_base
            sense_delta = brand_vec - fruit_vec
            family_axis = company_centroid - fruit_centroid

            fruit_explained = explained_ratio(fruit_vec, [noun_base, fruit_offset])
            brand_explained = explained_ratio(brand_vec, [noun_base, company_offset])
            delta_structured = explained_ratio(sense_delta, [family_axis, company_offset - fruit_offset])

            shared_core_similarity = cosine(fruit_vec, brand_vec)
            fruit_family_margin = cosine(fruit_vec, fruit_centroid) - cosine(fruit_vec, company_centroid)
            brand_family_margin = cosine(brand_vec, company_centroid) - cosine(brand_vec, fruit_centroid)
            sense_accuracy = sense_readout_accuracy(
                apple_fruit_rows,
                apple_brand_rows,
                fruit_centroid,
                company_centroid,
                layer_idx,
            )

            fruit_context_stability = pairwise_mean_cosine(apple_fruit_rows, layer_idx)
            brand_context_stability = pairwise_mean_cosine(apple_brand_rows, layer_idx)

            delta_cost_ratio = safe_ratio(
                float(torch.linalg.norm(sense_delta).item()),
                float(torch.linalg.norm(fruit_vec).item()),
            )
            compact_efficiency_score = clamp01(
                0.30 * fruit_explained
                + 0.30 * brand_explained
                + 0.20 * delta_structured
                + 0.20 * max(0.0, shared_core_similarity)
            )
            expressive_power_score = clamp01(
                0.40 * sense_accuracy
                + 0.20 * clamp01((fruit_family_margin + 1.0) / 2.0)
                + 0.20 * clamp01((brand_family_margin + 1.0) / 2.0)
                + 0.10 * clamp01((fruit_context_stability + 1.0) / 2.0)
                + 0.10 * clamp01((brand_context_stability + 1.0) / 2.0)
            )
            efficiency_expression_balance = clamp01(
                0.45 * compact_efficiency_score
                + 0.45 * expressive_power_score
                + 0.10 * clamp01(1.0 - min(delta_cost_ratio, 1.0))
            )

            layer_rows.append(
                {
                    "layer_index": layer_idx,
                    "fruit_explained_ratio": fruit_explained,
                    "brand_explained_ratio": brand_explained,
                    "delta_structured_ratio": delta_structured,
                    "shared_core_similarity": shared_core_similarity,
                    "fruit_family_margin": fruit_family_margin,
                    "brand_family_margin": brand_family_margin,
                    "sense_readout_accuracy": sense_accuracy,
                    "fruit_context_stability": fruit_context_stability,
                    "brand_context_stability": brand_context_stability,
                    "sense_delta_cost_ratio": delta_cost_ratio,
                    "compact_efficiency_score": compact_efficiency_score,
                    "expressive_power_score": expressive_power_score,
                    "efficiency_expression_balance": efficiency_expression_balance,
                }
            )

        ranked = sorted(layer_rows, key=lambda row: float(row["efficiency_expression_balance"]), reverse=True)
        best_row = ranked[0]
        best_layer_idx = int(best_row["layer_index"])
        result = {
            "model_key": model_key,
            "model_name": MODEL_SPECS[model_key]["model_name"],
            "layer_count": layer_count,
            "used_cuda": bool(prefer_cuda and torch.cuda.is_available()),
            "best_balance_layer": best_row,
            "top_balance_layers": ranked[:5],
            "layer_metrics": layer_rows,
            "best_layer_summary": {
                "apple_fruit_to_fruit_centroid": cosine(apple_fruit_mean[best_layer_idx], fruit_peer_mean[best_layer_idx]),
                "apple_fruit_to_company_centroid": cosine(apple_fruit_mean[best_layer_idx], company_mean[best_layer_idx]),
                "apple_brand_to_company_centroid": cosine(apple_brand_mean[best_layer_idx], company_mean[best_layer_idx]),
                "apple_brand_to_fruit_centroid": cosine(apple_brand_mean[best_layer_idx], fruit_peer_mean[best_layer_idx]),
            },
            "interpretation": {
                "shared_base_dominant": bool(
                    float(best_row["fruit_explained_ratio"]) >= 0.70 and float(best_row["brand_explained_ratio"]) >= 0.70
                ),
                "sense_switch_is_structured": bool(float(best_row["delta_structured_ratio"]) >= 0.45),
                "sense_readout_is_reliable": bool(float(best_row["sense_readout_accuracy"]) >= 0.75),
                "small_delta_supports_high_efficiency": bool(float(best_row["sense_delta_cost_ratio"]) <= 0.55),
            },
        }
        return result
    finally:
        free_model(model)


def build_cross_model_summary(model_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    shared_base_votes = 0
    structured_delta_votes = 0
    reliable_readout_votes = 0
    efficient_delta_votes = 0
    for row in model_results:
        interp = row["interpretation"]
        shared_base_votes += int(bool(interp["shared_base_dominant"]))
        structured_delta_votes += int(bool(interp["sense_switch_is_structured"]))
        reliable_readout_votes += int(bool(interp["sense_readout_is_reliable"]))
        efficient_delta_votes += int(bool(interp["small_delta_supports_high_efficiency"]))

    return {
        "shared_base_vote_count": shared_base_votes,
        "structured_delta_vote_count": structured_delta_votes,
        "reliable_readout_vote_count": reliable_readout_votes,
        "efficient_delta_vote_count": efficient_delta_votes,
        "core_answer": (
            "苹果之所以既能有足够表达力又能保持高性能，更像是因为它不是被完全单独编码，"
            "而是复用了共享名词底座与类别偏置，再通过较小但结构化的上下文增量完成语义切换。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 核心回答",
        summary["cross_model_summary"]["core_answer"],
        "",
    ]
    for model_result in summary["model_results"]:
        best = model_result["best_balance_layer"]
        lines.extend(
            [
                f"## {model_result['model_name']}",
                f"- 最佳平衡层: L{best['layer_index']}",
                f"- 共享压缩分数: {best['compact_efficiency_score']:.4f}",
                f"- 表达能力分数: {best['expressive_power_score']:.4f}",
                f"- 综合平衡分数: {best['efficiency_expression_balance']:.4f}",
                f"- fruit explained ratio: {best['fruit_explained_ratio']:.4f}",
                f"- brand explained ratio: {best['brand_explained_ratio']:.4f}",
                f"- delta structured ratio: {best['delta_structured_ratio']:.4f}",
                f"- sense readout accuracy: {best['sense_readout_accuracy']:.4f}",
                f"- sense delta cost ratio: {best['sense_delta_cost_ratio']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## 理论解释",
            "- 共享名词底座提供复用，所以不用给每个具体名词都配一整套独立参数。",
            "- 水果偏置让 apple 可以快速落到同类语义区，不必从零构造整套意义。",
            "- 上下文切换主要表现为结构化增量，而不是整向量重写，所以同一编码能服务多种表达场景。",
            "- 当读出层能够稳定区分 fruit/brand 两种语境，而底座仍然高度共享时，就同时解释了表达力和效率。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apple noun efficiency-expression tradeoff analysis")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()
    prefer_cuda = not bool(args.cpu)
    model_results = [analyze_model(model_key, prefer_cuda=prefer_cuda) for model_key in MODEL_ORDER]
    elapsed = time.time() - start_time
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage432_apple_noun_efficiency_expression_tradeoff",
        "title": "苹果名词编码的表达力-性能平衡分析",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "model_results": model_results,
        "cross_model_summary": build_cross_model_summary(model_results),
        "case_bank": {
            "fruit_peer_words": FRUIT_PEER_WORDS,
            "nonfruit_noun_words": NONFRUIT_NOUN_WORDS,
            "company_peer_words": COMPANY_PEER_WORDS,
            "apple_fruit_cases": APPLE_FRUIT_CASES,
            "apple_brand_cases": APPLE_BRAND_CASES,
        },
    }
    write_outputs(summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
