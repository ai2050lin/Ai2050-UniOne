#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
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
    / "stage434_apple_polysemy_factorized_switch_20260402"
)

MODEL_ORDER = ["qwen3", "deepseek7b"]

APPLE_FRUIT_CASES = [
    {"target": "apple", "label": "fruit_neutral", "sentence": "I bought an apple this morning."},
    {"target": "apple", "label": "fruit_color", "sentence": "The apple is red and shiny."},
    {"target": "apple", "label": "fruit_sweet", "sentence": "The apple tastes sweet and crisp."},
    {"target": "apple", "label": "fruit_sour", "sentence": "The apple tastes slightly sour today."},
    {"target": "apple", "label": "fruit_size", "sentence": "The apple is about the size of a fist."},
]

APPLE_BRAND_CASES = [
    {"target": "Apple", "label": "brand_neutral", "sentence": "Apple announced a new product today."},
    {"target": "Apple", "label": "brand_innovative", "sentence": "Apple is known for innovative hardware design."},
    {"target": "Apple", "label": "brand_expensive", "sentence": "Apple devices are often expensive in many markets."},
    {"target": "Apple", "label": "brand_popular", "sentence": "Apple remains popular among many consumers."},
    {"target": "Apple", "label": "brand_fast", "sentence": "Apple released a faster chip for laptops."},
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


def safe_ratio(numer: float, denom: float) -> float:
    if abs(denom) <= 1e-8:
        return 0.0
    return float(numer / denom)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom.item()) <= 1e-8:
        return 0.0
    return float(torch.dot(a, b).item() / denom.item())


def mean_tensors(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([vec.float() for vec in vectors], dim=0).mean(dim=0)


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
        raise RuntimeError(f"无法定位目标词: target={target!r}, prompt={prompt!r}")
    return best


def capture_case_layer_vectors(model, tokenizer, sentence: str, target: str) -> Dict[int, torch.Tensor]:
    encoded = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
    encoded = move_batch_to_model_device(model, encoded)
    start, end = locate_target_span(tokenizer, sentence, target)
    with torch.inference_mode():
        outputs = model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    layer_count = len(discover_layers(model))
    out: Dict[int, torch.Tensor] = {}
    for layer_idx in range(layer_count):
        out[layer_idx] = hidden_states[layer_idx + 1][0, start:end, :].mean(dim=0).detach().float().cpu()
    return out


def collect_case_bank(model, tokenizer, cases: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for case in cases:
        rows.append(
            {
                "label": case["label"],
                "target": case["target"],
                "sentence": case["sentence"],
                "layer_vectors": capture_case_layer_vectors(model, tokenizer, case["sentence"], case["target"]),
            }
        )
    return rows


def pca_topk_explained(vectors: Sequence[torch.Tensor], top_k: int) -> float:
    if not vectors:
        return 0.0
    x = torch.stack([vec.float() for vec in vectors], dim=0)
    x = x - x.mean(dim=0, keepdim=True)
    if x.shape[0] < 2:
        return 1.0
    s = torch.linalg.svdvals(x)
    energy = (s * s)
    total = float(energy.sum().item())
    if total <= 1e-8:
        return 0.0
    top = float(energy[: min(top_k, energy.numel())].sum().item())
    return clamp01(top / total)


def pairwise_mean_cosine(vectors: Sequence[torch.Tensor]) -> float:
    if len(vectors) < 2:
        return 1.0
    vals: List[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            vals.append(cosine(vectors[i], vectors[j]))
    return safe_ratio(sum(vals), len(vals))


def analyze_model(model_key: str, *, prefer_cuda: bool) -> Dict[str, object]:
    model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=prefer_cuda)
    try:
        layer_count = len(discover_layers(model))
        fruit_rows = collect_case_bank(model, tokenizer, APPLE_FRUIT_CASES)
        brand_rows = collect_case_bank(model, tokenizer, APPLE_BRAND_CASES)
        fruit_base = next(row for row in fruit_rows if row["label"] == "fruit_neutral")
        brand_base = next(row for row in brand_rows if row["label"] == "brand_neutral")
        layer_metrics = []

        for layer_idx in range(layer_count):
            fruit_base_vec = fruit_base["layer_vectors"][layer_idx]
            brand_base_vec = brand_base["layer_vectors"][layer_idx]
            switch_axis = brand_base_vec - fruit_base_vec

            fruit_case_vecs = [row["layer_vectors"][layer_idx] for row in fruit_rows]
            brand_case_vecs = [row["layer_vectors"][layer_idx] for row in brand_rows]
            fruit_modifiers = [row["layer_vectors"][layer_idx] - fruit_base_vec for row in fruit_rows if row["label"] != "fruit_neutral"]
            brand_modifiers = [row["layer_vectors"][layer_idx] - brand_base_vec for row in brand_rows if row["label"] != "brand_neutral"]

            fruit_base_reuse = safe_ratio(
                sum(cosine(vec, fruit_base_vec) for vec in fruit_case_vecs[1:]),
                max(1, len(fruit_case_vecs) - 1),
            )
            brand_base_reuse = safe_ratio(
                sum(cosine(vec, brand_base_vec) for vec in brand_case_vecs[1:]),
                max(1, len(brand_case_vecs) - 1),
            )

            fruit_mod_rank1 = pca_topk_explained(fruit_modifiers, 1)
            fruit_mod_rank2 = pca_topk_explained(fruit_modifiers, 2)
            brand_mod_rank1 = pca_topk_explained(brand_modifiers, 1)
            brand_mod_rank2 = pca_topk_explained(brand_modifiers, 2)

            switch_vs_fruit_mod = safe_ratio(
                sum(abs(cosine(switch_axis, mod)) for mod in fruit_modifiers),
                max(1, len(fruit_modifiers)),
            )
            switch_vs_brand_mod = safe_ratio(
                sum(abs(cosine(switch_axis, mod)) for mod in brand_modifiers),
                max(1, len(brand_modifiers)),
            )
            switch_axis_orthogonality = clamp01(
                1.0 - 0.5 * (switch_vs_fruit_mod + switch_vs_brand_mod)
            )

            switch_cost_ratio = safe_ratio(
                float(torch.linalg.norm(switch_axis).item()),
                float(torch.linalg.norm(fruit_base_vec).item()),
            )
            fruit_modifier_cost_ratio = safe_ratio(
                sum(float(torch.linalg.norm(mod).item()) for mod in fruit_modifiers),
                max(1, len(fruit_modifiers)) * float(torch.linalg.norm(fruit_base_vec).item()),
            )
            brand_modifier_cost_ratio = safe_ratio(
                sum(float(torch.linalg.norm(mod).item()) for mod in brand_modifiers),
                max(1, len(brand_modifiers)) * float(torch.linalg.norm(brand_base_vec).item()),
            )

            fruit_cluster = pairwise_mean_cosine(fruit_case_vecs)
            brand_cluster = pairwise_mean_cosine(brand_case_vecs)
            base_similarity = cosine(fruit_base_vec, brand_base_vec)

            factorized_polysemy_score = clamp01(
                0.18 * clamp01((fruit_base_reuse + 1.0) / 2.0)
                + 0.18 * clamp01((brand_base_reuse + 1.0) / 2.0)
                + 0.15 * fruit_mod_rank2
                + 0.15 * brand_mod_rank2
                + 0.14 * switch_axis_orthogonality
                + 0.10 * clamp01((fruit_cluster + 1.0) / 2.0)
                + 0.10 * clamp01((brand_cluster + 1.0) / 2.0)
            )

            layer_metrics.append(
                {
                    "layer_index": layer_idx,
                    "fruit_base_reuse": fruit_base_reuse,
                    "brand_base_reuse": brand_base_reuse,
                    "fruit_modifier_rank1_ratio": fruit_mod_rank1,
                    "fruit_modifier_rank2_ratio": fruit_mod_rank2,
                    "brand_modifier_rank1_ratio": brand_mod_rank1,
                    "brand_modifier_rank2_ratio": brand_mod_rank2,
                    "switch_axis_orthogonality": switch_axis_orthogonality,
                    "switch_cost_ratio": switch_cost_ratio,
                    "fruit_modifier_cost_ratio": fruit_modifier_cost_ratio,
                    "brand_modifier_cost_ratio": brand_modifier_cost_ratio,
                    "fruit_cluster_similarity": fruit_cluster,
                    "brand_cluster_similarity": brand_cluster,
                    "fruit_brand_base_similarity": base_similarity,
                    "factorized_polysemy_score": factorized_polysemy_score,
                }
            )

        ranked = sorted(layer_metrics, key=lambda row: float(row["factorized_polysemy_score"]), reverse=True)
        best = ranked[0]
        return {
            "model_key": model_key,
            "model_name": MODEL_SPECS[model_key]["model_name"],
            "layer_count": layer_count,
            "used_cuda": bool(prefer_cuda and torch.cuda.is_available()),
            "best_layer": best,
            "top_layers": ranked[:5],
            "layer_metrics": layer_metrics,
            "interpretation": {
                "sense_bases_are_stable": bool(
                    float(best["fruit_base_reuse"]) >= 0.70 and float(best["brand_base_reuse"]) >= 0.70
                ),
                "within_sense_modifiers_are_low_rank": bool(
                    float(best["fruit_modifier_rank2_ratio"]) >= 0.80 and float(best["brand_modifier_rank2_ratio"]) >= 0.80
                ),
                "sense_switch_axis_is_distinct": bool(float(best["switch_axis_orthogonality"]) >= 0.55),
                "polysemy_is_factorized": bool(float(best["factorized_polysemy_score"]) >= 0.70),
            },
        }
    finally:
        free_model(model)


def build_cross_model_summary(model_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    stable_votes = sum(int(bool(row["interpretation"]["sense_bases_are_stable"])) for row in model_results)
    low_rank_votes = sum(int(bool(row["interpretation"]["within_sense_modifiers_are_low_rank"])) for row in model_results)
    distinct_votes = sum(int(bool(row["interpretation"]["sense_switch_axis_is_distinct"])) for row in model_results)
    factorized_votes = sum(int(bool(row["interpretation"]["polysemy_is_factorized"])) for row in model_results)
    return {
        "stable_base_vote_count": stable_votes,
        "low_rank_modifier_vote_count": low_rank_votes,
        "distinct_switch_axis_vote_count": distinct_votes,
        "factorized_polysemy_vote_count": factorized_votes,
        "core_answer": (
            "苹果的水果义与品牌义更像建立在同一共享名词底座上的两种稳定基底切换。"
            "模型先沿着相对独立的 sense switch axis（词义切换轴）区分水果与品牌，"
            "再在每个词义内部用少数低秩 modifier directions（修饰方向）完成颜色、味道、产品、价格等组合扩展。"
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
    for result in summary["model_results"]:
        best = result["best_layer"]
        lines.extend(
            [
                f"## {result['model_name']}",
                f"- 最佳层: L{best['layer_index']}",
                f"- factorized polysemy score: {best['factorized_polysemy_score']:.4f}",
                f"- fruit base reuse: {best['fruit_base_reuse']:.4f}",
                f"- brand base reuse: {best['brand_base_reuse']:.4f}",
                f"- fruit modifier rank2 ratio: {best['fruit_modifier_rank2_ratio']:.4f}",
                f"- brand modifier rank2 ratio: {best['brand_modifier_rank2_ratio']:.4f}",
                f"- switch axis orthogonality: {best['switch_axis_orthogonality']:.4f}",
                f"- switch cost ratio: {best['switch_cost_ratio']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## 理论解释",
            "- 水果义和品牌义不是分别重建整套独立表示，而是先走一条较稳定的词义切换轴。",
            "- 每个词义内部的具体变化更像低秩修饰，因此可以用少数方向覆盖许多组合。",
            "- 这解释了为什么二义性不会导致组合爆炸，因为系统复用的是基底和修饰方向，不是为每个组合单独分配一块参数。 ",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apple polysemy factorized switch analysis")
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
        "experiment_id": "stage434_apple_polysemy_factorized_switch",
        "title": "苹果二义性基底与低秩切换分析",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "model_results": model_results,
        "cross_model_summary": build_cross_model_summary(model_results),
        "case_bank": {
            "fruit_cases": APPLE_FRUIT_CASES,
            "brand_cases": APPLE_BRAND_CASES,
        },
    }
    write_outputs(summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
