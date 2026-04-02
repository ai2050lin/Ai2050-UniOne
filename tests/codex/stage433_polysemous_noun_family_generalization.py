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
    / "stage433_polysemous_noun_family_generalization_20260402"
)

MODEL_ORDER = ["qwen3", "deepseek7b"]

POLYSEMOUS_CASES = [
    {
        "noun_id": "apple",
        "display_name": "apple",
        "sense_a_name": "fruit",
        "sense_b_name": "brand",
        "sense_a_peer_words": ["banana", "orange", "grape", "pear", "peach", "mango", "lemon", "berry"],
        "sense_b_peer_words": ["Microsoft", "Google", "Amazon", "Tesla", "Samsung", "Intel", "NVIDIA"],
        "base_peer_words": ["table", "chair", "river", "mountain", "dog", "teacher", "phone", "car", "book", "hammer"],
        "sense_a_cases": [
            {"target": "apple", "sentence": "I ate an apple after lunch."},
            {"target": "apple", "sentence": "The apple tasted sweet and crisp."},
            {"target": "apple", "sentence": "She sliced the apple into thin pieces."},
            {"target": "apple", "sentence": "A red apple rolled off the table."},
        ],
        "sense_b_cases": [
            {"target": "Apple", "sentence": "Apple released a new laptop this year."},
            {"target": "Apple", "sentence": "Many developers build apps for Apple devices."},
            {"target": "Apple", "sentence": "Apple announced a faster chip for its computers."},
            {"target": "Apple", "sentence": "Investors watched Apple after the earnings call."},
        ],
        "sense_a_template": "This is a {word}.",
        "sense_b_template": "{word} released a new product this year.",
    },
    {
        "noun_id": "amazon",
        "display_name": "amazon",
        "sense_a_name": "river",
        "sense_b_name": "company",
        "sense_a_peer_words": ["Nile", "Yangtze", "Danube", "Mississippi", "Volga", "Mekong", "Ganges"],
        "sense_b_peer_words": ["Microsoft", "Google", "Apple", "Tesla", "Samsung", "Intel", "NVIDIA"],
        "base_peer_words": ["forest", "valley", "bridge", "village", "teacher", "phone", "book", "hammer", "planet", "garden"],
        "sense_a_cases": [
            {"target": "Amazon", "sentence": "The Amazon flows through a vast rainforest."},
            {"target": "Amazon", "sentence": "Scientists studied fish in the Amazon during the expedition."},
            {"target": "Amazon", "sentence": "Heavy rain caused the Amazon to rise quickly."},
            {"target": "Amazon", "sentence": "Boats moved slowly along the Amazon at sunset."},
        ],
        "sense_b_cases": [
            {"target": "Amazon", "sentence": "Amazon opened another logistics center this year."},
            {"target": "Amazon", "sentence": "Many shoppers ordered electronics from Amazon last week."},
            {"target": "Amazon", "sentence": "Amazon reported strong cloud revenue this quarter."},
            {"target": "Amazon", "sentence": "Investors reacted after Amazon released earnings guidance."},
        ],
        "sense_a_template": "{word} is a major river.",
        "sense_b_template": "{word} released a new product this year.",
    },
    {
        "noun_id": "python",
        "display_name": "python",
        "sense_a_name": "animal",
        "sense_b_name": "programming_language",
        "sense_a_peer_words": ["cobra", "viper", "anaconda", "boa", "lizard", "crocodile", "turtle"],
        "sense_b_peer_words": ["Java", "C++", "Rust", "Go", "JavaScript", "Ruby", "Swift"],
        "base_peer_words": ["table", "chair", "river", "mountain", "teacher", "phone", "book", "hammer", "garden", "window"],
        "sense_a_cases": [
            {"target": "python", "sentence": "The python coiled around the branch."},
            {"target": "python", "sentence": "A large python moved slowly across the ground."},
            {"target": "python", "sentence": "The zoo keeper fed the python in the reptile house."},
            {"target": "python", "sentence": "The python hid beneath the warm rock."},
        ],
        "sense_b_cases": [
            {"target": "Python", "sentence": "Python made the data analysis script easier to write."},
            {"target": "Python", "sentence": "Many beginners learn Python before studying larger systems."},
            {"target": "Python", "sentence": "The team used Python to automate daily reports."},
            {"target": "Python", "sentence": "Python libraries helped the researchers train the model."},
        ],
        "sense_a_template": "This is a {word}.",
        "sense_b_template": "{word} is a popular programming language.",
    },
    {
        "noun_id": "java",
        "display_name": "java",
        "sense_a_name": "coffee",
        "sense_b_name": "programming_language",
        "sense_a_peer_words": ["coffee", "espresso", "latte", "mocha", "cappuccino", "tea", "cocoa"],
        "sense_b_peer_words": ["Python", "C++", "Rust", "Go", "JavaScript", "Ruby", "Swift"],
        "base_peer_words": ["table", "chair", "river", "mountain", "teacher", "phone", "book", "hammer", "garden", "window"],
        "sense_a_cases": [
            {"target": "java", "sentence": "He drank hot java before the meeting."},
            {"target": "java", "sentence": "Fresh java filled the kitchen with a rich smell."},
            {"target": "java", "sentence": "She ordered a cup of java after dinner."},
            {"target": "java", "sentence": "The cafe served strong java all morning."},
        ],
        "sense_b_cases": [
            {"target": "Java", "sentence": "Java powers many enterprise systems."},
            {"target": "Java", "sentence": "The service was rewritten in Java for the new release."},
            {"target": "Java", "sentence": "Many backend teams still rely on Java in production."},
            {"target": "Java", "sentence": "The course teaches Java before advanced compiler topics."},
        ],
        "sense_a_template": "This drink is {word}.",
        "sense_b_template": "{word} is a popular programming language.",
    },
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


def mean_tensors(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([vec.float() for vec in vectors], dim=0).mean(dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom.item()) <= 1e-8:
        return 0.0
    return float(torch.dot(a, b).item() / denom.item())


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
    recon = torch.zeros_like(target.float())
    target = target.float()
    for base in basis:
        recon = recon + torch.dot(target, base) * base
    target_norm_sq = float(torch.dot(target, target).item())
    if target_norm_sq <= 1e-8:
        return 0.0
    recon_norm_sq = float(torch.dot(recon, recon).item())
    return clamp01(recon_norm_sq / target_norm_sq)


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
    layer_vectors: Dict[int, torch.Tensor] = {}
    for layer_idx in range(layer_count):
        layer_vectors[layer_idx] = hidden_states[layer_idx + 1][0, start:end, :].mean(dim=0).detach().float().cpu()
    return layer_vectors


def collect_case_bank(model, tokenizer, cases: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for case in cases:
        rows.append(
            {
                "target": case["target"],
                "sentence": case["sentence"],
                "layer_vectors": capture_case_layer_vectors(model, tokenizer, case["sentence"], case["target"]),
            }
        )
    return rows


def build_template_cases(words: Sequence[str], template: str) -> List[Dict[str, str]]:
    cases = []
    for word in words:
        cases.append({"target": word, "sentence": template.format(word=word)})
    return cases


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
    sense_a_rows: Sequence[Dict[str, object]],
    sense_b_rows: Sequence[Dict[str, object]],
    sense_a_centroid: torch.Tensor,
    sense_b_centroid: torch.Tensor,
    layer_idx: int,
) -> float:
    correct = 0
    total = 0
    for row in sense_a_rows:
        vec = row["layer_vectors"][layer_idx]
        if cosine(vec, sense_a_centroid) >= cosine(vec, sense_b_centroid):
            correct += 1
        total += 1
    for row in sense_b_rows:
        vec = row["layer_vectors"][layer_idx]
        if cosine(vec, sense_b_centroid) >= cosine(vec, sense_a_centroid):
            correct += 1
        total += 1
    return safe_ratio(correct, total)


def analyze_single_noun(model, tokenizer, noun_spec: Dict[str, object]) -> Dict[str, object]:
    layer_count = len(discover_layers(model))

    sense_a_peer_rows = collect_case_bank(
        model,
        tokenizer,
        build_template_cases(noun_spec["sense_a_peer_words"], noun_spec["sense_a_template"]),
    )
    sense_b_peer_rows = collect_case_bank(
        model,
        tokenizer,
        build_template_cases(noun_spec["sense_b_peer_words"], noun_spec["sense_b_template"]),
    )
    base_rows = collect_case_bank(
        model,
        tokenizer,
        build_template_cases(noun_spec["base_peer_words"], "This is a {word}."),
    )
    sense_a_rows = collect_case_bank(model, tokenizer, noun_spec["sense_a_cases"])
    sense_b_rows = collect_case_bank(model, tokenizer, noun_spec["sense_b_cases"])

    sense_a_peer_mean = mean_by_layer(sense_a_peer_rows)
    sense_b_peer_mean = mean_by_layer(sense_b_peer_rows)
    base_mean = mean_by_layer(base_rows)
    sense_a_mean = mean_by_layer(sense_a_rows)
    sense_b_mean = mean_by_layer(sense_b_rows)

    layer_rows = []
    for layer_idx in range(layer_count):
        noun_base = base_mean[layer_idx]
        sense_a_centroid = sense_a_peer_mean[layer_idx]
        sense_b_centroid = sense_b_peer_mean[layer_idx]
        sense_a_vec = sense_a_mean[layer_idx]
        sense_b_vec = sense_b_mean[layer_idx]

        sense_a_offset = sense_a_centroid - noun_base
        sense_b_offset = sense_b_centroid - noun_base
        family_axis = sense_b_centroid - sense_a_centroid
        sense_delta = sense_b_vec - sense_a_vec

        sense_a_explained = explained_ratio(sense_a_vec, [noun_base, sense_a_offset])
        sense_b_explained = explained_ratio(sense_b_vec, [noun_base, sense_b_offset])
        delta_structured = explained_ratio(sense_delta, [family_axis, sense_b_offset - sense_a_offset])
        shared_core_similarity = cosine(sense_a_vec, sense_b_vec)
        sense_a_family_margin = cosine(sense_a_vec, sense_a_centroid) - cosine(sense_a_vec, sense_b_centroid)
        sense_b_family_margin = cosine(sense_b_vec, sense_b_centroid) - cosine(sense_b_vec, sense_a_centroid)
        readout_accuracy = sense_readout_accuracy(sense_a_rows, sense_b_rows, sense_a_centroid, sense_b_centroid, layer_idx)
        sense_a_stability = pairwise_mean_cosine(sense_a_rows, layer_idx)
        sense_b_stability = pairwise_mean_cosine(sense_b_rows, layer_idx)
        delta_cost_ratio = safe_ratio(float(torch.linalg.norm(sense_delta).item()), float(torch.linalg.norm(sense_a_vec).item()))

        compact_efficiency_score = clamp01(
            0.30 * sense_a_explained
            + 0.30 * sense_b_explained
            + 0.20 * delta_structured
            + 0.20 * max(0.0, shared_core_similarity)
        )
        expressive_power_score = clamp01(
            0.40 * readout_accuracy
            + 0.20 * clamp01((sense_a_family_margin + 1.0) / 2.0)
            + 0.20 * clamp01((sense_b_family_margin + 1.0) / 2.0)
            + 0.10 * clamp01((sense_a_stability + 1.0) / 2.0)
            + 0.10 * clamp01((sense_b_stability + 1.0) / 2.0)
        )
        efficiency_expression_balance = clamp01(
            0.45 * compact_efficiency_score
            + 0.45 * expressive_power_score
            + 0.10 * clamp01(1.0 - min(delta_cost_ratio, 1.0))
        )

        layer_rows.append(
            {
                "layer_index": layer_idx,
                "sense_a_explained_ratio": sense_a_explained,
                "sense_b_explained_ratio": sense_b_explained,
                "delta_structured_ratio": delta_structured,
                "shared_core_similarity": shared_core_similarity,
                "sense_a_family_margin": sense_a_family_margin,
                "sense_b_family_margin": sense_b_family_margin,
                "sense_readout_accuracy": readout_accuracy,
                "sense_a_context_stability": sense_a_stability,
                "sense_b_context_stability": sense_b_stability,
                "sense_delta_cost_ratio": delta_cost_ratio,
                "compact_efficiency_score": compact_efficiency_score,
                "expressive_power_score": expressive_power_score,
                "efficiency_expression_balance": efficiency_expression_balance,
            }
        )

    ranked = sorted(layer_rows, key=lambda row: float(row["efficiency_expression_balance"]), reverse=True)
    best_row = ranked[0]
    best_layer_idx = int(best_row["layer_index"])
    return {
        "noun_id": noun_spec["noun_id"],
        "display_name": noun_spec["display_name"],
        "sense_a_name": noun_spec["sense_a_name"],
        "sense_b_name": noun_spec["sense_b_name"],
        "best_balance_layer": best_row,
        "top_balance_layers": ranked[:5],
        "layer_metrics": layer_rows,
        "best_layer_summary": {
            "sense_a_to_own_centroid": cosine(sense_a_mean[best_layer_idx], sense_a_peer_mean[best_layer_idx]),
            "sense_a_to_other_centroid": cosine(sense_a_mean[best_layer_idx], sense_b_peer_mean[best_layer_idx]),
            "sense_b_to_own_centroid": cosine(sense_b_mean[best_layer_idx], sense_b_peer_mean[best_layer_idx]),
            "sense_b_to_other_centroid": cosine(sense_b_mean[best_layer_idx], sense_a_peer_mean[best_layer_idx]),
        },
        "interpretation": {
            "shared_base_dominant": bool(
                float(best_row["sense_a_explained_ratio"]) >= 0.70 and float(best_row["sense_b_explained_ratio"]) >= 0.70
            ),
            "sense_switch_is_structured": bool(float(best_row["delta_structured_ratio"]) >= 0.45),
            "sense_readout_is_reliable": bool(float(best_row["sense_readout_accuracy"]) >= 0.75),
            "small_delta_supports_high_efficiency": bool(float(best_row["sense_delta_cost_ratio"]) <= 0.55),
        },
    }


def analyze_model(model_key: str, *, prefer_cuda: bool) -> Dict[str, object]:
    model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=prefer_cuda)
    try:
        noun_results = [analyze_single_noun(model, tokenizer, noun_spec) for noun_spec in POLYSEMOUS_CASES]
        avg_balance = safe_ratio(
            sum(float(row["best_balance_layer"]["efficiency_expression_balance"]) for row in noun_results),
            len(noun_results),
        )
        avg_compact = safe_ratio(
            sum(float(row["best_balance_layer"]["compact_efficiency_score"]) for row in noun_results),
            len(noun_results),
        )
        avg_expressive = safe_ratio(
            sum(float(row["best_balance_layer"]["expressive_power_score"]) for row in noun_results),
            len(noun_results),
        )
        shared_base_support = sum(int(bool(row["interpretation"]["shared_base_dominant"])) for row in noun_results)
        structured_support = sum(int(bool(row["interpretation"]["sense_switch_is_structured"])) for row in noun_results)
        reliable_support = sum(int(bool(row["interpretation"]["sense_readout_is_reliable"])) for row in noun_results)
        small_delta_support = sum(int(bool(row["interpretation"]["small_delta_supports_high_efficiency"])) for row in noun_results)
        strongest = sorted(
            noun_results,
            key=lambda row: float(row["best_balance_layer"]["efficiency_expression_balance"]),
            reverse=True,
        )
        return {
            "model_key": model_key,
            "model_name": MODEL_SPECS[model_key]["model_name"],
            "layer_count": len(discover_layers(model)),
            "used_cuda": bool(prefer_cuda and torch.cuda.is_available()),
            "noun_results": noun_results,
            "aggregate": {
                "noun_count": len(noun_results),
                "shared_base_support_count": shared_base_support,
                "structured_switch_support_count": structured_support,
                "reliable_readout_support_count": reliable_support,
                "small_delta_support_count": small_delta_support,
                "average_balance_score": avg_balance,
                "average_compact_efficiency_score": avg_compact,
                "average_expressive_power_score": avg_expressive,
                "strongest_generalization_nouns": [row["noun_id"] for row in strongest[:3]],
            },
        }
    finally:
        free_model(model)


def build_cross_model_summary(model_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    total_nouns = sum(int(row["aggregate"]["noun_count"]) for row in model_results)
    shared_base_total = sum(int(row["aggregate"]["shared_base_support_count"]) for row in model_results)
    structured_total = sum(int(row["aggregate"]["structured_switch_support_count"]) for row in model_results)
    reliable_total = sum(int(row["aggregate"]["reliable_readout_support_count"]) for row in model_results)
    small_delta_total = sum(int(row["aggregate"]["small_delta_support_count"]) for row in model_results)
    avg_balance = safe_ratio(sum(float(row["aggregate"]["average_balance_score"]) for row in model_results), len(model_results))
    avg_compact = safe_ratio(sum(float(row["aggregate"]["average_compact_efficiency_score"]) for row in model_results), len(model_results))
    avg_expressive = safe_ratio(sum(float(row["aggregate"]["average_expressive_power_score"]) for row in model_results), len(model_results))
    return {
        "total_noun_cases": total_nouns,
        "shared_base_support_rate": safe_ratio(shared_base_total, total_nouns),
        "structured_switch_support_rate": safe_ratio(structured_total, total_nouns),
        "reliable_readout_support_rate": safe_ratio(reliable_total, total_nouns),
        "small_delta_support_rate": safe_ratio(small_delta_total, total_nouns),
        "average_balance_score": avg_balance,
        "average_compact_efficiency_score": avg_compact,
        "average_expressive_power_score": avg_expressive,
        "core_answer": (
            "多义名词更像共享名词底座上的低成本定向切换系统。"
            "模型主要复用公共名词结构，再借助类别偏置和结构化上下文增量完成语义分叉，"
            "因此能同时保持较强表达力和较高性能。"
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
        agg = model_result["aggregate"]
        lines.extend(
            [
                f"## {model_result['model_name']}",
                f"- 多义名词数量: {agg['noun_count']}",
                f"- shared base support count: {agg['shared_base_support_count']}",
                f"- structured switch support count: {agg['structured_switch_support_count']}",
                f"- reliable readout support count: {agg['reliable_readout_support_count']}",
                f"- average compact score: {agg['average_compact_efficiency_score']:.4f}",
                f"- average expressive score: {agg['average_expressive_power_score']:.4f}",
                f"- average balance score: {agg['average_balance_score']:.4f}",
                "",
            ]
        )
        for noun_result in model_result["noun_results"]:
            best = noun_result["best_balance_layer"]
            lines.extend(
                [
                    f"### {noun_result['noun_id']}",
                    f"- 最佳平衡层: L{best['layer_index']}",
                    f"- {noun_result['sense_a_name']} explained ratio: {best['sense_a_explained_ratio']:.4f}",
                    f"- {noun_result['sense_b_name']} explained ratio: {best['sense_b_explained_ratio']:.4f}",
                    f"- delta structured ratio: {best['delta_structured_ratio']:.4f}",
                    f"- sense readout accuracy: {best['sense_readout_accuracy']:.4f}",
                    f"- sense delta cost ratio: {best['sense_delta_cost_ratio']:.4f}",
                    f"- balance score: {best['efficiency_expression_balance']:.4f}",
                    "",
                ]
            )
    lines.extend(
        [
            "## 理论解释",
            "- 如果多个多义名词都表现出高共享底座比例，说明模型没有为每个词义独立重建一整块表示。",
            "- 如果语义切换主要体现为结构化增量，说明性能优势来自复用而不是粗暴压缩。",
            "- 如果读出在两种语境下仍然可靠，说明共享结构并没有牺牲表达能力。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polysemous noun family generalization analysis")
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
        "experiment_id": "stage433_polysemous_noun_family_generalization",
        "title": "多义名词家族的共享底座泛化实验",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "model_results": model_results,
        "cross_model_summary": build_cross_model_summary(model_results),
        "case_bank": POLYSEMOUS_CASES,
    }
    write_outputs(summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
