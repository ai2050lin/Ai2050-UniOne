#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage638: 多能力统一编码协议

目标：
1. 用同一套口径同时评估五类语言能力：
   - 多义消歧
   - 语法一致性
   - 关系抽取
   - 指代消解
   - 风格控制
2. 对每个能力样本统一测量：
   - 正确/错误候选 margin
   - preference_correct
   - 初层到末层差异向量范数增长
   - 逐层平均旋转角
   - 末层差异与候选方向对齐
   - 中层差异强度与中层对齐
   - 差异峰值层
3. 为后续“跨能力共享不变量”研究提供统一基线。

用法：
python tests/codex/stage638_multicapability_unified_protocol.py --model qwen3
python tests/codex/stage638_multicapability_unified_protocol.py --model qwen3 --capability syntax --max-pairs 1
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import discover_layers, free_model, load_model_bundle


OUTPUT_ROOT = PROJECT_ROOT / "tests" / "codex_temp"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class CapabilityCase:
    capability: str
    pair_id: str
    prompt_a: str
    positive_a: str
    negative_a: str
    prompt_b: str
    positive_b: str
    negative_b: str


CASES: List[CapabilityCase] = [
    CapabilityCase(
        capability="disamb",
        pair_id="bank_meaning",
        prompt_a="Sentence: The river bank was muddy. One-word sense of bank:",
        positive_a=" shore",
        negative_a=" finance",
        prompt_b="Sentence: The bank approved the loan. One-word sense of bank:",
        positive_b=" finance",
        negative_b=" shore",
    ),
    CapabilityCase(
        capability="disamb",
        pair_id="plant_meaning",
        prompt_a="Sentence: The plant was green. One-word sense of plant:",
        positive_a=" leaf",
        negative_a=" factory",
        prompt_b="Sentence: The plant closed down. One-word sense of plant:",
        positive_b=" factory",
        negative_b=" leaf",
    ),
    CapabilityCase(
        capability="disamb",
        pair_id="bat_meaning",
        prompt_a="Sentence: The bat flew through the cave. One-word sense of bat:",
        positive_a=" animal",
        negative_a=" sports",
        prompt_b="Sentence: He swung the bat at the ball. One-word sense of bat:",
        positive_b=" sports",
        negative_b=" animal",
    ),
    CapabilityCase(
        capability="disamb",
        pair_id="watch_meaning",
        prompt_a="Sentence: She looked at her watch before dinner. One-word sense of watch:",
        positive_a=" clock",
        negative_a=" observe",
        prompt_b="Sentence: Watch the road carefully. One-word sense of watch:",
        positive_b=" observe",
        negative_b=" clock",
    ),
    CapabilityCase(
        capability="disamb",
        pair_id="light_meaning",
        prompt_a="Sentence: The light filled the room. One-word sense of light:",
        positive_a=" lamp",
        negative_a=" weight",
        prompt_b="Sentence: The bag felt light in my hand. One-word sense of light:",
        positive_b=" weight",
        negative_b=" lamp",
    ),
    CapabilityCase(
        capability="syntax",
        pair_id="subject_verb_number",
        prompt_a="The key to the cabinet",
        positive_a=" is",
        negative_a=" are",
        prompt_b="The keys to the cabinet",
        positive_b=" are",
        negative_b=" is",
    ),
    CapabilityCase(
        capability="syntax",
        pair_id="agreement_with_clause",
        prompt_a="The report that was written by the interns",
        positive_a=" was",
        negative_a=" were",
        prompt_b="The reports that were written by the intern",
        positive_b=" were",
        negative_b=" was",
    ),
    CapabilityCase(
        capability="syntax",
        pair_id="plural_with_pp",
        prompt_a="The label on the bottles",
        positive_a=" is",
        negative_a=" are",
        prompt_b="The labels on the bottle",
        positive_b=" are",
        negative_b=" is",
    ),
    CapabilityCase(
        capability="syntax",
        pair_id="distance_agreement",
        prompt_a="The bouquet of roses",
        positive_a=" smells",
        negative_a=" smell",
        prompt_b="The roses in the bouquet",
        positive_b=" smell",
        negative_b=" smells",
    ),
    CapabilityCase(
        capability="syntax",
        pair_id="collective_local_noun",
        prompt_a="The picture near the windows",
        positive_a=" was",
        negative_a=" were",
        prompt_b="The pictures near the window",
        positive_b=" were",
        negative_b=" was",
    ),
    CapabilityCase(
        capability="relation",
        pair_id="capital_relation",
        prompt_a="Paris is the capital of France. The capital of France is",
        positive_a=" Paris",
        negative_a=" Berlin",
        prompt_b="Berlin is the capital of Germany. The capital of Germany is",
        positive_b=" Berlin",
        negative_b=" Paris",
    ),
    CapabilityCase(
        capability="relation",
        pair_id="author_relation",
        prompt_a="Shakespeare wrote Hamlet. The author of Hamlet is",
        positive_a=" Shakespeare",
        negative_a=" Dickens",
        prompt_b="Tolstoy wrote War and Peace. The author of War and Peace is",
        positive_b=" Tolstoy",
        negative_b=" Shakespeare",
    ),
    CapabilityCase(
        capability="relation",
        pair_id="chemical_symbol_relation",
        prompt_a="The chemical symbol for water is",
        positive_a=" H2O",
        negative_a=" CO2",
        prompt_b="The chemical symbol for carbon dioxide is",
        positive_b=" CO2",
        negative_b=" H2O",
    ),
    CapabilityCase(
        capability="relation",
        pair_id="currency_relation",
        prompt_a="The currency used in Japan is",
        positive_a=" yen",
        negative_a=" euro",
        prompt_b="The currency used in Britain is",
        positive_b=" pound",
        negative_b=" yen",
    ),
    CapabilityCase(
        capability="relation",
        pair_id="inventor_relation",
        prompt_a="The telephone was invented by",
        positive_a=" Bell",
        negative_a=" Edison",
        prompt_b="The light bulb is associated with",
        positive_b=" Edison",
        negative_b=" Bell",
    ),
    CapabilityCase(
        capability="coref",
        pair_id="winner_reference",
        prompt_a="Alice thanked Mary because Alice had won the prize. The person who won was",
        positive_a=" Alice",
        negative_a=" Mary",
        prompt_b="Alice thanked Mary because Mary had won the prize. The person who won was",
        positive_b=" Mary",
        negative_b=" Alice",
    ),
    CapabilityCase(
        capability="coref",
        pair_id="apology_reference",
        prompt_a="John apologized to David because John was late. The person who was late was",
        positive_a=" John",
        negative_a=" David",
        prompt_b="John apologized to David because David was late. The person who was late was",
        positive_b=" David",
        negative_b=" John",
    ),
    CapabilityCase(
        capability="coref",
        pair_id="help_reference",
        prompt_a="Emma called Sara because Emma needed advice. The one needing advice was",
        positive_a=" Emma",
        negative_a=" Sara",
        prompt_b="Emma called Sara because Sara needed advice. The one needing advice was",
        positive_b=" Sara",
        negative_b=" Emma",
    ),
    CapabilityCase(
        capability="coref",
        pair_id="congratulation_reference",
        prompt_a="Liam congratulated Noah because Liam had succeeded. The one who succeeded was",
        positive_a=" Liam",
        negative_a=" Noah",
        prompt_b="Liam congratulated Noah because Noah had succeeded. The one who succeeded was",
        positive_b=" Noah",
        negative_b=" Liam",
    ),
    CapabilityCase(
        capability="coref",
        pair_id="blame_reference",
        prompt_a="Olivia blamed Mia because Olivia was careless. The careless person was",
        positive_a=" Olivia",
        negative_a=" Mia",
        prompt_b="Olivia blamed Mia because Mia was careless. The careless person was",
        positive_b=" Mia",
        negative_b=" Olivia",
    ),
    CapabilityCase(
        capability="style",
        pair_id="formal_rewrite",
        prompt_a="Choose the more formal rewrite: I need your help with this request.",
        positive_a=" assistance",
        negative_a=" help",
        prompt_b="Choose the more casual rewrite: I require your assistance with this request.",
        positive_b=" help",
        negative_b=" assistance",
    ),
    CapabilityCase(
        capability="style",
        pair_id="formal_word_choice",
        prompt_a="Choose the more formal next word: The ceremony will",
        positive_a=" commence",
        negative_a=" start",
        prompt_b="Choose the more casual next word: The game will",
        positive_b=" start",
        negative_b=" commence",
    ),
    CapabilityCase(
        capability="style",
        pair_id="formal_request",
        prompt_a="Choose the more formal request ending: Please review the attached file and",
        positive_a=" advise",
        negative_a=" tell",
        prompt_b="Choose the more casual request ending: Please look at the file and",
        positive_b=" tell",
        negative_b=" advise",
    ),
    CapabilityCase(
        capability="style",
        pair_id="formal_apology",
        prompt_a="Choose the more formal apology word: We",
        positive_a=" regret",
        negative_a=" sorry",
        prompt_b="Choose the more casual apology word: I am",
        positive_b=" sorry",
        negative_b=" regret",
    ),
    CapabilityCase(
        capability="style",
        pair_id="formal_departure",
        prompt_a="Choose the more formal departure verb: The guests will",
        positive_a=" depart",
        negative_a=" leave",
        prompt_b="Choose the more casual departure verb: My friends will",
        positive_b=" leave",
        negative_b=" depart",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage638 多能力统一编码协议")
    parser.add_argument("--model", default="qwen3", choices=["qwen3", "deepseek7b", "glm4", "gemma4"])
    parser.add_argument("--capability", default="all", choices=["all", "disamb", "syntax", "relation", "coref", "style"])
    parser.add_argument("--max-pairs", type=int, default=0, help="每类最多运行多少对；0 表示全部")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true", help="只导出协议样本，不加载模型")
    return parser.parse_args()


def mean_or_zero(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def safe_float(value: float) -> float:
    if value is None or math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def vector_norm(tensor: torch.Tensor) -> float:
    return float(torch.norm(tensor).item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if torch.norm(a).item() == 0.0 or torch.norm(b).item() == 0.0:
        return 0.0
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_batch_to_device(batch: Dict[str, torch.Tensor], model) -> Dict[str, torch.Tensor]:
    device = get_model_device(model)
    return {key: value.to(device) for key, value in batch.items()}


def score_candidate_avg_logprob(model, tokenizer, prompt: str, candidate: str) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + candidate, add_special_tokens=False)["input_ids"]
    if len(full_ids) <= len(prompt_ids):
        return float("-inf")
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=get_model_device(model))
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    count = 0
    for pos in range(len(prompt_ids), len(full_ids)):
        total += float(log_probs[pos - 1, full_ids[pos]].item())
        count += 1
    return total / max(count, 1)


def candidate_direction(model, tokenizer, positive: str, negative: str) -> torch.Tensor:
    embed = model.get_output_embeddings()
    if embed is None or not hasattr(embed, "weight"):
        raise RuntimeError("模型没有可用的输出嵌入矩阵")
    weight = embed.weight.detach().float()
    pos_ids = tokenizer(positive, add_special_tokens=False)["input_ids"]
    neg_ids = tokenizer(negative, add_special_tokens=False)["input_ids"]
    if not pos_ids or not neg_ids:
        raise RuntimeError("候选词无法分词")
    pos_vec = weight[pos_ids].mean(dim=0).cpu()
    neg_vec = weight[neg_ids].mean(dim=0).cpu()
    return pos_vec - neg_vec


def extract_prompt_hidden_by_layer(model, tokenizer, prompt: str, layers, max_length: int) -> Dict[int, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    encoded = move_batch_to_device(encoded, model)
    hidden_states: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_states[layer_idx] = hidden[0, -1, :].detach().float().cpu()
        return hook_fn

    for layer_idx, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.inference_mode():
            model(**encoded)
    finally:
        for hook in hooks:
            hook.remove()

    return hidden_states


def layer_difference_metrics(hidden_a: Dict[int, torch.Tensor], hidden_b: Dict[int, torch.Tensor]) -> Dict[str, object]:
    layer_ids = sorted(set(hidden_a.keys()) & set(hidden_b.keys()))
    if not layer_ids:
        return {
            "layer_count": 0,
            "d_norms": [],
            "d_norm_growth": 0.0,
            "mean_rotation_deg": 0.0,
            "layer_peak": None,
            "final_cos_to_initial": 0.0,
        }

    diffs = [hidden_a[idx] - hidden_b[idx] for idx in layer_ids]
    d_norms = [vector_norm(diff) for diff in diffs]
    initial_norm = max(d_norms[0], 1e-8)
    final_norm = d_norms[-1]
    rotations: List[float] = []
    for prev, curr in zip(diffs[:-1], diffs[1:]):
        cos_value = max(-1.0, min(1.0, cosine(prev, curr)))
        rotations.append(math.degrees(math.acos(cos_value)))

    peak_offset = max(range(len(d_norms)), key=lambda idx: d_norms[idx])
    mid_offset = len(layer_ids) // 2
    return {
        "layer_count": len(layer_ids),
        "d_norms": d_norms,
        "d_norm_growth": safe_float(final_norm / initial_norm),
        "mean_rotation_deg": safe_float(mean_or_zero(rotations)),
        "layer_peak": int(layer_ids[peak_offset]),
        "mid_layer": int(layer_ids[mid_offset]),
        "mid_d_norm": safe_float(d_norms[mid_offset]),
        "final_cos_to_initial": safe_float(cosine(diffs[-1], diffs[0])),
        "mid_direction": diffs[mid_offset],
        "final_direction": diffs[-1],
    }


def analyze_case(model, tokenizer, layers, case: CapabilityCase, max_length: int) -> Dict[str, object]:
    hidden_a = extract_prompt_hidden_by_layer(model, tokenizer, case.prompt_a, layers, max_length)
    hidden_b = extract_prompt_hidden_by_layer(model, tokenizer, case.prompt_b, layers, max_length)
    diff_metrics = layer_difference_metrics(hidden_a, hidden_b)

    margin_a = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - score_candidate_avg_logprob(
        model, tokenizer, case.prompt_a, case.negative_a
    )
    margin_b = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_b) - score_candidate_avg_logprob(
        model, tokenizer, case.prompt_b, case.negative_b
    )

    candidate_dir_a = candidate_direction(model, tokenizer, case.positive_a, case.negative_a)
    candidate_dir_b = candidate_direction(model, tokenizer, case.positive_b, case.negative_b)
    mid_direction = diff_metrics.pop("mid_direction", None)
    final_direction = diff_metrics.pop("final_direction", None)
    mid_alignment_a = cosine(mid_direction, candidate_dir_a) if mid_direction is not None else 0.0
    mid_alignment_b = cosine(mid_direction, candidate_dir_b) if mid_direction is not None else 0.0
    final_alignment_a = cosine(final_direction, candidate_dir_a) if final_direction is not None else 0.0
    final_alignment_b = cosine(final_direction, candidate_dir_b) if final_direction is not None else 0.0

    return {
        "capability": case.capability,
        "pair_id": case.pair_id,
        "prompt_a": case.prompt_a,
        "prompt_b": case.prompt_b,
        "positive_a": case.positive_a,
        "negative_a": case.negative_a,
        "positive_b": case.positive_b,
        "negative_b": case.negative_b,
        "margin_a": safe_float(margin_a),
        "margin_b": safe_float(margin_b),
        "avg_margin": safe_float((margin_a + margin_b) / 2.0),
        "preference_correct_a": bool(margin_a > 0.0),
        "preference_correct_b": bool(margin_b > 0.0),
        "preference_correct_pair": bool(margin_a > 0.0 and margin_b > 0.0),
        "mid_alignment_a": safe_float(mid_alignment_a),
        "mid_alignment_b": safe_float(mid_alignment_b),
        "signed_avg_mid_alignment": safe_float((mid_alignment_a + mid_alignment_b) / 2.0),
        "avg_abs_mid_alignment": safe_float((abs(mid_alignment_a) + abs(mid_alignment_b)) / 2.0),
        "final_alignment_a": safe_float(final_alignment_a),
        "final_alignment_b": safe_float(final_alignment_b),
        "signed_avg_final_alignment": safe_float((final_alignment_a + final_alignment_b) / 2.0),
        "avg_abs_final_alignment": safe_float((abs(final_alignment_a) + abs(final_alignment_b)) / 2.0),
        **diff_metrics,
    }


def aggregate_results(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    capability_summary: Dict[str, Dict[str, object]] = {}
    for capability in sorted({record["capability"] for record in records}):
        subset = [record for record in records if record["capability"] == capability]
        capability_summary[capability] = {
            "count": len(subset),
            "pair_accuracy": safe_float(sum(1 for item in subset if item["preference_correct_pair"]) / max(len(subset), 1)),
            "mean_margin": safe_float(mean_or_zero([item["avg_margin"] for item in subset])),
            "mean_d_norm_growth": safe_float(mean_or_zero([item["d_norm_growth"] for item in subset])),
            "mean_rotation_deg": safe_float(mean_or_zero([item["mean_rotation_deg"] for item in subset])),
            "mean_mid_d_norm": safe_float(mean_or_zero([item["mid_d_norm"] for item in subset])),
            "mean_abs_mid_alignment": safe_float(mean_or_zero([item["avg_abs_mid_alignment"] for item in subset])),
            "mean_abs_final_alignment": safe_float(mean_or_zero([item["avg_abs_final_alignment"] for item in subset])),
            "mean_signed_final_alignment": safe_float(mean_or_zero([item["signed_avg_final_alignment"] for item in subset])),
            "mean_peak_layer": safe_float(mean_or_zero([item["layer_peak"] for item in subset if item["layer_peak"] is not None])),
            "mean_mid_layer": safe_float(mean_or_zero([item["mid_layer"] for item in subset if item["mid_layer"] is not None])),
        }

    return {
        "overall_pair_accuracy": safe_float(sum(1 for item in records if item["preference_correct_pair"]) / max(len(records), 1)),
        "overall_mean_margin": safe_float(mean_or_zero([item["avg_margin"] for item in records])),
        "overall_mean_d_norm_growth": safe_float(mean_or_zero([item["d_norm_growth"] for item in records])),
        "overall_mean_rotation_deg": safe_float(mean_or_zero([item["mean_rotation_deg"] for item in records])),
        "overall_mean_mid_d_norm": safe_float(mean_or_zero([item["mid_d_norm"] for item in records])),
        "overall_mean_abs_mid_alignment": safe_float(mean_or_zero([item["avg_abs_mid_alignment"] for item in records])),
        "overall_mean_abs_final_alignment": safe_float(mean_or_zero([item["avg_abs_final_alignment"] for item in records])),
        "overall_mean_signed_final_alignment": safe_float(mean_or_zero([item["signed_avg_final_alignment"] for item in records])),
        "capability_summary": capability_summary,
    }


def build_report(summary: Dict[str, object], records: Sequence[Dict[str, object]], args: argparse.Namespace) -> str:
    lines = [
        "# Stage638 多能力统一编码协议报告",
        "",
        f"- 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型: {args.model}",
        f"- 能力范围: {args.capability}",
        f"- 样本数: {len(records)}",
        "",
        "## 总览",
        f"- overall_pair_accuracy: {summary['overall_pair_accuracy']:.4f}",
        f"- overall_mean_margin: {summary['overall_mean_margin']:.4f}",
        f"- overall_mean_d_norm_growth: {summary['overall_mean_d_norm_growth']:.4f}",
        f"- overall_mean_rotation_deg: {summary['overall_mean_rotation_deg']:.4f}",
        f"- overall_mean_mid_d_norm: {summary['overall_mean_mid_d_norm']:.4f}",
        f"- overall_mean_abs_mid_alignment: {summary['overall_mean_abs_mid_alignment']:.4f}",
        f"- overall_mean_abs_final_alignment: {summary['overall_mean_abs_final_alignment']:.4f}",
        f"- overall_mean_signed_final_alignment: {summary['overall_mean_signed_final_alignment']:.4f}",
        "",
        "## 分能力摘要",
    ]
    for capability, data in summary["capability_summary"].items():
        lines.extend(
            [
                f"### {capability}",
                f"- count: {data['count']}",
                f"- pair_accuracy: {data['pair_accuracy']:.4f}",
                f"- mean_margin: {data['mean_margin']:.4f}",
                f"- mean_d_norm_growth: {data['mean_d_norm_growth']:.4f}",
                f"- mean_rotation_deg: {data['mean_rotation_deg']:.4f}",
                f"- mean_mid_d_norm: {data['mean_mid_d_norm']:.4f}",
                f"- mean_abs_mid_alignment: {data['mean_abs_mid_alignment']:.4f}",
                f"- mean_abs_final_alignment: {data['mean_abs_final_alignment']:.4f}",
                f"- mean_signed_final_alignment: {data['mean_signed_final_alignment']:.4f}",
                f"- mean_mid_layer: {data['mean_mid_layer']:.4f}",
                f"- mean_peak_layer: {data['mean_peak_layer']:.4f}",
                "",
            ]
        )

    lines.append("## 单样本明细")
    for record in records:
        lines.extend(
            [
                f"### {record['capability']} / {record['pair_id']}",
                f"- avg_margin: {record['avg_margin']:.4f}",
                f"- preference_correct_pair: {record['preference_correct_pair']}",
                f"- d_norm_growth: {record['d_norm_growth']:.4f}",
                f"- mean_rotation_deg: {record['mean_rotation_deg']:.4f}",
                f"- mid_layer: {record['mid_layer']}",
                f"- mid_d_norm: {record['mid_d_norm']:.4f}",
                f"- avg_abs_mid_alignment: {record['avg_abs_mid_alignment']:.4f}",
                f"- avg_abs_final_alignment: {record['avg_abs_final_alignment']:.4f}",
                f"- signed_avg_final_alignment: {record['signed_avg_final_alignment']:.4f}",
                f"- layer_peak: {record['layer_peak']}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def selected_cases(capability: str, max_pairs: int) -> List[CapabilityCase]:
    filtered = [case for case in CASES if capability == "all" or case.capability == capability]
    if max_pairs <= 0:
        return filtered
    per_capability: Dict[str, int] = {}
    result: List[CapabilityCase] = []
    for case in filtered:
        current = per_capability.get(case.capability, 0)
        if current >= max_pairs:
            continue
        per_capability[case.capability] = current + 1
        result.append(case)
    return result


def main() -> None:
    args = parse_args()
    cases = selected_cases(args.capability, args.max_pairs)
    run_dir = OUTPUT_ROOT / f"stage638_multicapability_unified_protocol_{args.model}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.model,
            "capability": args.capability,
            "case_count": len(cases),
            "cases": [case.__dict__ for case in cases],
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "REPORT.md").write_text("# Stage638 Dry Run\n\n仅导出了协议样本，未加载模型。\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    model = None
    try:
        model, tokenizer = load_model_bundle(args.model, prefer_cuda=True)
        layers = discover_layers(model)
        records = [analyze_case(model, tokenizer, layers, case, args.max_length) for case in cases]
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.model,
            "capability": args.capability,
            "case_count": len(records),
            "aggregate": aggregate_results(records),
            "records": records,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        report = build_report(summary["aggregate"], records, args)
        (run_dir / "REPORT.md").write_text(report, encoding="utf-8")
        print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
        print(f"结果已写入: {run_dir}")
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
