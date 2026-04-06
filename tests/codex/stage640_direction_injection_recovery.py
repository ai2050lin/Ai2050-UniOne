#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage640: 差异方向注入恢复实验

目标：
1. 先在 syntax / relation / coref 三类任务上找到“最伤”的组件消融点。
2. 再在同一位置注入 clean 差异方向，测试能否恢复 margin。
3. 用四模型比较：
   - 差异方向是否具有恢复能力
   - 恢复更依赖 attn 破坏还是 mlp 破坏

用法：
python tests/codex/stage640_direction_injection_recovery.py --model qwen3
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from multimodel_language_shared import (
    ablate_layer_component,
    discover_layers,
    evenly_spaced_layers,
    free_model,
    load_model_bundle,
    restore_layer_component,
    score_candidate_avg_logprob,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "tests" / "codex_temp"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SCALES = [0.25, 0.5, 1.0]


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
        pair_id="distance_agreement",
        prompt_a="The bouquet of roses",
        positive_a=" smells",
        negative_a=" smell",
        prompt_b="The roses in the bouquet",
        positive_b=" smell",
        negative_b=" smells",
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
        pair_id="currency_relation",
        prompt_a="The currency used in Japan is",
        positive_a=" yen",
        negative_a=" euro",
        prompt_b="The currency used in Britain is",
        positive_b=" pound",
        negative_b=" yen",
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
        pair_id="help_reference",
        prompt_a="Emma called Sara because Emma needed advice. The one needing advice was",
        positive_a=" Emma",
        negative_a=" Sara",
        prompt_b="Emma called Sara because Sara needed advice. The one needing advice was",
        positive_b=" Sara",
        negative_b=" Emma",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage640 差异方向注入恢复实验")
    parser.add_argument("--model", required=True, choices=["qwen3", "deepseek7b", "gemma4", "glm4"])
    parser.add_argument("--layer-count", type=int, default=5)
    return parser.parse_args()


def mean_or_zero(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def case_margin(model, tokenizer, case: CapabilityCase) -> Dict[str, float | bool]:
    margin_a = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - score_candidate_avg_logprob(
        model, tokenizer, case.prompt_a, case.negative_a
    )
    margin_b = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_b) - score_candidate_avg_logprob(
        model, tokenizer, case.prompt_b, case.negative_b
    )
    avg_margin = float((margin_a + margin_b) / 2.0)
    return {
        "margin_a": float(margin_a),
        "margin_b": float(margin_b),
        "avg_margin": avg_margin,
        "pair_correct": bool(margin_a > 0.0 and margin_b > 0.0),
    }


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def encode_prompt(tokenizer, prompt: str, device: torch.device):
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    return {key: value.to(device) for key, value in encoded.items()}


def extract_layer_last_token(model, tokenizer, prompt: str, layer_idx: int) -> torch.Tensor:
    layers = discover_layers(model)
    device = get_model_device(model)
    encoded = encode_prompt(tokenizer, prompt, device)
    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["value"] = hidden[0, -1, :].detach().float().cpu()

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            model(**encoded)
    finally:
        handle.remove()
    return captured["value"]


def injected_case_margin(
    model,
    tokenizer,
    case: CapabilityCase,
    layer_idx: int,
    component: str,
    direction: torch.Tensor,
    scale: float,
) -> Dict[str, float | bool]:
    layers = discover_layers(model)
    layer = layers[layer_idx]
    device = get_model_device(model)
    direction = direction.to(device)

    def run_single(prompt: str, positive: str, negative: str, sign: float) -> float:
        comp_layer, original = ablate_layer_component(model, layer_idx, component)

        def inject_hook_with_pos(target_pos: int):
            def inject_hook(module, inputs, output):
                if isinstance(output, tuple):
                    hidden = output[0].clone()
                    hidden[0, target_pos, :] = hidden[0, target_pos, :] + sign * scale * direction
                    return (hidden,) + output[1:]
                hidden = output.clone()
                hidden[0, target_pos, :] = hidden[0, target_pos, :] + sign * scale * direction
                return hidden
            return inject_hook

        def score_full_sequence(full_ids: List[int], prompt_len: int) -> float:
            target_pos = max(prompt_len - 1, 0)
            inject_handle = layer.register_forward_hook(inject_hook_with_pos(target_pos))
            try:
                with torch.inference_mode():
                    logits = model(input_ids=torch.tensor([full_ids], dtype=torch.long, device=device)).logits[0].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                score = 0.0
                for pos in range(prompt_len, len(full_ids)):
                    score += float(log_probs[pos - 1, full_ids[pos]].item())
                return score / max(len(full_ids) - prompt_len, 1)
            finally:
                inject_handle.remove()

        try:
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_pos = tokenizer(prompt + positive, add_special_tokens=False)["input_ids"]
            full_neg = tokenizer(prompt + negative, add_special_tokens=False)["input_ids"]
            if len(full_pos) <= len(prompt_ids) or len(full_neg) <= len(prompt_ids):
                return float("-inf")
            pos_score = score_full_sequence(full_pos, len(prompt_ids))
            neg_score = score_full_sequence(full_neg, len(prompt_ids))
            return pos_score - neg_score
        finally:
            restore_layer_component(comp_layer, component, original)

    margin_a = run_single(case.prompt_a, case.positive_a, case.negative_a, +1.0)
    margin_b = run_single(case.prompt_b, case.positive_b, case.negative_b, -1.0)
    avg_margin = float((margin_a + margin_b) / 2.0)
    return {
        "margin_a": margin_a,
        "margin_b": margin_b,
        "avg_margin": avg_margin,
        "pair_correct": bool(margin_a > 0.0 and margin_b > 0.0),
    }


def build_report(args: argparse.Namespace, records: Sequence[Dict[str, object]], summary: Dict[str, object]) -> str:
    lines = [
        "# Stage640 差异方向注入恢复报告",
        "",
        f"- 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型: {args.model}",
        f"- 样本数: {len(records)}",
        "",
        "## 分能力摘要",
    ]
    for capability, item in summary.items():
        lines.extend(
            [
                f"### {capability}",
                f"- baseline_margin: {item['baseline_margin']:.4f}",
                f"- ablated_margin: {item['ablated_margin']:.4f}",
                f"- recovered_margin: {item['recovered_margin']:.4f}",
                f"- mean_recovery_gain: {item['mean_recovery_gain']:.4f}",
                f"- mean_recovery_ratio: {item['mean_recovery_ratio']:.4f}",
                "",
            ]
        )
    lines.append("## 单样本摘要")
    for record in records:
        lines.extend(
            [
                f"### {record['capability']} / {record['pair_id']}",
                f"- best_damage_component: {record['best_damage_component']}",
                f"- best_damage_layer: {record['best_damage_layer']}",
                f"- baseline_margin: {record['baseline_margin']:.4f}",
                f"- ablated_margin: {record['ablated_margin']:.4f}",
                f"- best_recovered_margin: {record['best_recovered_margin']:.4f}",
                f"- recovery_gain: {record['recovery_gain']:.4f}",
                f"- recovery_ratio: {record['recovery_ratio']:.4f}",
                f"- best_scale: {record['best_scale']}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dir = OUTPUT_ROOT / f"stage640_direction_injection_recovery_{args.model}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        model, tokenizer = load_model_bundle(args.model, prefer_cuda=True)
        layer_indices = evenly_spaced_layers(model, count=args.layer_count)
        records: List[Dict[str, object]] = []

        for case in CASES:
            baseline = case_margin(model, tokenizer, case)
            damage_candidates: List[Dict[str, object]] = []

            for layer_idx in layer_indices:
                for component in ("attn", "mlp"):
                    layer, original = ablate_layer_component(model, layer_idx, component)
                    try:
                        ablated = case_margin(model, tokenizer, case)
                    finally:
                        restore_layer_component(layer, component, original)
                    damage_candidates.append(
                        {
                            "layer": layer_idx,
                            "component": component,
                            "avg_margin": ablated["avg_margin"],
                            "pair_correct": ablated["pair_correct"],
                            "margin_drop": float(baseline["avg_margin"] - ablated["avg_margin"]),
                        }
                    )

            best_damage = max(damage_candidates, key=lambda item: item["margin_drop"])
            direction = extract_layer_last_token(model, tokenizer, case.prompt_a, best_damage["layer"]) - extract_layer_last_token(
                model, tokenizer, case.prompt_b, best_damage["layer"]
            )

            recovery_trials: List[Dict[str, object]] = []
            for scale in SCALES:
                recovered = injected_case_margin(
                    model, tokenizer, case, best_damage["layer"], best_damage["component"], direction, scale
                )
                recovery_trials.append(
                    {
                        "scale": scale,
                        "avg_margin": recovered["avg_margin"],
                        "pair_correct": recovered["pair_correct"],
                        "recovery_gain": float(recovered["avg_margin"] - best_damage["avg_margin"]),
                    }
                )

            best_recovery = max(recovery_trials, key=lambda item: item["avg_margin"])
            denom = max(baseline["avg_margin"] - best_damage["avg_margin"], 1e-8)
            recovery_ratio = float((best_recovery["avg_margin"] - best_damage["avg_margin"]) / denom)

            records.append(
                {
                    "capability": case.capability,
                    "pair_id": case.pair_id,
                    "layer_indices": layer_indices,
                    "baseline_margin": baseline["avg_margin"],
                    "baseline_pair_correct": baseline["pair_correct"],
                    "best_damage_component": best_damage["component"],
                    "best_damage_layer": best_damage["layer"],
                    "ablated_margin": best_damage["avg_margin"],
                    "ablated_pair_correct": best_damage["pair_correct"],
                    "best_recovered_margin": best_recovery["avg_margin"],
                    "best_recovered_pair_correct": best_recovery["pair_correct"],
                    "best_scale": best_recovery["scale"],
                    "recovery_gain": float(best_recovery["avg_margin"] - best_damage["avg_margin"]),
                    "recovery_ratio": recovery_ratio,
                    "damage_candidates": damage_candidates,
                    "recovery_trials": recovery_trials,
                }
            )

        summary: Dict[str, Dict[str, float]] = {}
        for capability in sorted({item["capability"] for item in records}):
            subset = [item for item in records if item["capability"] == capability]
            summary[capability] = {
                "baseline_margin": mean_or_zero([item["baseline_margin"] for item in subset]),
                "ablated_margin": mean_or_zero([item["ablated_margin"] for item in subset]),
                "recovered_margin": mean_or_zero([item["best_recovered_margin"] for item in subset]),
                "mean_recovery_gain": mean_or_zero([item["recovery_gain"] for item in subset]),
                "mean_recovery_ratio": mean_or_zero([item["recovery_ratio"] for item in subset]),
            }

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.model,
            "layer_indices": layer_indices,
            "records": records,
            "summary": summary,
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        report = build_report(args, records, summary)
        (run_dir / "REPORT.md").write_text(report, encoding="utf-8")
        print(json.dumps({"model": args.model, "layer_indices": layer_indices, "summary": summary}, ensure_ascii=False, indent=2))
        print(f"结果已写入: {run_dir}")
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
