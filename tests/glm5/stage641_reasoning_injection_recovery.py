#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage641: 推理能力差异方向注入恢复实验

目标：
1. 在 syllogism / arithmetic / implication 三类推理任务上找到"最伤"的组件消融点
2. 在同一位置注入 clean 差异方向，测试能否恢复推理 margin
3. 四模型比较：推理delta_l是否具有因果恢复力
4. 对比Stage640语法/relation/coref的恢复模式——推理恢复是否更弱（更依赖分布式编码）

预注册判伪条件：
  如果推理任务的recovery_ratio均值<0.2（恢复不到20%），则
  "推理能力由可提取的差异方向编码"(INV-187)被推翻——说明推理能力
  不是集中编码而是分布式编码，无法通过单一方向恢复。

用法：
  python stage641_reasoning_injection_recovery.py [qwen3|deepseek7b|glm4|gemma4]
"""

from __future__ import annotations

import sys
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    ablate_layer_component,
    discover_layers,
    evenly_spaced_layers,
    free_model,
    load_model_bundle,
    restore_layer_component,
    score_candidate_avg_logprob,
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
SCALES = [0.25, 0.5, 1.0]


@dataclass(frozen=True)
class ReasoningCase:
    capability: str
    pair_id: str
    prompt_a: str
    positive_a: str
    negative_a: str
    prompt_b: str
    positive_b: str
    negative_b: str


CASES: List[ReasoningCase] = [
    ReasoningCase(
        capability="syllogism",
        pair_id="barbara",
        prompt_a="All mammals are animals. All dogs are mammals. Therefore, all dogs are",
        positive_a=" animals",
        negative_a=" birds",
        prompt_b="All birds are animals. All sparrows are birds. Therefore, all sparrows are",
        positive_b=" animals",
        negative_b=" mammals",
    ),
    ReasoningCase(
        capability="syllogism",
        pair_id="celarent",
        prompt_a="No reptiles are mammals. All snakes are reptiles. Therefore, no snakes are",
        positive_a=" mammals",
        negative_a=" reptiles",
        prompt_b="No insects are mammals. All ants are insects. Therefore, no ants are",
        positive_b=" mammals",
        negative_b=" insects",
    ),
    ReasoningCase(
        capability="arithmetic",
        pair_id="addition",
        prompt_a="What is 7 plus 8? The answer is",
        positive_a=" 15",
        negative_a=" 16",
        prompt_b="What is 9 plus 6? The answer is",
        positive_b=" 15",
        negative_b=" 14",
    ),
    ReasoningCase(
        capability="arithmetic",
        pair_id="multiplication",
        prompt_a="What is 6 times 7? The answer is",
        positive_a=" 42",
        negative_a=" 43",
        prompt_b="What is 8 times 5? The answer is",
        positive_b=" 40",
        negative_b=" 45",
    ),
    ReasoningCase(
        capability="implication",
        pair_id="modus_ponens",
        prompt_a="If it rains, the ground gets wet. It rains. Therefore, the ground",
        positive_a=" gets wet",
        negative_a=" stays dry",
        prompt_b="If it snows, the roads get icy. It snows. Therefore, the roads",
        positive_b=" get icy",
        negative_b=" stay clear",
    ),
    ReasoningCase(
        capability="implication",
        pair_id="modus_tollens",
        prompt_a="If the alarm rings, there is a fire. The alarm did not ring. Therefore, there is",
        positive_a=" no fire",
        negative_a=" a fire",
        prompt_b="If the light is on, someone is home. The light is off. Therefore, no one is",
        positive_b=" home",
        negative_b=" away",
    ),
]


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def case_margin(model, tokenizer, case: ReasoningCase) -> Dict[str, float | bool]:
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


def extract_layer_last_token(model, tokenizer, prompt: str, layer_idx: int) -> torch.Tensor:
    layers = discover_layers(model)
    device = get_model_device(model)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["value"] = hidden[0, -1, :].detach().float().cpu()

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["value"]


def injected_case_margin(
    model,
    tokenizer,
    case: ReasoningCase,
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

        def score_full_sequence(full_ids: list, prompt_len: int) -> float:
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


def mean_or_zero(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python stage641_reasoning_injection_recovery.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)
    model_key = sys.argv[1]
    run_dir = OUTPUT_DIR / f"stage641_reasoning_injection_recovery_{model_key}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        print(f"[Stage641] 加载模型 {model_key}...")
        model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
        layer_indices = evenly_spaced_layers(model, count=5)
        print(f"[Stage641] 层索引: {layer_indices}")
        records: List[Dict[str, object]] = []

        for i, case in enumerate(CASES):
            print(f"[Stage641] ({i+1}/{len(CASES)}) {case.capability}/{case.pair_id}")
            baseline = case_margin(model, tokenizer, case)
            print(f"  baseline_margin={baseline['avg_margin']:.4f}")

            damage_candidates: List[Dict[str, object]] = []
            for layer_idx in layer_indices:
                for component in ("attn", "mlp"):
                    layer, original = ablate_layer_component(model, layer_idx, component)
                    try:
                        ablated = case_margin(model, tokenizer, case)
                    finally:
                        restore_layer_component(layer, component, original)
                    damage_candidates.append({
                        "layer": layer_idx,
                        "component": component,
                        "avg_margin": ablated["avg_margin"],
                        "pair_correct": ablated["pair_correct"],
                        "margin_drop": float(baseline["avg_margin"] - ablated["avg_margin"]),
                    })

            best_damage = max(damage_candidates, key=lambda item: item["margin_drop"])
            print(f"  best_damage: layer={best_damage['layer']}, comp={best_damage['component']}, drop={best_damage['margin_drop']:.4f}")

            direction = extract_layer_last_token(model, tokenizer, case.prompt_a, best_damage["layer"]) - extract_layer_last_token(
                model, tokenizer, case.prompt_b, best_damage["layer"]
            )

            recovery_trials: List[Dict[str, object]] = []
            for scale in SCALES:
                recovered = injected_case_margin(
                    model, tokenizer, case, best_damage["layer"], best_damage["component"], direction, scale
                )
                recovery_trials.append({
                    "scale": scale,
                    "avg_margin": recovered["avg_margin"],
                    "pair_correct": recovered["pair_correct"],
                    "recovery_gain": float(recovered["avg_margin"] - best_damage["avg_margin"]),
                })

            best_recovery = max(recovery_trials, key=lambda item: item["avg_margin"])
            denom = max(baseline["avg_margin"] - best_damage["avg_margin"], 1e-8)
            recovery_ratio = float((best_recovery["avg_margin"] - best_damage["avg_margin"]) / denom)
            print(f"  recovery_ratio={recovery_ratio:.4f}, best_scale={best_recovery['scale']}")

            records.append({
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
                "damage_candidates": [
                    {"layer": d["layer"], "component": d["component"], "margin_drop": d["margin_drop"]}
                    for d in damage_candidates
                ],
                "recovery_trials": recovery_trials,
            })

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

        overall_recovery = mean_or_zero([item["recovery_ratio"] for item in records])
        falsified = overall_recovery < 0.2
        print(f"\n[Stage641] 整体recovery_ratio={overall_recovery:.4f}")
        print(f"[Stage641] 预注册判伪: {'INV-187被推翻' if falsified else 'INV-187存活'}")

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_key,
            "layer_indices": layer_indices,
            "overall_recovery_ratio": overall_recovery,
            "falsified": falsified,
            "records": records,
            "summary": summary,
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入: {run_dir / 'summary.json'}")
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
