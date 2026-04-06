#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage646: 方向传播保真度追踪

目标：逐层追踪推理方向在残差流中的保真度，量化旋转速度对方向保留的影响
方法：
1. 在推理任务的最伤层提取delta_l（A-B差异方向）
2. 逐层提取后续层的delta，计算 cos(delta_l, delta_{l+k})
3. 计算平均逐层旋转角度
4. 对比四模型的保真度衰减曲线

预注册判伪条件：
INV-211: "推理方向在Qwen3/DS7B/GLM4中快速衰减(10层内cos<0.3)"
如果3/4模型的推理方向在10层内cos>0.3，则INV-211被推翻。

INV-212: "Gemma4的方向保真度高于其他模型"
如果Gemma4的平均方向保真度不高于其他三个模型，则INV-212被推翻。
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
import numpy as np

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


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@dataclass(frozen=True)
class ReasoningCase:
    name: str
    prompt_a: str
    positive_a: str
    negative_a: str
    prompt_b: str
    positive_b: str
    negative_b: str


REASONING_CASES: List[ReasoningCase] = [
    ReasoningCase(
        name="syllogism",
        prompt_a="All mammals are animals. All cats are mammals. Therefore, all cats are",
        positive_a=" animals",
        negative_a=" reptiles",
        prompt_b="All birds are animals. All sparrows are birds. Therefore, all sparrows are",
        positive_b=" animals",
        negative_b=" insects",
    ),
    ReasoningCase(
        name="arithmetic",
        prompt_a="If x = 7 and y = 3, then x + y =",
        positive_a=" 10",
        negative_a=" 11",
        prompt_b="If x = 15 and y = 8, then x + y =",
        positive_b=" 23",
        negative_b=" 22",
    ),
    ReasoningCase(
        name="implication",
        prompt_a="If it rains then the ground is wet. It rains. Therefore, the ground is",
        positive_a=" wet",
        negative_a=" dry",
        prompt_b="If it snows then the road is slippery. It snows. Therefore, the road is",
        positive_b=" slippery",
        negative_b=" dry",
    ),
    ReasoningCase(
        name="negation",
        prompt_a="Tom is NOT tall. Which is true?",
        positive_a=" Tom is short",
        negative_a=" Tom is tall",
        prompt_b="The answer is NOT B. The answer is",
        positive_b=" A",
        negative_b=" B",
    ),
]


def case_margin(model, tokenizer, case: ReasoningCase) -> Dict[str, float]:
    margin_a = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - score_candidate_avg_logprob(
        model, tokenizer, case.prompt_a, case.negative_a
    )
    margin_b = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_b) - score_candidate_avg_logprob(
        model, tokenizer, case.prompt_b, case.negative_b
    )
    return {
        "margin_a": float(margin_a),
        "margin_b": float(margin_b),
        "avg_margin": float((margin_a + margin_b) / 2.0),
    }


def extract_layer_last_token(model, tokenizer, prompt: str, layer_idx: int) -> torch.Tensor:
    """在指定层的最后一个token提取hidden state"""
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


def find_most_harmful_layer(model, tokenizer, case: ReasoningCase, layer_indices: List[int]) -> Dict:
    """找到对margin影响最大的层和组件"""
    baseline = case_margin(model, tokenizer, case)
    best_drop = 0
    best_layer = 0
    best_component = "attn"

    for layer_idx in layer_indices:
        for component in ("attn", "mlp"):
            layer, original = ablate_layer_component(model, layer_idx, component)
            try:
                ablated = case_margin(model, tokenizer, case)
            finally:
                restore_layer_component(layer, component, original)
            drop = baseline["avg_margin"] - ablated["avg_margin"]
            if drop > best_drop:
                best_drop = drop
                best_layer = layer_idx
                best_component = component

    return {
        "layer": best_layer,
        "component": best_component,
        "baseline_margin": baseline["avg_margin"],
        "damage": best_drop,
    }


def trace_direction_cos(model, tokenizer, case: ReasoningCase, anchor_layer: int,
                        num_layers: int, max_trace: int = 15) -> Dict:
    """
    追踪方向在后续层中的余弦相似度衰减
    anchor_layer: 起始层
    """
    delta_anchor = extract_layer_last_token(model, tokenizer, case.prompt_a, anchor_layer) - \
                   extract_layer_last_token(model, tokenizer, case.prompt_b, anchor_layer)
    delta_anchor_norm = delta_anchor / (torch.norm(delta_anchor) + 1e-10)

    cos_values = {}
    end_layer = min(anchor_layer + max_trace, num_layers)

    for target_layer in range(anchor_layer + 1, end_layer):
        delta_target = extract_layer_last_token(model, tokenizer, case.prompt_a, target_layer) - \
                       extract_layer_last_token(model, tokenizer, case.prompt_b, target_layer)
        delta_target_norm = delta_target / (torch.norm(delta_target) + 1e-10)
        cos_val = torch.dot(delta_anchor_norm, delta_target_norm).item()
        cos_values[target_layer] = cos_val

    return cos_values


def compute_avg_layer_rotation(model, tokenizer, num_layers: int, num_samples: int = 8) -> float:
    """计算平均逐层旋转角度"""
    templates = [
        "The cat sat on the mat.",
        "The dog ran in the park.",
        "A bird flew over the house.",
        "She read a book yesterday.",
        "They went to the store.",
        "He plays piano every day.",
        "The sun rises in the east.",
        "We had dinner at seven.",
    ]
    rotations = []
    for i in range(min(num_samples, len(templates))):
        prompt = templates[i]
        for layer_pair in [(0, 1), (5, 6), (10, 11), (15, 16)]:
            l1, l2 = layer_pair
            if l2 >= num_layers:
                continue
            try:
                h1 = extract_layer_last_token(model, tokenizer, prompt, l1)
                h2 = extract_layer_last_token(model, tokenizer, prompt, l2)
                delta = h2 - h1
                if torch.norm(h1) > 1e-8 and torch.norm(delta) > 1e-8:
                    h1n = h1 / torch.norm(h1)
                    dn = delta / torch.norm(delta)
                    cos_val = torch.dot(h1n, dn).item()
                    cos_val = max(-1.0, min(1.0, cos_val))
                    angle = np.degrees(np.arccos(cos_val))
                    rotations.append(abs(angle))
            except Exception:
                continue

    return float(np.mean(rotations)) if rotations else 0.0


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python stage646_direction_propagation_fidelity.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)
    model_key = sys.argv[1]
    run_dir = OUTPUT_DIR / f"stage646_fidelity_{model_key}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        print(f"[Stage646] 加载模型 {model_key}...")
        model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
        num_layers = len(discover_layers(model))
        layer_indices = evenly_spaced_layers(model, count=7)
        print(f"[Stage646] 层数={num_layers}, 采样层={layer_indices}")

        # 平均旋转速度
        print(f"[Stage646] 计算平均旋转速度...")
        avg_rotation = compute_avg_layer_rotation(model, tokenizer, num_layers)
        print(f"[Stage646] 平均旋转速度: {avg_rotation:.1f}°/层")

        records = []
        for i, case in enumerate(REASONING_CASES):
            print(f"\n[Stage646] ({i+1}/{len(REASONING_CASES)}) {case.name}")
            harm_info = find_most_harmful_layer(model, tokenizer, case, layer_indices)
            print(f"  最伤层: L{harm_info['layer']}/{harm_info['component']}, 损伤={harm_info['damage']:.4f}")

            if harm_info["damage"] < 0.01:
                print("  损伤过小，跳过追踪")
                continue

            cos_values = trace_direction_cos(model, tokenizer, case, harm_info["layer"], num_layers, max_trace=15)

            # 提取关键指标
            offsets = sorted(cos_values.keys())
            cos_3layer = None
            cos_5layer = None
            cos_10layer = None
            for k in offsets:
                offset = k - harm_info["layer"]
                if offset == 3:
                    cos_3layer = cos_values[k]
                elif offset == 5:
                    cos_5layer = cos_values[k]
                elif offset == 10:
                    cos_10layer = cos_values[k]

            # 找到cos首次<0.3的层
            decay_layer = None
            for k in offsets:
                if abs(cos_values[k]) < 0.3:
                    decay_layer = k - harm_info["layer"]
                    break

            print(f"  3层cos={cos_3layer:.4f}" if cos_3layer is not None else "  3层: N/A")
            print(f"  5层cos={cos_5layer:.4f}" if cos_5layer is not None else "  5层: N/A")
            print(f"  10层cos={cos_10layer:.4f}" if cos_10layer is not None else "  10层: N/A")
            print(f"  cos<0.3衰减层: {decay_layer if decay_layer else '>15层'}")

            records.append({
                "case_name": case.name,
                "harm_layer": harm_info["layer"],
                "harm_component": harm_info["component"],
                "baseline_margin": harm_info["baseline_margin"],
                "damage": harm_info["damage"],
                "cos_values": {str(k): round(v, 4) for k, v in cos_values.items()},
                "cos_3layer": round(cos_3layer, 4) if cos_3layer is not None else None,
                "cos_5layer": round(cos_5layer, 4) if cos_5layer is not None else None,
                "cos_10layer": round(cos_10layer, 4) if cos_10layer is not None else None,
                "decay_to_03_layer": decay_layer,
            })

        # 汇总
        valid_cos5 = [r["cos_5layer"] for r in records if r["cos_5layer"] is not None and not np.isnan(r["cos_5layer"])]
        valid_cos10 = [r["cos_10layer"] for r in records if r["cos_10layer"] is not None and not np.isnan(r["cos_10layer"])]
        avg_cos5 = float(np.mean(valid_cos5)) if valid_cos5 else None
        avg_cos10 = float(np.mean(valid_cos10)) if valid_cos10 else None

        valid_decay = [r["decay_to_03_layer"] for r in records if r["decay_to_03_layer"] is not None]
        avg_decay = float(np.mean(valid_decay)) if valid_decay else None

        print(f"\n{'='*60}")
        print(f"[Stage646] 汇总: {model_key}")
        print(f"  avg rotation: {avg_rotation:.1f} deg/layer")
        print(f"  平均5层cos: {avg_cos5:.4f}" if avg_cos5 is not None else "  平均5层cos: N/A")
        print(f"  平均10层cos: {avg_cos10:.4f}" if avg_cos10 is not None else "  平均10层cos: N/A")
        print(f"  平均cos<0.3衰减层: {avg_decay:.1f}" if avg_decay is not None else "  平均cos<0.3衰减层: >15")

        # 判伪
        # INV-211: 推理方向在10层内cos<0.3
        if avg_cos10 is not None:
            inv211 = "SURVIVED" if avg_cos10 < 0.3 else "FALSIFIED"
        else:
            inv211 = "INCONCLUSIVE"
        print(f"\n  INV-211 (快速衰减): {inv211}")

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_key,
            "num_layers": num_layers,
            "avg_rotation_speed": round(avg_rotation, 1),
            "avg_cos_5layer": round(avg_cos5, 4) if avg_cos5 is not None else None,
            "avg_cos_10layer": round(avg_cos10, 4) if avg_cos10 is not None else None,
            "avg_decay_to_03_layer": round(avg_decay, 1) if avg_decay is not None else None,
            "inv211_result": inv211,
            "records": records,
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入: {run_dir / 'summary.json'}")
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
