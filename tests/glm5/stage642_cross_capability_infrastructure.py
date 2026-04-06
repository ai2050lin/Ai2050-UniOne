#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage642: 跨能力基础设施验证——统一编码架构的三重检验

目标：
1. 交叉干扰测试：语法delta_l注入后，推理/共指/relation能力是否被改变？
   - 如果注入语法方向只改变语法而不影响其他能力→支持"独立通道假说"
   - 如果注入语法方向同时改变推理→支持"统一基底假说"
   
2. 方向正交性验证：语法/relation/coref/推理的差异方向之间cos值
   - 如果跨能力方向互相正交(cos<0.1)→支持"独立编码"
   - 如果跨能力方向有显著重叠(cos>0.3)→支持"共享基底+路由"

3. 层带一致性验证：不同能力的"最伤层"是否一致
   - Stage640找语法/relation/coref的最伤层
   - Stage641找推理的最伤层
   - 如果所有能力最伤层相同→支持"通用处理层"假说
   - 如果不同能力最伤层不同→支持"功能特化层"假说

预注册判伪条件：
  如果所有跨能力cos>0.3（方向高度重叠），则"多能力独立编码"(INV-188)被推翻。

用法：
  python stage642_cross_capability_infrastructure.py [qwen3|deepseek7b|glm4|gemma4]
"""

from __future__ import annotations

import sys
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


@dataclass(frozen=True)
class TestPair:
    capability: str
    pair_id: str
    prompt_a: str
    positive_a: str
    negative_a: str
    prompt_b: str
    positive_b: str
    negative_b: str


ALL_CASES: List[TestPair] = [
    # === 语法 ===
    TestPair(
        capability="syntax", pair_id="subj_verb",
        prompt_a="The key to the cabinet", positive_a=" is", negative_a=" are",
        prompt_b="The keys to the cabinet", positive_b=" are", negative_b=" is",
    ),
    # === 关系 ===
    TestPair(
        capability="relation", pair_id="capital",
        prompt_a="Paris is the capital of France. The capital of France is", positive_a=" Paris", negative_a=" Berlin",
        prompt_b="Berlin is the capital of Germany. The capital of Germany is", positive_b=" Berlin", negative_b=" Paris",
    ),
    # === 共指 ===
    TestPair(
        capability="coref", pair_id="winner",
        prompt_a="Alice thanked Mary because Alice had won. The winner was", positive_a=" Alice", negative_a=" Mary",
        prompt_b="Alice thanked Mary because Mary had won. The winner was", positive_b=" Mary", negative_b=" Alice",
    ),
    # === 三段论推理 ===
    TestPair(
        capability="syllogism", pair_id="barbara",
        prompt_a="All mammals are animals. All dogs are mammals. Therefore all dogs are", positive_a=" animals", negative_a=" birds",
        prompt_b="All birds are animals. All sparrows are birds. Therefore all sparrows are", positive_b=" animals", negative_b=" mammals",
    ),
    # === 算术推理 ===
    TestPair(
        capability="arithmetic", pair_id="add7p8",
        prompt_a="What is 7 plus 8? The answer is", positive_a=" 15", negative_a=" 16",
        prompt_b="What is 9 plus 6? The answer is", positive_b=" 15", negative_b=" 14",
    ),
    # === 蕴含推理 ===
    TestPair(
        capability="implication", pair_id="modus_ponens",
        prompt_a="If it rains, the ground gets wet. It rains. Therefore the ground", positive_a=" gets wet", negative_a=" stays dry",
        prompt_b="If it snows, the roads get icy. It snows. Therefore the roads", positive_b=" get icy", negative_b=" stay clear",
    ),
]


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def case_margin(model, tokenizer, case: TestPair) -> float:
    margin_a = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - score_candidate_avg_logprob(
        model, tokenizer, case.prompt_a, case.negative_a
    )
    margin_b = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_b) - score_candidate_avg_logprob(
        model, tokenizer, case.prompt_b, case.negative_b
    )
    return float((margin_a + margin_b) / 2.0)


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


def mean_or_zero(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python stage642_cross_capability_infrastructure.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)
    model_key = sys.argv[1]
    run_dir = OUTPUT_DIR / f"stage642_cross_capability_{model_key}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        print(f"[Stage642] 加载模型 {model_key}...")
        model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
        layer_indices = evenly_spaced_layers(model, count=5)
        print(f"[Stage642] 层索引: {layer_indices}")

        # ===== Part 1: 每个能力的最伤层和差异方向 =====
        print("\n[Part 1] 寻找每个能力的最伤层和差异方向...")
        capability_info: Dict[str, Dict] = {}
        directions: Dict[str, torch.Tensor] = {}

        for case in ALL_CASES:
            cap = case.capability
            if cap in capability_info:
                continue
            print(f"  {cap}/{case.pair_id}...")
            baseline = case_margin(model, tokenizer, case)

            best_drop = 0.0
            best_layer = 0
            best_comp = "attn"
            for layer_idx in layer_indices:
                for component in ("attn", "mlp"):
                    layer, original = ablate_layer_component(model, layer_idx, component)
                    try:
                        ablated = case_margin(model, tokenizer, case)
                    finally:
                        restore_layer_component(layer, component, original)
                    drop = baseline - ablated
                    if drop > best_drop:
                        best_drop = drop
                        best_layer = layer_idx
                        best_comp = component

            direction = extract_layer_last_token(model, tokenizer, case.prompt_a, best_layer) - extract_layer_last_token(
                model, tokenizer, case.prompt_b, best_layer
            )
            # normalize
            direction = direction / (direction.norm() + 1e-10)

            capability_info[cap] = {
                "case_id": case.pair_id,
                "baseline_margin": baseline,
                "best_damage_drop": best_drop,
                "best_damage_layer": best_layer,
                "best_damage_component": best_comp,
            }
            directions[cap] = direction
            print(f"    best: layer={best_layer}, comp={best_comp}, drop={best_drop:.4f}")

        # ===== Part 2: 跨能力方向余弦矩阵 =====
        print("\n[Part 2] 跨能力方向余弦矩阵...")
        caps = sorted(directions.keys())
        cos_matrix: Dict[str, Dict[str, float]] = {}
        all_cos_values = []
        for c1 in caps:
            cos_matrix[c1] = {}
            for c2 in caps:
                cos_val = float(torch.nn.functional.cosine_similarity(
                    directions[c1].unsqueeze(0), directions[c2].unsqueeze(0)
                ).item())
                cos_matrix[c1][c2] = cos_val
                if c1 < c2:
                    all_cos_values.append(abs(cos_val))

        for c1 in caps:
            row = "  ".join(f"{cos_matrix[c1][c2]:+.3f}" for c2 in caps)
            print(f"  {c1:12s}: {row}")

        mean_abs_cos = mean_or_zero(all_cos_values) if all_cos_values else 0.0
        max_abs_cos = max(all_cos_values) if all_cos_values else 0.0
        print(f"  mean|cos|={mean_abs_cos:.4f}, max|cos|={max_abs_cos:.4f}")

        # ===== Part 3: 交叉干扰测试 =====
        print("\n[Part 3] 交叉干扰测试...")
        cross_interference: Dict[str, Dict[str, float]] = {}

        for src_cap in caps:
            cross_interference[src_cap] = {}
            src_dir = directions[src_cap]
            src_info = capability_info[src_cap]
            inject_layer = src_info["best_damage_layer"]
            inject_comp = src_info["best_damage_component"]
            layer_module = discover_layers(model)[inject_layer]
            device = get_model_device(model)

            for tgt_case in ALL_CASES:
                tgt_cap = tgt_case.capability
                if tgt_cap in cross_interference[src_cap]:
                    continue

                # 基线
                baseline_tgt = case_margin(model, tokenizer, tgt_case)

                # 注入源方向
                def make_hook(direction_tensor, target_pos):
                    def inject_hook(module, inputs, output):
                        if isinstance(output, tuple):
                            hidden = output[0].clone()
                            hidden[0, target_pos, :] = hidden[0, target_pos, :] + direction_tensor
                            return (hidden,) + output[1:]
                        hidden = output.clone()
                        hidden[0, target_pos, :] = hidden[0, target_pos, :] + direction_tensor
                        return hidden
                    return inject_hook

                prompt_ids = tokenizer(tgt_case.prompt_a, add_special_tokens=False)["input_ids"]
                target_pos = max(len(prompt_ids) - 1, 0)
                hook = make_hook(src_dir.to(device), target_pos)

                inject_handle = layer_module.register_forward_hook(hook)
                try:
                    # 简单测试：注入后prompt_a的正负margin
                    pos_lp = score_candidate_avg_logprob(model, tokenizer, tgt_case.prompt_a, tgt_case.positive_a)
                    neg_lp = score_candidate_avg_logprob(model, tokenizer, tgt_case.prompt_a, tgt_case.negative_a)
                    injected_margin = pos_lp - neg_lp
                finally:
                    inject_handle.remove()

                change_pct = abs(injected_margin - baseline_tgt) / (abs(baseline_tgt) + 1e-8) * 100
                cross_interference[src_cap][tgt_cap] = {
                    "baseline_margin": baseline_tgt,
                    "injected_margin": injected_margin,
                    "change_pct": change_pct,
                }

            row = "  ".join(
                f"{cross_interference[src_cap][c]['change_pct']:6.1f}%" for c in caps
            )
            print(f"  {src_cap:12s} →: {row}")

        # ===== Part 4: 层带一致性分析 =====
        print("\n[Part 4] 层带一致性...")
        layer_map = {cap: info["best_damage_layer"] for cap, info in capability_info.items()}
        comp_map = {cap: info["best_damage_component"] for cap, info in capability_info.items()}
        layers_list = list(layer_map.values())
        unique_layers = set(layers_list)
        layer_consistency = 1.0 if len(unique_layers) == 1 else (max(layers_list) - min(layers_list)) / max(len(layer_indices) - 1, 1)
        print(f"  层分布: {layer_map}")
        print(f"  组件分布: {comp_map}")
        print(f"  层一致性(0=完全一致,1=最大分散): {layer_consistency:.4f}")

        # ===== 判伪 =====
        all_cross_cos = []
        for c1 in caps:
            for c2 in caps:
                if c1 != c2:
                    all_cross_cos.append(abs(cos_matrix[c1][c2]))
        mean_cross_cos = mean_or_zero(all_cross_cos)
        falsified = mean_cross_cos > 0.3
        print(f"\n[Stage642] 跨能力mean|cos|={mean_cross_cos:.4f}")
        print(f"[Stage642] 预注册判伪: {'INV-188被推翻（方向高度重叠→共享基底）' if falsified else 'INV-188存活（方向近似正交→独立编码）'}")

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_key,
            "layer_indices": layer_indices,
            "capability_info": capability_info,
            "cos_matrix": cos_matrix,
            "mean_abs_cos": mean_abs_cos,
            "max_abs_cos": max_abs_cos,
            "cross_interference": cross_interference,
            "layer_consistency": layer_consistency,
            "mean_cross_cos": mean_cross_cos,
            "falsified": falsified,
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入: {run_dir / 'summary.json'}")
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
