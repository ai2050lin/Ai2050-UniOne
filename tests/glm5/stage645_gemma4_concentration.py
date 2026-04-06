#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage645: Gemma4集中编码机制深挖——对比Gemma4与其他模型的编码集中度

核心问题：为什么Gemma4在Stage640(275%)和Stage641(79%)中的recovery远超其他模型？

已知差异：
- Gemma4高速旋转(66.1°/层) vs Qwen3(22°)/DS7B(~30°)/GLM4(~20°)
- Gemma4 8Q+1KV头 vs 其他模型标准MHA
- Gemma4消歧差(15%) vs Qwen3(70%)

假说：
H1：高速旋转导致信息被压缩到少数高活性维度→方向更集中
H2：GQ+KV架构导致不同能力共享更多编码维度→方向耦合更强(Stage642: cos=0.72)
H3：Gemma4的层分布更极端(L0+L34双峰)→特定层承担了更多功能

实验设计：
1. 编码集中度分析：每个能力的delta_l的奇异值能量集中度(top-1/top-5/top-10)
2. 逐层信息瓶颈：哪些层的信息最集中(用effective rank衡量)
3. 旋转-集中度相关性：逐层旋转角度与编码集中度的相关性
4. 跨模型编码效率对比：四模型的"编码集中度 vs 消歧效率"散点图

预注册判伪：
  如果Gemma4的编码集中度(top-1能量占比)不高于其他模型>10%，
  则"高速旋转导致编码集中"(INV-201)被推翻。

用法：
  python stage645_gemma4_concentration.py [qwen3|deepseek7b|glm4|gemma4]
"""

from __future__ import annotations

import sys
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    discover_layers,
    evenly_spaced_layers,
    free_model,
    load_model_bundle,
    score_candidate_avg_logprob,
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


@dataclass(frozen=True)
class CapabilityPrompt:
    capability: str
    prompt_a: str
    prompt_b: str


PROMPTS = [
    CapabilityPrompt("syntax", "The key to the cabinet", "The keys to the cabinet"),
    CapabilityPrompt("relation", "Paris is the capital of France", "Berlin is the capital of Germany"),
    CapabilityPrompt("coref", "Alice won because Alice was best", "Mary won because Mary was best"),
    CapabilityPrompt("syllogism", "All mammals are animals. All dogs are", "All birds are animals. All sparrows are"),
    CapabilityPrompt("arithmetic", "What is 7 plus 8?", "What is 9 plus 6?"),
    CapabilityPrompt("implication", "If it rains, the ground gets wet.", "If it snows, the roads get icy."),
]


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def extract_all_layers(model, tokenizer, prompt: str) -> Dict[int, torch.Tensor]:
    layers = discover_layers(model)
    device = get_model_device(model)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    captured: Dict[int, torch.Tensor] = {}

    def make_hook(lidx):
        def hook_fn(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[lidx] = hidden[0, -1, :].detach().float().cpu()
        return hook_fn

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(len(layers))]
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    return captured


def compute_concentration_metrics(delta: torch.Tensor) -> Dict[str, float]:
    """计算一个差异向量的集中度指标"""
    # SVD能量分布
    delta_2d = delta.unsqueeze(0)
    try:
        U, S, Vt = torch.linalg.svd(delta_2d, full_matrices=False)
    except Exception:
        return {"top1_pct": 100.0, "top5_pct": 100.0, "effective_rank": 1.0, "norm": delta.norm().item()}

    total = S.sum().item()
    if total < 1e-10:
        return {"top1_pct": 100.0, "top5_pct": 100.0, "effective_rank": 1.0, "norm": 0.0}

    top1 = S[0].item() / total * 100
    top5 = min(5, len(S))
    top5_pct = sum(S[:top5].tolist()) / total * 100
    # effective rank: exp(entropy)
    probs = S / total
    entropy = -sum(float(p) * float(torch.log(p + 1e-30)) for p in probs if p > 1e-10)
    eff_rank = float(torch.exp(torch.tensor(entropy)).item())

    return {
        "top1_pct": top1,
        "top5_pct": top5_pct,
        "effective_rank": eff_rank,
        "norm": delta.norm().item(),
        "total_energy": total,
    }


def layer_rotation_angle(h_prev: torch.Tensor, h_curr: torch.Tensor) -> float:
    """计算相邻层hidden state的旋转角度"""
    cos_sim = torch.nn.functional.cosine_similarity(h_prev.unsqueeze(0), h_curr.unsqueeze(0)).item()
    cos_sim = max(-1.0, min(1.0, cos_sim))
    import math
    return math.degrees(math.acos(cos_sim))


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python stage645_gemma4_concentration.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)
    model_key = sys.argv[1]
    run_dir = OUTPUT_DIR / f"stage645_gemma4_concentration_{model_key}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        print(f"[Stage645] 加载模型 {model_key}...")
        model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
        n_layers = len(discover_layers(model))
        print(f"[Stage645] 层数: {n_layers}")

        results: List[Dict] = []

        for i, cp in enumerate(PROMPTS):
            print(f"\n[Stage645] ({i+1}/{len(PROMPTS)}) {cp.capability}")

            h_a = extract_all_layers(model, tokenizer, cp.prompt_a)
            h_b = extract_all_layers(model, tokenizer, cp.prompt_b)

            layer_metrics = []
            for lidx in range(n_layers):
                delta = h_a[lidx] - h_b[lidx]
                metrics = compute_concentration_metrics(delta)
                metrics["layer"] = lidx
                layer_metrics.append(metrics)

            # 旋转角度
            rotation_angles = []
            for lidx in range(1, n_layers):
                angle = layer_rotation_angle(h_a[lidx-1], h_a[lidx])
                rotation_angles.append(angle)

            # 找最集中层
            best_conc_layer = min(layer_metrics, key=lambda x: x["effective_rank"])

            # 旋转-集中度相关性
            conc_values = [m["top1_pct"] for m in layer_metrics[1:]]  # L1..L_last
            if len(rotation_angles) == len(conc_values) and len(conc_values) > 2:
                mean_r = sum(rotation_angles) / len(rotation_angles)
                mean_c = sum(conc_values) / len(conc_values)
                num = sum((r - mean_r) * (c - mean_c) for r, c in zip(rotation_angles, conc_values))
                den_r = sum((r - mean_r)**2 for r in rotation_angles) ** 0.5
                den_c = sum((c - mean_c)**2 for c in conc_values) ** 0.5
                rot_conc_corr = num / (den_r * den_c + 1e-10)
            else:
                rot_conc_corr = 0.0

            avg_top1 = statistics.mean([m["top1_pct"] for m in layer_metrics])
            avg_eff_rank = statistics.mean([m["effective_rank"] for m in layer_metrics])

            print(f"  avg_top1={avg_top1:.1f}%, avg_eff_rank={avg_eff_rank:.2f}")
            print(f"  best_conc_layer={best_conc_layer['layer']}(eff_rank={best_conc_layer['effective_rank']:.2f})")
            print(f"  rot_conc_corr={rot_conc_corr:.4f}")

            results.append({
                "capability": cp.capability,
                "layer_metrics": layer_metrics,
                "rotation_angles": rotation_angles,
                "avg_top1_pct": avg_top1,
                "avg_eff_rank": avg_eff_rank,
                "best_conc_layer": best_conc_layer["layer"],
                "best_conc_eff_rank": best_conc_layer["effective_rank"],
                "rot_conc_corr": rot_conc_corr,
                "mean_rotation_angle": statistics.mean(rotation_angles) if rotation_angles else 0,
            })

        # 汇总
        avg_top1_all = statistics.mean([r["avg_top1_pct"] for r in results])
        avg_eff_rank_all = statistics.mean([r["avg_eff_rank"] for r in results])
        avg_rot = statistics.mean([r["mean_rotation_angle"] for r in results])
        avg_rot_conc = statistics.mean([r["rot_conc_corr"] for r in results])

        print(f"\n[Stage645] 汇总:")
        print(f"  avg_top1_pct={avg_top1_all:.1f}%")
        print(f"  avg_eff_rank={avg_eff_rank_all:.2f}")
        print(f"  avg_rotation_angle={avg_rot:.1f}°")
        print(f"  avg_rot_conc_corr={avg_rot_conc:.4f}")
        print(f"  （需要四个模型结果对比后才能判断INV-201）")

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_key,
            "avg_top1_pct": avg_top1_all,
            "avg_eff_rank": avg_eff_rank_all,
            "avg_rotation_angle": avg_rot,
            "avg_rot_conc_corr": avg_rot_conc,
            "records": results,
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入: {run_dir / 'summary.json'}")
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
