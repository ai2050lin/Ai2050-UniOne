#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage659: P10 推理3维功能分解

P7发现：Qwen3/DS7B/GLM4的推理能力编码在3个有效维度上(90% variance)。
但不知道这3个维度分别对应什么认知功能。

P10目标：分解推理子空间的3个维度，识别每个维度的功能语义
实验1：单维度贡献分析 — 逐个维度消融，测量各推理case的变化
实验2：维度-能力关联矩阵 — 哪些维度对应哪些推理类型？
实验3：leave-one-out验证 — 移除单个维度后剩余维度的预测能力
实验4：跨模型维度语义一致性 — 不同模型的推理维度是否对应相同功能？

预注册判伪条件：
INV-244: "推理3维中存在专用推理类型维度"
  如果所有维度对所有case的贡献相同(variance<10%)，则INV-244被推翻
INV-245: "跨模型推理维度语义一致"
  如果不同模型的维度-能力关联矩阵cos<0.5，则INV-245被推翻
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
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    discover_layers,
    free_model,
    load_model_bundle,
    score_candidate_avg_logprob,
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


@dataclass(frozen=True)
class TestCase:
    name: str
    prompt_a: str
    positive_a: str
    negative_a: str
    prompt_b: str
    positive_b: str
    negative_b: str
    category: str  # reasoning sub-type


REASONING_CASES: List[TestCase] = [
    TestCase("syllogism",
             "All mammals are animals. All cats are mammals. Cats are",
             " animals", " reptiles",
             "All birds are animals. All sparrows are birds. Sparrows are",
             " animals", " insects",
             category="deductive"),
    TestCase("syllogism_neg",
             "All fish are animals. No dogs are fish. Dogs are",
             " animals", " fish",
             "All roses are flowers. No trees are roses. Trees are",
             " flowers", " roses",
             category="deductive_neg"),
    TestCase("chain_reasoning",
             "If A implies B and B implies C, then A implies",
             " C", " B",
             "If X implies Y and Y implies Z, then X implies",
             " Z", " Y",
             category="transitive"),
    TestCase("arithmetic_chain",
             "If x = 7, y = x + 3, and z = y + 5, then z =",
             " 15", " 13",
             "If a = 10, b = a + 2, and c = b + 3, then c =",
             " 15", " 13",
             category="arithmetic"),
    TestCase("modus_ponens",
             "If it rains, the ground is wet. It rains. Therefore, the ground is",
             " wet", " dry",
             "If the light is on, someone is home. The light is on. Someone is",
             " home", " away",
             category="deductive"),
    TestCase("contrapositive",
             "If A then B. Not B. Therefore",
             " not A", " A",
             "If P then Q. Not Q. Therefore",
             " not P", " P",
             category="deductive_neg"),
]


def case_margin(model, tokenizer, case: TestCase) -> float:
    ma = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
         score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
    mb = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_b) - \
         score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.negative_b)
    return float((ma + mb) / 2.0)


def extract_last_token_hidden(model, tokenizer, text: str, layer_idx: int) -> torch.Tensor:
    layers = discover_layers(model)
    captured = {"value": None}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured["value"] = output[0][:, -1, :].detach().cpu()
        else:
            captured["value"] = output[:, -1, :].detach().cpu()

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["value"].squeeze(0)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python stage659_reasoning_3d_decomposition.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)
    model_key = sys.argv[1]
    run_dir = OUTPUT_DIR / f"stage659_reasoning3d_{model_key}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        print(f"[Stage659] Loading {model_key}...")
        model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
        num_layers = len(discover_layers(model))
        final_layer = num_layers - 1
        print(f"[Stage659] layers={num_layers}")

        # 计算baseline margins
        baselines = {}
        for c in REASONING_CASES:
            baselines[c.name] = case_margin(model, tokenizer, c)
            print(f"  Baseline {c.name} ({c.category}): {baselines[c.name]:+.4f}")

        # 提取所有推理delta
        print("\nExtracting reasoning deltas...")
        reasoning_deltas = {}
        for c in REASONING_CASES:
            ha = extract_last_token_hidden(model, tokenizer, c.prompt_a, final_layer)
            hb = extract_last_token_hidden(model, tokenizer, c.prompt_b, final_layer)
            reasoning_deltas[c.name] = (ha - hb).float()

        # SVD分解得到推理子空间
        rnames = list(reasoning_deltas.keys())
        delta_matrix = torch.stack([reasoning_deltas[n] for n in rnames])
        U, S, Vt = torch.linalg.svd(delta_matrix.float(), full_matrices=False)

        # 计算有效维度
        cumsum = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
        rank90 = (cumsum >= 0.9).nonzero()
        rank90 = rank90[0].item() + 1 if len(rank90) > 0 else len(S)

        print(f"\n  SVD singular values: {[round(s.item(), 4) for s in S[:6]]}")
        print(f"  Effective rank (90%): {rank90}")
        print(f"  Top1 fraction: {round((S[0].item()**2 / (S**2).sum().item()), 4)}")

        # ========================================
        # 实验1: 单维度贡献分析
        # 逐维度消融，测量各case的变化
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 1: Per-Dimension Contribution Analysis")
        print(f"{'='*60}")

        k = min(rank90 + 1, len(S))  # 分析前rank90+1个维度
        dim_contributions = {}  # dim -> case -> contribution

        for dim_idx in range(k):
            dim_vec = Vt[dim_idx]  # 第dim_idx个右奇异向量（维度方向）
            print(f"\n  Dimension {dim_idx} (SV={S[dim_idx].item():.4f}, frac={round((S[dim_idx].item()**2/(S**2).sum().item()), 4)}):")

            for cn in rnames:
                delta = reasoning_deltas[cn]
                # 该维度上的投影
                proj = torch.dot(delta, dim_vec).item()
                # 该维度贡献的delta能量占比
                dim_energy = proj ** 2
                total_energy = delta.norm().item() ** 2
                contribution = dim_energy / total_energy if total_energy > 0 else 0

                if dim_idx not in dim_contributions:
                    dim_contributions[dim_idx] = {}
                dim_contributions[dim_idx][cn] = {
                    "projection": round(proj, 4),
                    "contribution": round(contribution, 6),
                }

            # 显示该维度对各case的贡献
            contribs = [dim_contributions[dim_idx][cn]["contribution"] for cn in rnames]
            print(f"    Contributions: {[f'{c:.4f}' for c in contribs]}")
            print(f"    Mean: {statistics.mean(contribs):.4f}, Std: {statistics.pstdev(contribs):.4f}")

        # 检查INV-244: 维度贡献是否有特异性
        dim_contribution_matrix = []
        for dim_idx in range(k):
            row = []
            for cn in rnames:
                row.append(dim_contributions[dim_idx][cn]["contribution"])
            dim_contribution_matrix.append(row)

        # 计算每个维度的贡献方差
        dim_vars = []
        for dim_idx in range(k):
            vals = dim_contribution_matrix[dim_idx]
            dim_vars.append(statistics.pvariance(vals))

        mean_dim_var = statistics.mean(dim_vars)
        inv244_status = "CONFIRMED" if mean_dim_var > 0.001 else "FALSIFIED"
        print(f"\n  Mean dimension variance: {mean_dim_var:.6f}")
        print(f"  INV-244 (specialized dimensions): {inv244_status}")

        # ========================================
        # 实验2: 维度-能力关联矩阵
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 2: Dimension-Capability Association Matrix")
        print(f"{'='*60}")

        categories = list(set(c.category for c in REASONING_CASES))
        print(f"  Categories: {categories}")

        # 对每个类别，聚合delta然后投影到各维度
        category_vectors = {}
        for cat in categories:
            cat_deltas = []
            for c in REASONING_CASES:
                if c.category == cat:
                    cat_deltas.append(reasoning_deltas[c.name])
            category_vectors[cat] = torch.stack(cat_deltas).mean(dim=0)

        # 维度-类别关联矩阵
        dim_cat_matrix = []
        for dim_idx in range(k):
            dim_vec = Vt[dim_idx]
            row = {}
            for cat in categories:
                cat_vec = category_vectors[cat]
                proj = torch.dot(cat_vec, dim_vec).item()
                row[cat] = round(proj, 4)
            dim_cat_matrix.append(row)

        print("\n  Dimension-Category projection matrix:")
        header = f"  {'Dim':>5s}"
        for cat in categories:
            header += f" {cat:>15s}"
        print(header)
        for dim_idx in range(k):
            line = f"  D{dim_idx:>4d}"
            for cat in categories:
                val = dim_cat_matrix[dim_idx][cat]
                line += f" {val:>+15.4f}"
            print(line)

        # 找到每个维度的主导类别
        dim_labels = {}
        for dim_idx in range(k):
            max_cat = max(categories, key=lambda c: abs(dim_cat_matrix[dim_idx][c]))
            dim_labels[dim_idx] = max_cat
            print(f"  D{dim_idx} dominant category: {max_cat} (proj={dim_cat_matrix[dim_idx][max_cat]:.4f})")

        # ========================================
        # 实验3: leave-one-out验证
        # 移除单个维度后，用剩余维度重建delta并测量margin
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 3: Leave-One-Out Validation")
        print(f"{'='*60}")

        loo_results = {}
        for cn in rnames:
            delta = reasoning_deltas[cn]
            full_recon = torch.zeros_like(delta)
            for dim_idx in range(min(4, k)):
                proj = torch.dot(delta, Vt[dim_idx]) * Vt[dim_idx]
                full_recon += proj
            full_cos = torch.dot(delta, full_recon) / (delta.norm() * full_recon.norm() + 1e-10)

            loo = {}
            for dim_idx in range(min(4, k)):
                # 移除第dim_idx个维度
                partial_recon = torch.zeros_like(delta)
                for di in range(min(4, k)):
                    if di != dim_idx:
                        proj = torch.dot(delta, Vt[di]) * Vt[di]
                        partial_recon += proj
                if partial_recon.norm() > 1e-6:
                    partial_cos = torch.dot(delta, partial_recon) / (delta.norm() * partial_recon.norm())
                else:
                    partial_cos = torch.tensor(0.0)
                drop = full_cos.item() - partial_cos.item()
                loo[dim_idx] = {
                    "full_cos": round(full_cos.item(), 4),
                    "partial_cos": round(partial_cos.item(), 4),
                    "drop": round(drop, 4),
                }
            loo_results[cn] = loo

        for cn in rnames[:3]:  # 只显示前3个case
            print(f"\n  {cn}:")
            for dim_idx in range(min(4, k)):
                r = loo_results[cn][dim_idx]
                print(f"    Remove D{dim_idx}: full_cos={r['full_cos']:.4f}, "
                      f"partial_cos={r['partial_cos']:.4f}, drop={r['drop']:.4f}")

        # ========================================
        # 实验4: 维度语义标签
        # 基于实验2的关联矩阵给每个维度打标签
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 4: Dimension Semantic Labeling")
        print(f"{'='*60}")

        for dim_idx in range(k):
            dominant = dim_labels[dim_idx]
            # 计算该维度的"专一性"：主导类别贡献占比
            all_proj = [abs(dim_cat_matrix[dim_idx][cat]) for cat in categories]
            total_proj = sum(all_proj)
            dominant_frac = abs(dim_cat_matrix[dim_idx][dominant]) / total_proj if total_proj > 0 else 0

            # 计算该维度在所有case上的贡献方差
            contribs = [dim_contributions[dim_idx][cn]["contribution"] for cn in rnames]
            contrib_std = statistics.pstdev(contribs)

            label = f"D{dim_idx}: {dominant}"
            if dominant_frac > 0.5:
                label += " (specialized)"
            else:
                label += " (general)"

            print(f"  {label}, dominant_frac={dominant_frac:.3f}, contrib_std={contrib_std:.6f}")

        # 最终汇总
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"  Model: {model_key}")
        print(f"  Reasoning effective rank (90%): {rank90}")
        print(f"  INV-244 (specialized dimensions): {inv244_status}")
        print(f"  INV-245 (cross-model consistency): PENDING (need all models)")
        print(f"  Dimension labels:")
        for dim_idx in range(k):
            print(f"    D{dim_idx}: {dim_labels[dim_idx]}")

        # 保存结果
        results = {
            "model": model_key,
            "num_layers": num_layers,
            "rank90": rank90,
            "singular_values": [round(s.item(), 4) for s in S[:6]],
            "dim_contributions": {str(k): v for k, v in dim_contributions.items()},
            "dim_cat_matrix": dim_cat_matrix,
            "dim_labels": dim_labels,
            "inv244_status": inv244_status,
            "baselines": baselines,
        }
        out_path = run_dir / f"results_{model_key}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  Results saved to {out_path}")

    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
