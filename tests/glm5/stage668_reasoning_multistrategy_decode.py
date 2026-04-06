#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage668: P21 推理编码的多策略解码

问题诊断：推理能力的恢复率<4%（INV-220），是所有能力中唯一无法用低维捕捉的

四种解码策略：
- 策略1：门控模式解码 — 不看delta大小，看符号模式(正/负/零分布)
- 策略2：子任务分解 — 将推理分解为子步骤，逐步提取编码方向
- 策略3：高维探针 — 用1层MLP做分类，测试非线性可分性
- 策略4：层级信息整合 — 用所有层的hidden state做联合分类

判伪标准：
INV-297: "任一策略恢复率>30% → 推理低维编码假说确认"
INV-298: "所有策略恢复率<10% → 推理是高维分布式编码"
"""

from __future__ import annotations

import sys
import io
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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
class ReasonCase:
    name: str
    prompt_a: str  # 推理前提A（导致结论A）
    prompt_b: str  # 推理前提B（导致结论B）
    positive_a: str  # A的正确答案
    negative_a: str  # A的错误答案（B的正确答案）
    # 推理样本：A和B有不同推理结论
    reasoning_type: str


CASES: List[ReasonCase] = [
    # 空间推理
    ReasonCase(name="spatial_left",
               prompt_a="If Tom is to the left of Jerry, and Jerry is to the left of Alice, who is in the middle?",
               prompt_b="If Tom is to the left of Jerry, and Alice is to the left of Jerry, who is in the middle?",
               positive_a=" Jerry", negative_a=" Alice",
               reasoning_type="spatial"),
    ReasonCase(name="spatial_order",
               prompt_a="The red ball is behind the blue ball, and the blue ball is behind the green ball. The green ball is in front of",
               prompt_b="The red ball is behind the blue ball, and the green ball is behind the blue ball. The green ball is in front of",
               positive_a=" the red ball", negative_a=" the blue ball",
               reasoning_type="spatial"),
    ReasonCase(name="spatial_direction",
               prompt_a="North of the library is the park, and north of the park is the school. Going north from the library you reach",
               prompt_b="North of the library is the park, and south of the park is the school. Going north from the library you reach",
               positive_a=" the park", negative_a=" the school",
               reasoning_type="spatial"),
    # 逻辑推理
    ReasonCase(name="logic_syllogism",
               prompt_a="All roses are flowers. All flowers need water. Therefore, roses need",
               prompt_b="All roses are flowers. No flowers need water. Therefore, roses need",
               positive_a=" water", negative_a=" no water",
               reasoning_type="logic"),
    ReasonCase(name="logic_negation",
               prompt_a="If it rains, the ground is wet. It rained. The ground is",
               prompt_b="If it rains, the ground is wet. It did not rain. The ground is",
               positive_a=" wet", negative_a=" dry",
               reasoning_type="logic"),
    ReasonCase(name="logic_conditional",
               prompt_a="If x > 5 then x > 3. x = 7. Therefore x > 3 is",
               prompt_b="If x > 5 then x > 3. x = 2. Therefore x > 3 is",
               positive_a=" true", negative_a=" false",
               reasoning_type="logic"),
    # 数学推理
    ReasonCase(name="math_arithmetic",
               prompt_a="If you have 3 apples and get 4 more, you have",
               prompt_b="If you have 3 apples and give away 4, you have",
               positive_a=" 7", negative_a=" -1",
               reasoning_type="math"),
    ReasonCase(name="math_comparison",
               prompt_a="15 is greater than 12 but less than 20. The number between 15 and 20 could be",
               prompt_b="15 is greater than 12 but less than 20. The number between 12 and 15 could be",
               positive_a=" 18", negative_a=" 14",
               reasoning_type="math"),
    ReasonCase(name="math_sequence",
               prompt_a="The sequence is 2, 4, 8, 16. The next number is",
               prompt_b="The sequence is 2, 4, 6, 8. The next number is",
               positive_a=" 32", negative_a=" 10",
               reasoning_type="math"),
    # 因果推理
    ReasonCase(name="causal_cause",
               prompt_a="The streets are wet because it rained. The cause of wet streets is",
               prompt_b="The streets are wet because the fire hydrant burst. The cause of wet streets is",
               positive_a=" rain", negative_a=" the fire hydrant",
               reasoning_type="causal"),
    ReasonCase(name="causal_effect",
               prompt_a="Turning off the lights makes the room dark. The effect of turning off lights is",
               prompt_b="Opening the curtains makes the room bright. The effect of opening curtains is",
               positive_a=" darkness", negative_a=" brightness",
               reasoning_type="causal"),
    # 类比推理
    ReasonCase(name="analogy_pattern",
               prompt_a="Dog is to puppy as cat is to",
               prompt_b="Dog is to puppy as cat is to",
               positive_a=" kitten", negative_a=" cub",
               reasoning_type="analogy"),
    ReasonCase(name="analogy_function",
               prompt_a="Pen is to write as knife is to",
               prompt_b="Pen is to write as thermometer is to",
               positive_a=" cut", negative_a=" measure",
               reasoning_type="analogy"),
]


def extract_all_layer_hidden(model, tokenizer, prompt: str) -> Dict[int, torch.Tensor]:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    hidden_states = {}
    hooks = []
    layers = discover_layers(model)

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states[layer_idx] = output[0][:, -1, :].detach().cpu().float()
            else:
                hidden_states[layer_idx] = output[:, -1, :].detach().cpu().float()
        return hook_fn

    for li in range(len(layers)):
        hooks.append(layers[li].register_forward_hook(make_hook(li)))

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in hooks:
            h.remove()
    return hidden_states


# ========== 策略1：门控模式解码 ==========
def strategy1_gate_pattern(hidden_a: Dict[int, torch.Tensor],
                            hidden_b: Dict[int, torch.Tensor],
                            hidden_dim: int) -> Dict:
    """
    不看delta大小，看符号模式：
    - 对每层的delta，计算正/负/零的维度分布
    - 用符号模式(而非幅度)做分类
    """
    num_layers = len(hidden_a)
    gate_features = {}
    
    for li in range(num_layers):
        h_a = hidden_a.get(li)
        h_b = hidden_b.get(li)
        if h_a is None or h_b is None:
            continue
        
        delta = (h_a - h_b).flatten()
        # 符号模式：正值比例、负值比例、零值比例
        pos_ratio = float((delta > 0.01).sum()) / hidden_dim
        neg_ratio = float((delta < -0.01).sum()) / hidden_dim
        zero_ratio = 1.0 - pos_ratio - neg_ratio
        
        # 符号的统计特征
        mean_val = float(delta.mean())
        std_val = float(delta.std())
        # 手动计算偏度
        if std_val > 1e-6:
            centered = delta - delta.mean()
            skew_val = float((centered ** 3).mean() / (std_val ** 3 + 1e-10))
        else:
            skew_val = 0
        
        gate_features[li] = {
            "pos_ratio": pos_ratio,
            "neg_ratio": neg_ratio,
            "zero_ratio": zero_ratio,
            "mean": mean_val,
            "std": std_val,
            "skew": skew_val,
            "top10_pos_dims": torch.topk(delta, 10).indices.tolist(),
            "top10_neg_dims": torch.topk(-delta, 10).indices.tolist(),
        }
    
    return gate_features


# ========== 策略2：子任务分解 ==========
def strategy2_subtask_decomposition(model, tokenizer, case: ReasonCase,
                                     num_layers: int) -> Dict:
    """
    将推理分解为子步骤，逐步提取编码方向。
    思路：推理任务可以分解为"理解前提"→"建立关系"→"推导结论"三个阶段。
    对每个子步骤，单独提取差异向量，然后组合。
    """
    # 子步骤1：前提理解
    premise_a = case.prompt_a.split(".")[0] + "." if "." in case.prompt_a else case.prompt_a
    premise_b = case.prompt_b.split(".")[0] + "." if "." in case.prompt_b else case.prompt_b
    
    # 子步骤2：中间步骤（如果有）
    sentences_a = [s.strip() for s in case.prompt_a.split(".") if s.strip()]
    sentences_b = [s.strip() for s in case.prompt_b.split(".") if s.strip()]
    
    # 对每个逐步增长的prefix提取hidden state
    steps_a = []
    steps_b = []
    
    # 累积构建prefix
    prefix_a = ""
    prefix_b = ""
    for i, (sa, sb) in enumerate(zip(sentences_a, sentences_b)):
        prefix_a = sa if i == 0 else prefix_a + " " + sa
        prefix_b = sb if i == 0 else prefix_b + " " + sb
        steps_a.append(prefix_a)
        steps_b.append(prefix_b)
    
    step_diffs = {}
    for i, (pa, pb) in enumerate(zip(steps_a, steps_b)):
        if i >= len(steps_a) or i >= len(steps_b):
            break
        ha = extract_all_layer_hidden(model, tokenizer, pa)
        hb = extract_all_layer_hidden(model, tokenizer, pb)
        
        # 每个子步骤的差异向量（最后一层）
        if num_layers - 1 in ha and num_layers - 1 in hb:
            diff = ha[num_layers - 1].flatten() - hb[num_layers - 1].flatten()
            step_diffs[i] = diff
    
    # 组合策略：逐步加权和
    if len(step_diffs) >= 2:
        # 各子步骤的cos相似度
        step_keys = sorted(step_diffs.keys())
        cos_between_steps = []
        for j in range(len(step_keys) - 1):
            k1, k2 = step_keys[j], step_keys[j + 1]
            d1 = step_diffs[k1]
            d2 = step_diffs[k2]
            cos_val = float(torch.dot(d1, d2) / (d1.norm() * d2.norm() + 1e-10))
            cos_between_steps.append(cos_val)
        
        # 加权组合：用各步骤的范数作为权重
        weights = {k: step_diffs[k].norm() for k in step_keys}
        total_weight = sum(weights.values())
        combined = sum(step_diffs[k] * (weights[k] / total_weight) for k in step_keys)
        
        # 完整prompt的差异向量
        full_a = extract_all_layer_hidden(model, tokenizer, case.prompt_a)
        full_b = extract_all_layer_hidden(model, tokenizer, case.prompt_b)
        full_diff = full_a[num_layers - 1].flatten() - full_b[num_layers - 1].flatten()
        
        # 组合方向与完整方向的对齐度
        cos_combined = float(torch.dot(combined, full_diff) / (combined.norm() * full_diff.norm() + 1e-10))
        
        return {
            "n_steps": len(step_diffs),
            "cos_between_steps_mean": statistics.mean(cos_between_steps) if cos_between_steps else 0,
            "cos_combined_vs_full": cos_combined,
            "step_norms": {str(k): float(v) for k, v in weights.items()},
        }
    
    return {"n_steps": len(step_diffs), "cos_between_steps_mean": 0, "cos_combined_vs_full": 0}


# ========== 策略3：高维探针（MLP） ==========
def strategy3_mlp_probe(hidden_a: Dict[int, torch.Tensor],
                          hidden_b: Dict[int, torch.Tensor],
                          hidden_dim: int,
                          num_layers: int,
                          n_train: int = 3,
                          n_test: int = 2) -> Dict:
    """
    用1层MLP做分类，测试非线性可分性。
    构造训练样本：每层+组合 → 训练MLP → 在新样本上测试
    """
    # 用最后一层的hidden state做分类
    # 在这里我们用"leave-one-out"方式：用n_train层训练，用剩余层测试
    # 但由于我们只有一对(A,B)，无法真正做交叉验证
    # 替代方案：用不同层组合构造"伪样本"
    
    # 构造"正样本"（A的hidden state）和"负样本"（B的hidden state）
    # 用多层hidden state组合构造更多样本
    features_a = []
    features_b = []
    
    for li in range(num_layers):
        h_a = hidden_a.get(li)
        h_b = hidden_b.get(li)
        if h_a is None or h_b is None:
            continue
        features_a.append(h_a.flatten())
        features_b.append(h_b.flatten())
    
    if len(features_a) < 4:
        return {"mlp_acc": -1, "note": "too few layers"}
    
    # 构造训练集：用前n_train层的差作为正样本特征，用负标签的
    # 简化版：直接训练一个MLP来分类delta的符号模式
    # 正类=A的每层hidden，负类=B的每层hidden
    
    X_train = []
    y_train = []
    
    # 用逐层delta的统计特征作为MLP输入
    for li in range(len(features_a)):
        delta = features_a[li] - features_b[li]
        # 特征：top-50维的绝对值 + 符号
        topk_vals, topk_idx = torch.topk(delta.abs(), min(50, hidden_dim))
        feat = torch.cat([topk_vals.float(), torch.sign(delta[topk_idx]).float()])
        X_train.append(feat)
        y_train.append(1.0)  # A > B in this dimension
    
    for li in range(len(features_b)):
        delta = features_b[li] - features_a[li]
        topk_vals, topk_idx = torch.topk(delta.abs(), min(50, hidden_dim))
        feat = torch.cat([topk_vals.float(), torch.sign(delta[topk_idx]).float()])
        X_train.append(feat)
        y_train.append(0.0)
    
    X_train = torch.stack(X_train)
    y_train = torch.tensor(y_train).unsqueeze(1)
    
    # 1层MLP探针
    input_dim = X_train.shape[1]
    mlp = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )
    
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
    
    # 训练
    mlp.train()
    for epoch in range(100):
        optimizer.zero_grad()
        pred = mlp(X_train)
        loss = F.binary_cross_entropy(pred, y_train)
        loss.backward()
        optimizer.step()
    
    # 测试：在训练集上的准确率（过拟合指标，但可以看是否能完美拟合）
    mlp.eval()
    with torch.no_grad():
        pred_test = mlp(X_train)
        pred_labels = (pred_test > 0.5).float().flatten()
        acc = float((pred_labels == y_train.flatten()).float().mean())
    
    # 交叉验证：留一层出来
    # 用前n-1层训练，第n层测试
    if len(features_a) >= 6:
        n_split = len(features_a) // 2
        
        X_tr = []
        y_tr = []
        X_te = []
        y_te = []
        
        for li in range(len(features_a)):
            delta = features_a[li] - features_b[li]
            topk_vals, topk_idx = torch.topk(delta.abs(), min(50, hidden_dim))
            feat = torch.cat([topk_vals.float(), torch.sign(delta[topk_idx]).float()])
            
            if li < n_split:
                X_tr.append(feat)
                y_tr.append(1.0)
            else:
                X_te.append(feat)
                y_te.append(1.0)
        
        for li in range(len(features_b)):
            delta = features_b[li] - features_a[li]
            topk_vals, topk_idx = torch.topk(delta.abs(), min(50, hidden_dim))
            feat = torch.cat([topk_vals.float(), torch.sign(delta[topk_idx]).float()])
            
            if li < n_split:
                X_tr.append(feat)
                y_tr.append(0.0)
            else:
                X_te.append(feat)
                y_te.append(0.0)
        
        X_tr = torch.stack(X_tr)
        y_tr = torch.tensor(y_tr).unsqueeze(1)
        X_te = torch.stack(X_te)
        y_te = torch.tensor(y_te).unsqueeze(1)
        
        mlp2 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        optimizer2 = torch.optim.Adam(mlp2.parameters(), lr=0.01)
        
        mlp2.train()
        for epoch in range(100):
            optimizer2.zero_grad()
            pred = mlp2(X_tr)
            loss = F.binary_cross_entropy(pred, y_tr)
            loss.backward()
            optimizer2.step()
        
        mlp2.eval()
        with torch.no_grad():
            pred_te = mlp2(X_te)
            pred_labels_te = (pred_te > 0.5).float().flatten()
            cv_acc = float((pred_labels_te == y_te.flatten()).float().mean()) if len(y_te) > 0 else -1
    else:
        cv_acc = -1
    
    return {
        "mlp_train_acc": acc,
        "mlp_cv_acc": cv_acc,
        "n_train_samples": len(X_train),
        "n_features": input_dim,
    }


# ========== 策略4：层级信息整合 ==========
def strategy4_layer_integration(hidden_a: Dict[int, torch.Tensor],
                                 hidden_b: Dict[int, torch.Tensor],
                                 hidden_dim: int,
                                 num_layers: int) -> Dict:
    """
    用所有层的hidden state做联合分类。
    方法：
    1. 对每层计算delta的特征（范数、top-k维度投影等）
    2. 用PCA降维所有层的delta
    3. 逐层累积信息的"增长曲线"
    """
    # 计算每层delta
    layer_deltas = {}
    for li in range(num_layers):
        h_a = hidden_a.get(li)
        h_b = hidden_b.get(li)
        if h_a is None or h_b is None:
            continue
        layer_deltas[li] = (h_a - h_b).flatten()
    
    if not layer_deltas:
        return {"note": "no layers"}
    
    # PCA on all layer deltas
    delta_list = [layer_deltas[k] for k in sorted(layer_deltas.keys())]
    stacked = torch.stack(delta_list)
    U_s, S_s, Vt_s = torch.linalg.svd(stacked, full_matrices=False)
    
    total_energy = (S_s ** 2).sum()
    cum_energy = torch.cumsum(S_s ** 2, dim=0) / total_energy
    
    # 逐层累积：信息如何随层数增长
    cumulative_info = []
    running_delta = torch.zeros(hidden_dim)
    prev_norm = 0
    
    for li in sorted(layer_deltas.keys()):
        running_delta = running_delta + layer_deltas[li]
        current_norm = float(running_delta.norm())
        incremental = current_norm - prev_norm
        
        # 投影到PCA空间
        proj = running_delta @ Vt_s[:min(5, len(Vt_s))].T
        
        cumulative_info.append({
            "layer": li,
            "running_norm": current_norm,
            "incremental": incremental,
            "pca5_proj_norm": float(proj.norm()),
        })
        prev_norm = current_norm
    
    # 逐层增量与最终方向的对齐度
    final_delta = sum(delta_list)
    final_unit = final_delta / (final_delta.norm() + 1e-10)
    
    layer_alignment = {}
    for li, d in layer_deltas.items():
        cos_val = float(torch.dot(d, final_unit) / (d.norm() + 1e-10))
        layer_alignment[li] = cos_val
    
    # 信息增长曲线的特征
    norms = [c["running_norm"] for c in cumulative_info]
    increments = [c["incremental"] for c in cumulative_info]
    
    # 找"信息突变层"：增量最大的层
    if increments:
        peak_idx = increments.index(max(increments))
        peak_layer = cumulative_info[peak_idx]["layer"]
        peak_increment = max(increments)
    else:
        peak_layer = 0
        peak_increment = 0
    
    return {
        "pca_top5_energy_pct": [float(e * 100) for e in cum_energy[:5]],
        "pca_rank90": int((cum_energy >= 0.9).nonzero()[0].item() + 1) if (cum_energy >= 0.9).any() else len(S_s),
        "peak_layer": peak_layer,
        "peak_increment": peak_increment,
        "total_norm": float(final_delta.norm()),
        "n_active_layers": len([d for d in layer_deltas.values() if d.norm() > 0.1]),
        "layer_alignment_range": f"[{min(layer_alignment.values()):.3f}, {max(layer_alignment.values()):.3f}]",
        "alignment_positive_pct": len([v for v in layer_alignment.values() if v > 0]) / len(layer_alignment) * 100,
    }


def run_experiment(model_key: str):
    print(f"\n{'='*70}")
    print(f"Stage668: P21 推理编码多策略解码 - {model_key}")
    print(f"{'='*70}")

    model, tokenizer = load_model_bundle(model_key)
    num_layers = len(discover_layers(model))
    device = next(model.parameters()).device
    
    # 获取hidden_dim
    hidden_dim = None
    for case in CASES[:1]:
        ha = extract_all_layer_hidden(model, tokenizer, case.prompt_a)
        for h in ha.values():
            if h is not None:
                hidden_dim = h.shape[-1]
                break
        break
    
    print(f"  模型: {model_key}, 层数: {num_layers}, 维度: {hidden_dim}")
    
    all_results = {}
    strategy_accuracies = {1: [], 2: [], 3: [], 4: []}
    
    for case in CASES:
        print(f"\n  [{case.reasoning_type}/{case.name}] ", end="", flush=True)
        
        # 测量margin
        margin_a = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
                   score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
        
        if margin_a == float('-inf') or abs(margin_a) < 0.01:
            print(f"margin={margin_a:.3f} (跳过，margin无意义)")
            all_results[case.name] = {"margin_a": margin_a, "skipped": True}
            continue
        
        # 提取hidden states
        hidden_a = extract_all_layer_hidden(model, tokenizer, case.prompt_a)
        hidden_b = extract_all_layer_hidden(model, tokenizer, case.prompt_b)
        
        case_result = {"margin_a": margin_a, "reasoning_type": case.reasoning_type}
        
        # ========== 策略1：门控模式 ==========
        gate = strategy1_gate_pattern(hidden_a, hidden_b, hidden_dim)
        case_result["strategy1"] = gate
        
        # 评估门控模式的区分度：A的符号模式 vs B的符号模式应该不同
        # 用符号模式的一致性作为度量
        pos_ratios = [gate[li]["pos_ratio"] for li in gate]
        neg_ratios = [gate[li]["neg_ratio"] for li in gate]
        gate_distinctness = statistics.mean([abs(p - n) for p, n in zip(pos_ratios, neg_ratios)])
        strategy_accuracies[1].append(gate_distinctness)
        
        # ========== 策略2：子任务分解 ==========
        print(f"子任务...", end=" ", flush=True)
        subtask = strategy2_subtask_decomposition(model, tokenizer, case, num_layers)
        case_result["strategy2"] = subtask
        strategy_accuracies[2].append(abs(subtask.get("cos_combined_vs_full", 0)))
        
        # ========== 策略3：MLP探针 ==========
        print(f"MLP...", end=" ", flush=True)
        mlp = strategy3_mlp_probe(hidden_a, hidden_b, hidden_dim, num_layers)
        case_result["strategy3"] = mlp
        strategy_accuracies[3].append(mlp.get("mlp_cv_acc", mlp.get("mlp_train_acc", -1)))
        
        # ========== 策略4：层级整合 ==========
        integration = strategy4_layer_integration(hidden_a, hidden_b, hidden_dim, num_layers)
        case_result["strategy4"] = integration
        # 层级整合的"恢复率"：正对齐层比例
        alignment_pct = integration.get("alignment_positive_pct", 0)
        strategy_accuracies[4].append(alignment_pct / 100)
        
        print(f"m={margin_a:.3f} | S1(gate)={gate_distinctness:.3f} | "
              f"S2(sub)={subtask.get('cos_combined_vs_full', 0):.3f} | "
              f"S3(mlp_cv)={mlp.get('mlp_cv_acc', -1):.3f} | "
              f"S4(integ)={alignment_pct:.1f}%")
        
        all_results[case.name] = case_result
    
    # ========== 汇总 ==========
    print(f"\n{'='*70}")
    print(f"汇总：各策略解码效果")
    print(f"{'='*70}")
    
    for s_id, s_name in [(1, "门控模式"), (2, "子任务分解"), (3, "MLP探针"), (4, "层级整合")]:
        accs = strategy_accuracies[s_id]
        if accs:
            valid = [a for a in accs if a >= 0]
            if valid:
                print(f"  策略{s_id}({s_name}): mean={statistics.mean(valid):.3f}, "
                      f"max={max(valid):.3f}, n={len(valid)}")
            else:
                print(f"  策略{s_id}({s_name}): 无有效结果")
    
    # INV-297判伪
    best_strategy_acc = 0
    for s_id in [1, 2, 3, 4]:
        valid = [a for a in strategy_accuracies[s_id] if a >= 0]
        if valid:
            best = max(valid)
            if best > best_strategy_acc:
                best_strategy_acc = best
    
    print(f"\n  INV-297: 最佳策略恢复率={best_strategy_acc:.3f} → "
          f"{'>0.3 确认(推理低维编码)' if best_strategy_acc > 0.3 else '<0.3 推翻(推理高维编码)'}")
    print(f"  INV-298: 所有策略恢复率<0.1 → "
          f"{'确认(推理高维分布式编码)' if best_strategy_acc < 0.1 else '推翻(至少一种策略有效)'}")
    
    # 保存
    out_path = OUTPUT_DIR / f"stage668_{model_key}_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")
    
    free_model(model)
    return all_results


if __name__ == "__main__":
    model_key = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_key)
