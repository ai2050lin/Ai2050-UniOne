#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage664: P17 增量差抵消的信号/噪声分离

P14发现：56-78%的增量差被抵消
P17目标：对逐层增量差做SVD分解，测试"信号被保留、噪声被抵消"假说

核心思路：
1. 提取每层的increment_diff_k = increment_A_k - increment_B_k
2. 对每个increment_diff_k做SVD，分解为top-k(信号)和rest(噪声)
3. 计算最终d_L中的信号/噪声保留率
4. 比较信号分量vs噪声分量的逐层演化
5. 测试：被抵消的分量是否集中在"噪声"维度

三个假说：
H1(噪声过滤): 信号分量保留率高(>70%)，噪声分量保留率低(<30%)
H2(维度分离): 不同层的增量差在方向上逐渐正交化
H3(精度控制): 保留的分量数量决定了编码精度

预注册判伪：
INV-285: "信号分量保留率 > 噪声分量保留率"
  如果信号保留率 < 噪声保留率，则抵消不是噪声过滤
INV-286: "被抵消分量的方向接近随机(与d_L的cos接近0)"
INV-287: "SVD top-k的累积保留率随k增长，存在明显的'拐点'"
"""

from __future__ import annotations

import sys
import io
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    discover_layers,
    free_model,
    load_model_bundle,
    score_candidate_avg_logprob,
    evenly_spaced_layers,
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


@dataclass(frozen=True)
class TestCase:
    name: str
    prompt_a: str
    prompt_b: str
    positive_a: str
    negative_a: str


CASES = [
    TestCase(
        name="bank_financial",
        prompt_a="The bank approved the loan with favorable terms.",
        prompt_b="The river bank was covered with wild flowers.",
        positive_a="financial",
        negative_a="river",
    ),
    TestCase(
        name="grammar_subj",
        prompt_a="The tall building stands proudly.",
        prompt_b="The tall buildings stand proudly.",
        positive_a="stands",
        negative_a="stand",
    ),
    TestCase(
        name="coreference",
        prompt_a="Mary gave the book to John because she wanted to help.",
        prompt_b="Mary gave the book to John because he wanted to help.",
        positive_a="Mary",
        negative_a="John",
    ),
    TestCase(
        name="relation_capital",
        prompt_a="The capital of France is",
        prompt_b="The capital of Japan is",
        positive_a="Paris",
        negative_a="Tokyo",
    ),
]


def extract_all_layer_hidden(model, tokenizer, prompt: str) -> Dict[int, torch.Tensor]:
    """提取所有层的最后一个token的hidden state"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    device = next(model.parameters()).device
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


def analyze_increment_diff_signal_noise(increment_diffs: Dict[int, torch.Tensor],
                                          d_final: torch.Tensor,
                                          hidden_dim: int) -> Dict:
    """
    对逐层增量差做信号/噪声分离分析
    
    核心思路：
    - 对增量差矩阵(stacked)做PCA，得到主方向（信号空间）
    - 信号方向 = PCA top-k方向
    - 对每个increment_diff_k，分解为信号分量和噪声分量（垂直于信号空间）
    - 逐层追踪信号/噪声的累积和保留率
    """
    n_layers = len(increment_diffs)
    d_norm = float(d_final.norm())
    d_final_flat = d_final.flatten()
    d_unit = d_final_flat / (d_norm + 1e-10)
    
    # ========== 逐层分解 ==========
    # 信号分量：投影到d_final方向；噪声分量：垂直于d_final
    layer_analysis = {}
    for li in range(n_layers):
        diff_k = increment_diffs[li]
        if diff_k is None or diff_k.norm() < 1e-10:
            layer_analysis[li] = {"norm": 0, "signal_proj": 0, "noise_norm": 0,
                                   "signal_ratio": 0, "alignment_cos": 0}
            continue
        
        diff_k_flat = diff_k.flatten()
        norm_k = float(diff_k_flat.norm())
        signal_proj = float(torch.dot(diff_k_flat, d_unit))
        signal_component = signal_proj * d_unit
        noise_component = diff_k_flat - signal_component
        noise_norm = float(noise_component.norm())
        
        layer_analysis[li] = {
            "norm": norm_k,
            "signal_proj": signal_proj,
            "signal_abs": abs(signal_proj),
            "noise_norm": noise_norm,
            "signal_ratio": abs(signal_proj) / (norm_k + 1e-10),
            "noise_ratio": noise_norm / (norm_k + 1e-10),
            "alignment_cos": float(torch.dot(diff_k_flat, d_unit) / (norm_k + 1e-10)),
        }
    
    # ========== 逐层累加的信号/噪声分离 ==========
    # 用PCA on stacked increment diffs定义信号空间
    diff_list = [increment_diffs[li].flatten() for li in range(n_layers) if increment_diffs[li] is not None]
    
    if not diff_list:
        return {"layer_analysis": layer_analysis, "cumulative": {}, "n_signal_dims": 1, "d_final_norm": d_norm}
    
    stacked = torch.stack(diff_list)
    U_s, S_s, Vt_s = torch.linalg.svd(stacked, full_matrices=False)
    total_energy = (S_s ** 2).sum()
    cumsum_s = torch.cumsum(S_s ** 2, dim=0) / total_energy
    rank90 = (cumsum_s >= 0.9).nonzero()
    n_signal_dims = int(rank90[0].item() + 1) if len(rank90) > 0 else min(10, len(S_s))
    signal_directions = Vt_s[:n_signal_dims]
    
    cumulative_results = {}
    running_sum = torch.zeros(hidden_dim)
    for li in range(n_layers):
        diff_k = increment_diffs[li]
        if diff_k is not None:
            running_sum = running_sum + diff_k.flatten()
        
        signal_proj = running_sum @ signal_directions.T
        signal_component = signal_proj @ signal_directions
        noise_component = running_sum - signal_component
        
        signal_norm = float(signal_component.norm())
        noise_norm = float(noise_component.norm())
        total_norm = float(running_sum.norm())
        
        cumulative_results[li] = {
            "running_norm": total_norm,
            "signal_norm": signal_norm,
            "noise_norm": noise_norm,
            "signal_pct": float(signal_norm / (total_norm + 1e-10)),
            "noise_pct": float(noise_norm / (total_norm + 1e-10)),
            "signal_retention_vs_final": signal_norm / (d_norm + 1e-10),
            "noise_retention_vs_final": noise_norm / (d_norm + 1e-10),
        }
    
    return {
        "layer_analysis": layer_analysis,
        "cumulative": cumulative_results,
        "n_signal_dims": n_signal_dims,
        "d_final_norm": d_norm,
        "pca_top_sv": [float(x) for x in S_s[:10]],
    }


def analyze_cancellation_pattern(increment_diffs: Dict[int, torch.Tensor],
                                  d_final: torch.Tensor) -> Dict:
    n_layers = len(increment_diffs)
    
    layer_alignment = {}
    d_flat = d_final.flatten()
    d_norm = float(d_flat.norm())
    for li in range(n_layers):
        diff_k = increment_diffs[li]
        if diff_k is None:
            layer_alignment[li] = 0.0
            continue
        dk = diff_k.flatten()
        cos_val = float(torch.dot(dk, d_flat) / (dk.norm() * d_norm + 1e-10))
        layer_alignment[li] = cos_val
    
    adjacent_cos = []
    for li in range(n_layers - 1):
        d1 = increment_diffs[li]
        d2 = increment_diffs[li + 1]
        if d1 is None or d2 is None or d1.norm() < 1e-10 or d2.norm() < 1e-10:
            continue
        cos_val = float(torch.dot(d1.flatten(), d2.flatten()) / (d1.norm() * d2.norm() + 1e-10))
        adjacent_cos.append(cos_val)
    
    norms = []
    for li in range(n_layers):
        diff_k = increment_diffs[li]
        if diff_k is not None:
            norms.append(float(diff_k.norm()))
    
    if norms:
        sum_norms = sum(norms)
        actual_cancellation = 1 - d_norm / sum_norms if sum_norms > 0 else 0
        
        dim = d_flat.shape[0]
        n_sim = 100
        theoretical_cancel_rates = []
        for _ in range(n_sim):
            random_vecs = [torch.randn(dim) * n for n in norms]
            random_sum = sum(random_vecs)
            random_cancel = 1 - float(random_sum.norm()) / sum_norms if sum_norms > 0 else 0
            theoretical_cancel_rates.append(random_cancel)
        theoretical_mean = statistics.mean(theoretical_cancel_rates)
        theoretical_std = statistics.stdev(theoretical_cancel_rates) if len(theoretical_cancel_rates) > 1 else 0
    else:
        actual_cancellation = 0
        theoretical_mean = 0
        theoretical_std = 0
    
    return {
        "layer_alignment_with_final": layer_alignment,
        "adjacent_cos_mean": statistics.mean(adjacent_cos) if adjacent_cos else 0,
        "adjacent_cos_std": statistics.stdev(adjacent_cos) if len(adjacent_cos) > 1 else 0,
        "actual_cancellation_rate": actual_cancellation,
        "theoretical_random_cancel_rate": theoretical_mean,
        "theoretical_random_cancel_std": theoretical_std,
        "cancellation_vs_random_diff": actual_cancellation - theoretical_mean,
        "z_score": (actual_cancellation - theoretical_mean) / (theoretical_std + 1e-10),
    }


def run_experiment(model_key: str):
    print(f"\n{'='*70}")
    print(f"Stage664: P17 信号/噪声分离 - {model_key}")
    print(f"{'='*70}")

    model, tokenizer = load_model_bundle(model_key)
    num_layers = len(discover_layers(model))
    device = next(model.parameters()).device
    hidden_dim = None

    all_results = {}

    for case in CASES:
        print(f"\n--- {case.name} ---")
        
        # 提取所有层hidden state
        print("  提取hidden states...")
        hidden_a = extract_all_layer_hidden(model, tokenizer, case.prompt_a)
        hidden_b = extract_all_layer_hidden(model, tokenizer, case.prompt_b)
        
        # 获取hidden_dim
        for li, h in hidden_a.items():
            if h is not None:
                hidden_dim = h.shape[-1]
                break
        
        # 计算逐层增量差
        increment_diffs = {}
        prev_a = None
        prev_b = None
        for li in range(num_layers):
            h_a = hidden_a.get(li)
            h_b = hidden_b.get(li)
            if h_a is None or h_b is None:
                increment_diffs[li] = None
                continue
            
            if prev_a is None:
                # 第一层：increment = h_l (因为h_0通常是embedding)
                increment_a = h_a
                increment_b = h_b
            else:
                increment_a = h_a - prev_a
                increment_b = h_b - prev_b
            
            increment_diffs[li] = increment_a - increment_b
            prev_a = h_a
            prev_b = h_b
        
        # 最终d_L
        d_final = hidden_a[num_layers - 1] - hidden_b[num_layers - 1]
        
        # ========== 实验1：信号/噪声分离 ==========
        print("  实验1: 信号/噪声分离...")
        sn_results = analyze_increment_diff_signal_noise(increment_diffs, d_final, hidden_dim)
        
        # 打印关键结果
        print(f"  d_final维度: {hidden_dim}, 信号维度: {sn_results['n_signal_dims']}")
        print(f"  d_final范数: {sn_results['d_final_norm']:.4f}")
        print(f"  PCA信号维度(90%能量): {sn_results['n_signal_dims']}")
        
        # 逐层增量差的特征
        print(f"\n  逐层增量差分析:")
        print(f"  {'层':>3} | {'norm':>8} | {'sig_proj':>8} | {'noise_norm':>9} | {'sig_ratio':>8} | {'align_cos':>9}")
        print(f"  {'-'*3}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*9}")
        for li in range(num_layers):
            la = sn_results['layer_analysis'].get(li, {})
            if la.get('norm', 0) > 0:
                print(f"  {li:>3} | {la['norm']:>8.4f} | {la['signal_proj']:>8.4f} | "
                      f"{la['noise_norm']:>9.4f} | {la['signal_ratio']*100:>7.1f}% | "
                      f"{la['alignment_cos']:>9.4f}")
        
        # 累积保留率
        cum = sn_results['cumulative']
        print(f"\n  逐层累积信号/噪声保留率(vs d_final):")
        print(f"  {'层':>3} | {'running':>8} | {'signal':>8} | {'noise':>8} | {'sig_ret':>7} | {'noise_ret':>9}")
        print(f"  {'-'*3}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}")
        for li in range(num_layers):
            if li in cum:
                c = cum[li]
                print(f"  {li:>3} | {c['running_norm']:>8.4f} | {c['signal_norm']:>8.4f} | "
                      f"{c['noise_norm']:>8.4f} | {c['signal_retention_vs_final']*100:>6.1f}% | "
                      f"{c['noise_retention_vs_final']*100:>8.1f}%")
        
        # ========== 实验2：抵消模式分析 ==========
        print(f"\n  实验2: 抵消模式分析...")
        cancel_results = analyze_cancellation_pattern(increment_diffs, d_final)
        
        print(f"  实际抵消率: {cancel_results['actual_cancellation_rate']*100:.1f}%")
        print(f"  理论随机抵消率: {cancel_results['theoretical_random_cancel_rate']*100:.1f}% "
              f"(±{cancel_results['theoretical_random_cancel_std']*100:.1f}%)")
        print(f"  差异: {cancel_results['cancellation_vs_random_diff']*100:.1f}%")
        print(f"  z-score: {cancel_results['z_score']:.2f}")
        print(f"  相邻层cos均值: {cancel_results['adjacent_cos_mean']:.4f} "
              f"(±{cancel_results['adjacent_cos_std']:.4f})")
        
        # 各层与d_L的对齐
        alignment = cancel_results['layer_alignment_with_final']
        aligned_layers = [li for li, a in alignment.items() if a > 0.3]
        anti_aligned = [li for li, a in alignment.items() if a < -0.3]
        print(f"  正对齐层(cos>0.3): {aligned_layers}")
        print(f"  反对齐层(cos<-0.3): {anti_aligned}")
        
        # ========== 实验3：信号/噪声假说检验 ==========
        print(f"\n  实验3: 信号/噪声假说检验")
        
        # 对每个层k，计算增量差的信号投影vs噪声投影对最终d_L的贡献
        signal_contributions = []
        noise_contributions = []
        for li in range(num_layers):
            diff_k = increment_diffs[li]
            if diff_k is None or diff_k.norm() < 1e-10:
                continue
            
            dk = diff_k.flatten()
            d_unit = d_final.flatten() / (d_final.norm() + 1e-10)
            signal_proj = float(torch.dot(dk, d_unit))
            noise_proj_vec = dk - signal_proj * d_unit
            noise_proj_norm = float(noise_proj_vec.norm())
            
            signal_contributions.append(signal_proj)
            noise_contributions.append(noise_proj_norm)
        
        if signal_contributions:
            # 信号方向的净投影（正-负）
            net_signal = sum(signal_contributions)
            total_signal_magnitude = sum(abs(s) for s in signal_contributions)
            signal_efficiency = abs(net_signal) / (total_signal_magnitude + 1e-10)
            
            print(f"  信号方向净投影: {net_signal:.4f}")
            print(f"  信号方向总幅度: {total_signal_magnitude:.4f}")
            print(f"  信号效率(净/总): {signal_efficiency*100:.1f}%")
            print(f"  噪声方向总幅度: {sum(noise_contributions):.4f}")
            
            # 假说判断
            if signal_efficiency > 0.5:
                print(f"  H1(噪声过滤): 信号效率={signal_efficiency*100:.1f}% > 50% → 部分支持")
            else:
                print(f"  H1(噪声过滤): 信号效率={signal_efficiency*100:.1f}% < 50% → 不支持")
            
            if cancel_results['z_score'] > 1.96:
                print(f"  H1补充: 实际抵消率显著高于随机(z={cancel_results['z_score']:.2f}>1.96) → 抵消是有序的")
            elif cancel_results['z_score'] < -1.96:
                print(f"  H1补充: 实际抵消率显著低于随机(z={cancel_results['z_score']:.2f}<-1.96) → 网络在抑制抵消")
            else:
                print(f"  H1补充: 实际抵消率与随机无显著差异(z={cancel_results['z_score']:.2f}) → 抵消可能是随机的")
        
        # ========== 实验4：不同k的累积保留率 ==========
        print(f"\n  实验4: 不同k的信号维度保留率(基于PCA)")
        diff_list4 = [increment_diffs[li].flatten() for li in range(num_layers) if increment_diffs[li] is not None]
        if diff_list4:
            stacked4 = torch.stack(diff_list4)
            U_s4, S_s4, Vt_s4 = torch.linalg.svd(stacked4, full_matrices=False)
            for k_test in [1, 2, 3, 5, 10, min(20, len(S_s4))]:
                if k_test > len(S_s4):
                    continue
                signal_dirs_k = Vt_s4[:k_test]
                running_sum4 = torch.zeros(hidden_dim)
                for li in range(num_layers):
                    diff_k = increment_diffs[li]
                    if diff_k is not None:
                        running_sum4 = running_sum4 + diff_k.flatten()
                proj = running_sum4 @ signal_dirs_k.T
                recon = proj @ signal_dirs_k
                ret = float(recon.norm() / (d_final.norm() + 1e-10))
                print(f"  k={k_test:>3}: 最终保留率={ret*100:.1f}%")
        
        all_results[case.name] = {
            "signal_noise": {k: v for k, v in sn_results.items() if k != 'cumulative'},
            "cumulative_final": {li: cum[li] for li in cum},
            "cancellation": cancel_results,
            "signal_efficiency": signal_efficiency if signal_contributions else 0,
        }

    # ========== 汇总统计 ==========
    print(f"\n{'='*70}")
    print(f"汇总统计")
    print(f"{'='*70}")
    
    case_names = list(all_results.keys())
    cancellation_rates = [all_results[cn]['cancellation']['actual_cancellation_rate'] for cn in case_names]
    z_scores = [all_results[cn]['cancellation']['z_score'] for cn in case_names]
    signal_efficiencies = [all_results[cn]['signal_efficiency'] for cn in case_names]
    
    print(f"\n  抵消率: {[f'{r*100:.1f}%' for r in cancellation_rates]}")
    print(f"  均值: {statistics.mean(cancellation_rates)*100:.1f}%")
    print(f"  z-scores: {[f'{z:.2f}' for z in z_scores]}")
    print(f"  信号效率: {[f'{e*100:.1f}%' for e in signal_efficiencies]}")
    print(f"  均值: {statistics.mean(signal_efficiencies)*100:.1f}%")
    
    # INV-285判伪
    avg_signal_eff = statistics.mean(signal_efficiencies)
    print(f"\n  INV-285: 信号效率均值={avg_signal_eff*100:.1f}% {'> 50% → 确认' if avg_signal_eff > 0.5 else '< 50% → 推翻'}")
    
    # INV-286: 被抵消分量与d_L的cos
    avg_z = statistics.mean([abs(z) for z in z_scores])
    print(f"  INV-286: z-score均值={avg_z:.2f} {'> 1.96 → 抵消有序(推翻)' if avg_z > 1.96 else '< 1.96 → 抵消随机(确认)'}")
    
    # 保存结果
    out_path = OUTPUT_DIR / f"stage664_{model_key}_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")
    
    free_model(model)
    return all_results


if __name__ == "__main__":
    model_key = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_key)
