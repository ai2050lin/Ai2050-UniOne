#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage673: P27 多步生成过程分析框架 — Logit方程修复v2

P15-P16核心结论：Logit方程失败是"范式"问题——单步logit分析无法解释margin
P27目标：建立多步生成过程的分析框架，追踪消歧信号在自回归生成中的传播

核心实验：
  A. 多步margin追踪：自回归生成N步，每步计算margin
  B. 隐藏状态投影演化：追踪d_L在每步对unembed矩阵的投影
  C. 首token锚定效应：第一个生成的token如何锁定后续生成方向
  D. 多步信号放大：消歧信号是否在多步中被放大？
  E. 跨模型对比：单域vs多域模型的多步生成动力学差异
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


# 测试用例：消歧 + 关系 + 语法
@dataclass(frozen=True)
class TestCase:
    capability: str
    prompt: str
    correct: str
    incorrect: str
    continuation: str  # 正确答案后的续写

CASES = [
    TestCase("disamb", "The river bank was",
             "muddy", "approved", " and the water was rising fast."),
    TestCase("relation", "The capital of France is",
             "Paris", "Berlin", ", and it is famous for the Eiffel Tower."),
    TestCase("syntax", "She ran quickly to the",
             "store", "quickly", " because she needed milk."),
]


def get_unembed_matrix(model, hidden_dim: int) -> torch.Tensor:
    """提取unembed矩阵 W_u"""
    for name, param in model.named_parameters():
        if 'lm_head' in name.lower() and param.dim() == 2:
            # lm_head: [vocab_size, hidden_dim] or [hidden_dim, vocab_size]
            if param.shape[0] > 1000 and param.shape[1] == hidden_dim:
                return param.data.float()
            elif param.shape[1] > 1000 and param.shape[0] == hidden_dim:
                return param.data.T.float()  # 转置
        elif 'unembed' in name.lower() or 'output' in name.lower():
            if param.dim() == 2 and param.shape[0] > 1000:
                return param.data.float()
    # 尝试从embed_weight的转置获取
    for name, param in model.named_parameters():
        if 'embed' in name.lower() and param.dim() == 2 and param.shape[0] > 1000:
            return param.data.float()
    return None


def get_token_logits(model, tokenizer, input_ids: torch.Tensor, 
                      target_token_id: int, correct_id: int, incorrect_id: int,
                      temperature: float = 1.0) -> Dict:
    """从模型forward直接获取logits"""
    device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(input_ids.to(device), output_hidden_states=True)
        logits = outputs.logits[0, -1, :].float().cpu()  # 最后一个位置
        last_hidden = outputs.hidden_states[-1][0, -1, :].float().cpu()
    
    # 温度缩放
    if temperature != 1.0:
        logits = logits / temperature
    
    probs = F.softmax(logits, dim=-1)
    
    return {
        "logit_correct": float(logits[correct_id]),
        "logit_incorrect": float(logits[incorrect_id]),
        "logit_target": float(logits[target_token_id]),
        "margin": float(logits[correct_id] - logits[incorrect_id]),
        "prob_correct": float(probs[correct_id]),
        "prob_incorrect": float(probs[incorrect_id]),
        "prob_target": float(probs[target_token_id]),
        "prob_ratio": float(probs[correct_id] / (probs[incorrect_id] + 1e-10)),
        "hidden_norm": float(last_hidden.norm()),
        "hidden": last_hidden,
    }


def experiment_a_multistep_margin(model, tokenizer, case: TestCase) -> Dict:
    """实验A: 多步margin追踪"""
    print(f"\n  A. 多步margin追踪 [{case.capability}]")
    print(f"    Prompt: '{case.prompt}'")
    print(f"    Correct: '{case.correct}', Incorrect: '{case.incorrect}'")
    
    correct_ids = tokenizer.encode(case.correct, add_special_tokens=False)
    incorrect_ids = tokenizer.encode(case.incorrect, add_special_tokens=False)
    
    if not correct_ids or not incorrect_ids:
        print(f"    ERROR: token编码失败")
        return {}
    
    correct_id = correct_ids[0] if len(correct_ids) == 1 else correct_ids[0]
    incorrect_id = incorrect_ids[0] if len(incorrect_ids) == 1 else incorrect_ids[0]
    
    device = next(model.parameters()).device
    results = {"steps": []}
    
    # 逐步生成并追踪margin
    current_ids = tokenizer.encode(case.prompt, return_tensors="pt")
    current_ids = current_ids.to(device)
    
    max_steps = min(8, len(tokenizer.encode(case.correct + " " + case.continuation, add_special_tokens=False)) + 2)
    
    print(f"    {'Step':>4} {'Token':>10} {'LogitC':>8} {'LogitI':>8} {'Margin':>8} {'ProbC':>7} {'ProbI':>7} {'Ratio':>8}")
    print(f"    {'-'*70}")
    
    for step in range(max_steps):
        with torch.no_grad():
            outputs = model(current_ids, output_hidden_states=True)
            logits = outputs.logits[0, -1, :].float().cpu()
            probs = F.softmax(logits, dim=-1)
            last_hidden = outputs.hidden_states[-1][0, -1, :].float().cpu()
        
        logit_c = float(logits[correct_id])
        logit_i = float(logits[incorrect_id])
        margin = logit_c - logit_i
        prob_c = float(probs[correct_id])
        prob_i = float(probs[incorrect_id])
        
        # 获取实际生成的token
        top_token_id = int(torch.argmax(logits))
        top_token = tokenizer.decode([top_token_id]).strip()
        
        step_data = {
            "step": step,
            "token": top_token,
            "logit_correct": logit_c,
            "logit_incorrect": logit_i,
            "margin": margin,
            "prob_correct": prob_c,
            "prob_incorrect": prob_i,
            "prob_ratio": prob_c / (prob_i + 1e-10),
            "hidden_norm": float(last_hidden.norm()),
        }
        results["steps"].append(step_data)
        
        print(f"    {step:>4} {top_token:>10} {logit_c:>8.2f} {logit_i:>8.2f} {margin:>8.2f} "
              f"{prob_c:>7.4f} {prob_i:>7.4f} {step_data['prob_ratio']:>8.1f}")
        
        # 追加生成的token
        current_ids = torch.cat([current_ids, torch.tensor([[top_token_id]], device=device)], dim=1)
        
        if top_token_id == correct_id:
            break
    
    # 分析margin趋势
    if len(results["steps"]) > 1:
        margins = [s["margin"] for s in results["steps"]]
        results["margin_trend"] = "increasing" if margins[-1] > margins[0] else "decreasing"
        results["margin_ratio"] = margins[-1] / (abs(margins[0]) + 1e-10)
        print(f"\n    Margin趋势: {results['margin_trend']} (首步={margins[0]:.2f}, 末步={margins[-1]:.2f}, 比={results['margin_ratio']:.2f}x)")
    
    torch.cuda.empty_cache()
    return results


def experiment_b_hidden_projection(model, tokenizer, case: TestCase, 
                                     unembed: Optional[torch.Tensor]) -> Dict:
    """实验B: 隐藏状态对unembed的投影分析"""
    print(f"\n  B. 隐藏状态投影分析 [{case.capability}]")
    
    correct_ids = tokenizer.encode(case.correct, add_special_tokens=False)
    incorrect_ids = tokenizer.encode(case.incorrect, add_special_tokens=False)
    correct_id = correct_ids[0] if correct_ids else 0
    incorrect_id = incorrect_ids[0] if incorrect_ids else 0
    
    if unembed is None:
        print(f"    跳过: 无法获取unembed矩阵")
        return {}
    
    # 确保token id在unembed范围内
    if correct_id >= unembed.shape[0] or incorrect_id >= unembed.shape[0]:
        print(f"    跳过: token id超出unembed范围 ({correct_id}, {incorrect_id} vs {unembed.shape[0]})")
        return {}
    
    device = next(model.parameters()).device
    unembed_cpu = unembed.cpu()  # 确保在CPU上
    
    # 获取最后一步的hidden state
    input_ids = tokenizer.encode(case.prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        h_L = outputs.hidden_states[-1][0, -1, :].float().cpu()
        logits = outputs.logits[0, -1, :].float().cpu()
    
    # u_correct和u_incorrect: unembed矩阵的对应行
    u_correct = unembed_cpu[correct_id]  # [hidden_dim]
    u_incorrect = unembed_cpu[incorrect_id]
    u_diff = u_correct - u_incorrect
    
    # 各种投影计算
    cos_h_u = float(torch.dot(h_L, u_diff) / (h_L.norm() * u_diff.norm() + 1e-10))
    proj_h_on_udiff = float(torch.dot(h_L, u_diff) / (u_diff.norm() + 1e-10))
    
    # h_L的top-10维度对margin的贡献
    h_abs = torch.abs(h_L)
    topk_vals, topk_idx = torch.topk(h_abs, min(20, len(h_abs)))
    
    # 将h_L的top-k维度投影到u_diff上
    h_topk = torch.zeros_like(h_L)
    h_topk[topk_idx] = h_L[topk_idx]
    cos_topk = float(torch.dot(h_topk, u_diff) / (h_topk.norm() * u_diff.norm() + 1e-10))
    margin_topk = float(torch.dot(h_topk, u_diff))
    
    # 去掉top-k后的贡献
    h_rest = h_L.clone()
    h_rest[topk_idx] = 0
    cos_rest = float(torch.dot(h_rest, u_diff) / (h_rest.norm() * u_diff.norm() + 1e-10))
    margin_rest = float(torch.dot(h_rest, u_diff))
    
    # full margin via logit
    full_margin = float(logits[correct_id] - logits[incorrect_id])
    
    # unembed矩阵的范数
    u_norm_avg = float(unembed_cpu.norm(dim=1).mean())
    
    print(f"    ||h_L|| = {h_L.norm():.2f}")
    print(f"    ||u_diff|| = {u_diff.norm():.4f}")
    print(f"    cos(h_L, u_diff) = {cos_h_u:.4f}")
    print(f"    proj(h_L, u_diff) = {proj_h_on_udiff:.4f}")
    print(f"    full_margin (logit) = {full_margin:.4f}")
    print(f"    单步logit方程预测: cos×||h||×||u|| = {cos_h_u * h_L.norm() * u_diff.norm():.4f}")
    print(f"    预测/实际 = {(cos_h_u * h_L.norm() * u_diff.norm()) / (abs(full_margin) + 1e-10):.2f}x")
    print(f"\n    Top-20维度分析:")
    print(f"      top-20贡献: margin={margin_topk:.4f}, cos={cos_topk:.4f}, 占比={abs(margin_topk)/(abs(full_margin)+1e-10):.1%}")
    print(f"      其余维度贡献: margin={margin_rest:.4f}, cos={cos_rest:.4f}, 占比={abs(margin_rest)/(abs(full_margin)+1e-10):.1%}")
    print(f"      ||u||均值 = {u_norm_avg:.4f}")
    
    torch.cuda.empty_cache()
    
    return {
        "cos_h_udiff": cos_h_u,
        "logit_equation_pred": cos_h_u * h_L.norm() * u_diff.norm(),
        "actual_margin": full_margin,
        "ratio": (cos_h_u * h_L.norm() * u_diff.norm()) / (abs(full_margin) + 1e-10),
        "top20_contribution_ratio": abs(margin_topk) / (abs(full_margin) + 1e-10),
        "rest_contribution_ratio": abs(margin_rest) / (abs(full_margin) + 1e-10),
    }


def experiment_c_first_token_anchoring(model, tokenizer, case: TestCase) -> Dict:
    """实验C: 首token锚定效应"""
    print(f"\n  C. 首token锚定效应 [{case.capability}]")
    
    correct_ids = tokenizer.encode(case.correct, add_special_tokens=False)
    incorrect_ids = tokenizer.encode(case.incorrect, add_special_tokens=False)
    correct_id = correct_ids[0] if correct_ids else 0
    incorrect_id = incorrect_ids[0] if incorrect_ids else 0
    
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(case.prompt, return_tensors="pt").to(device)
    
    # Scenario 1: 自然生成（首token自由选择）
    with torch.no_grad():
        out_natural = model(prompt_ids, output_hidden_states=True)
        logits_natural = out_natural.logits[0, -1, :].float().cpu()
        probs_natural = F.softmax(logits_natural, dim=-1)
    
    margin_natural = float(logits_natural[correct_id] - logits_natural[incorrect_id])
    top1_id = int(torch.argmax(logits_natural))
    top1_prob = float(probs_natural[top1_id])
    
    # Scenario 2: 强制correct为第一个token
    forced_correct_ids = torch.cat([prompt_ids, torch.tensor([[correct_id]], device=device)], dim=1)
    with torch.no_grad():
        out_forced_c = model(forced_correct_ids, output_hidden_states=True)
        logits_forced_c = out_forced_c.logits[0, -1, :].float().cpu()
    
    # 在强制correct后，margin对续写token的变化
    cont_tokens = tokenizer.encode(case.continuation.strip().split()[0] if case.continuation.strip() else "the", 
                                    add_special_tokens=False)
    if cont_tokens:
        cont_id = cont_tokens[0]
        margin_after_correct = float(logits_forced_c[cont_id]) - float(logits_forced_c.min())
    else:
        margin_after_correct = float(logits_forced_c.max()) - float(logits_forced_c.min())
    
    # Scenario 3: 强制incorrect为第一个token
    forced_incorrect_ids = torch.cat([prompt_ids, torch.tensor([[incorrect_id]], device=device)], dim=1)
    with torch.no_grad():
        out_forced_i = model(forced_incorrect_ids, output_hidden_states=True)
        logits_forced_i = out_forced_i.logits[0, -1, :].float().cpu()
    
    if cont_tokens:
        margin_after_incorrect = float(logits_forced_i[cont_id]) - float(logits_forced_i.min())
    else:
        margin_after_incorrect = float(logits_forced_i.max()) - float(logits_forced_i.min())
    
    # 首token后hidden state的差异
    h_correct = out_forced_c.hidden_states[-1][0, -1, :].float().cpu()
    h_incorrect = out_forced_i.hidden_states[-1][0, -1, :].float().cpu()
    h_diff_norm = float((h_correct - h_incorrect).norm())
    h_natural_last = out_natural.hidden_states[-1][0, -1, :].float().cpu()
    
    print(f"    自然生成: margin={margin_natural:.3f}, top1='{tokenizer.decode([top1_id]).strip()}' (p={top1_prob:.3f})")
    print(f"    强制correct后: 续写logit_range={margin_after_correct:.3f}")
    print(f"    强制incorrect后: 续写logit_range={margin_after_incorrect:.3f}")
    print(f"    首token后的h_diff范数: {h_diff_norm:.2f}")
    print(f"    锚定强度: {h_diff_norm / (h_natural_last.norm() + 1e-10):.4f}")
    
    torch.cuda.empty_cache()
    
    return {
        "natural_margin": margin_natural,
        "natural_top1": tokenizer.decode([top1_id]).strip(),
        "forced_correct_continuation": margin_after_correct,
        "forced_incorrect_continuation": margin_after_incorrect,
        "h_diff_norm": h_diff_norm,
        "anchoring_strength": h_diff_norm / (h_natural_last.norm() + 1e-10),
    }


def experiment_d_signal_amplification(model, tokenizer, case: TestCase) -> Dict:
    """实验D: 多步信号放大——追踪消歧方向在生成过程中的演化"""
    print(f"\n  D. 多步信号放大 [{case.capability}]")
    
    correct_ids = tokenizer.encode(case.correct, add_special_tokens=False)
    incorrect_ids = tokenizer.encode(case.incorrect, add_special_tokens=False)
    correct_id = correct_ids[0] if correct_ids else 0
    incorrect_id = incorrect_ids[0] if incorrect_ids else 0
    
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(case.prompt, return_tensors="pt").to(device)
    
    # 获取初始hidden state差异方向
    # 通过在prompt后面分别加correct和incorrect来获取
    prompt_c = tokenizer.encode(case.prompt + " " + case.correct, return_tensors="pt", 
                                 truncation=True, max_length=64).to(device)
    prompt_i = tokenizer.encode(case.prompt + " " + case.incorrect, return_tensors="pt",
                                 truncation=True, max_length=64).to(device)
    
    with torch.no_grad():
        out_c = model(prompt_c, output_hidden_states=True)
        out_i = model(prompt_i, output_hidden_states=True)
        h_c = out_c.hidden_states[-1][0, -1, :].float().cpu()
        h_i = out_i.hidden_states[-1][0, -1, :].float().cpu()
    
    d_ref = h_c - h_i
    if d_ref.norm() > 1e-6:
        d_ref_unit = d_ref / d_ref.norm()
    else:
        d_ref_unit = d_ref
    
    # 现在逐步生成，追踪d_ref_unit在每步hidden state中的投影
    print(f"    {'Step':>4} {'Token':>10} {'Proj_d':>10} {'||h||':>8} {'Margin':>8} {'d/||h||':>8}")
    print(f"    {'-'*60}")
    
    results = {"steps": []}
    current_ids = tokenizer.encode(case.prompt, return_tensors="pt").to(device)
    
    for step in range(min(6, len(tokenizer.encode(case.correct + " " + case.continuation, add_special_tokens=False)) + 2)):
        with torch.no_grad():
            outputs = model(current_ids, output_hidden_states=True)
            logits = outputs.logits[0, -1, :].float().cpu()
            h = outputs.hidden_states[-1][0, -1, :].float().cpu()
        
        proj_d = float(torch.dot(h, d_ref_unit))
        h_norm = float(h.norm())
        margin = float(logits[correct_id] - logits[incorrect_id])
        
        top_id = int(torch.argmax(logits))
        top_token = tokenizer.decode([top_id]).strip()
        
        results["steps"].append({
            "step": step,
            "token": top_token,
            "proj_d": proj_d,
            "h_norm": h_norm,
            "margin": margin,
            "proj_ratio": proj_d / (h_norm + 1e-10),
        })
        
        print(f"    {step:>4} {top_token:>10} {proj_d:>10.4f} {h_norm:>8.2f} {margin:>8.2f} {proj_d/(h_norm+1e-10):>8.4f}")
        
        current_ids = torch.cat([current_ids, torch.tensor([[top_id]], device=device)], dim=1)
        torch.cuda.empty_cache()
    
    # 分析放大效应
    if len(results["steps"]) >= 2:
        first_proj = abs(results["steps"][0]["proj_d"])
        max_proj = max(abs(s["proj_d"]) for s in results["steps"])
        amplification = max_proj / (first_proj + 1e-10)
        
        first_margin = results["steps"][0]["margin"]
        max_margin = max(s["margin"] for s in results["steps"])
        margin_amp = max_margin / (abs(first_margin) + 1e-10)
        
        results["amplification"] = amplification
        results["margin_amplification"] = margin_amp
        
        print(f"\n    信号放大: {amplification:.2f}x (|proj_d|)")
        print(f"    Margin放大: {margin_amp:.2f}x")
    
    return results


def run_model(model_name: str):
    """运行单个模型的完整分析"""
    print(f"\n{'#'*60}")
    print(f"# P27 多步生成过程分析: {model_name}")
    print(f"{'#'*60}")
    
    model, tokenizer = load_model_bundle(model_name)
    if model is None:
        print(f"  错误: 无法加载模型 {model_name}")
        return None
    
    try:
        # 获取unembed矩阵
        unembed = get_unembed_matrix(model, next(model.parameters()).shape[-1])
        if unembed is not None:
            print(f"\n  Unembed矩阵: shape={unembed.shape}")
        
        all_results = {}
        
        for case in CASES:
            print(f"\n{'='*60}")
            print(f"  能力: {case.capability}")
            print(f"{'='*60}")
            
            case_result = {}
            
            # A: 多步margin追踪
            case_result["A"] = experiment_a_multistep_margin(model, tokenizer, case)
            
            # B: 隐藏状态投影
            case_result["B"] = experiment_b_hidden_projection(model, tokenizer, case, unembed)
            
            # C: 首token锚定
            case_result["C"] = experiment_c_first_token_anchoring(model, tokenizer, case)
            
            # D: 信号放大
            case_result["D"] = experiment_d_signal_amplification(model, tokenizer, case)
            
            all_results[case.capability] = {k: v for k, v in case_result.items() if isinstance(v, dict)}
        
        # 汇总
        print(f"\n{'='*60}")
        print(f"  P27 汇总: {model_name}")
        print(f"{'='*60}")
        
        for cap in all_results:
            r = all_results[cap]
            
            # A: margin趋势
            if r.get("A") and r["A"].get("steps"):
                steps_a = r["A"]["steps"]
                if len(steps_a) > 1:
                    print(f"  {cap:>8} A: margin趋势={r['A'].get('margin_trend', '?')}, "
                          f"比={r['A'].get('margin_ratio', 0):.2f}x")
            
            # B: logit方程比
            if r.get("B") and "ratio" in r["B"]:
                print(f"  {cap:>8} B: logit方程比={r['B']['ratio']:.2f}x, "
                      f"top20占比={r['B']['top20_contribution_ratio']:.1%}")
            
            # C: 锚定强度
            if r.get("C") and "anchoring_strength" in r["C"]:
                print(f"  {cap:>8} C: 锚定强度={r['C']['anchoring_strength']:.4f}, "
                          f"自然margin={r['C']['natural_margin']:.3f}")
            
            # D: 放大效应
            if r.get("D") and "amplification" in r["D"]:
                print(f"  {cap:>8} D: 信号放大={r['D']['amplification']:.2f}x, "
                          f"margin放大={r['D'].get('margin_amplification', 0):.2f}x")
        
        # 保存
        save_data = {}
        for cap, r in all_results.items():
            save_data[cap] = {}
            for exp_key in ["A", "B", "C", "D"]:
                if exp_key in r and isinstance(r[exp_key], dict):
                    save_data[cap][exp_key] = {k: v for k, v in r[exp_key].items() 
                                               if isinstance(v, (int, float, str, bool))}
        
        output_path = OUTPUT_DIR / f"stage673_{model_name}_{TIMESTAMP}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"\n  结果已保存: {output_path}")
        
        return save_data
    
    finally:
        free_model(model)


def main():
    if len(sys.argv) < 2:
        print("用法: python stage673_multistep_generation.py <model_name>")
        print("  model_name: qwen3, deepseek7b, glm4, gemma4")
        return
    
    model_name = sys.argv[1]
    run_model(model_name)


if __name__ == "__main__":
    main()
