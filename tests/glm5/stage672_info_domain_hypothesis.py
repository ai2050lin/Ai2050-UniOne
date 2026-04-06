#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage672: P26 信息域假说验证 (INV-308)

核心假说：编码策略(SEPARATED vs DENSE)由模型的信息域数量决定
  - 单信息域(纯语言) → SEPARATED: 抵消率30-90%, 信号效率>50%, PCA90<20
  - 多信息域(语言+视觉+听觉) → DENSE: 抵消率>90%, 信号效率低, PCA90高

验证方法：
  1. 架构分析：统计每个模型的norm类型/数量/位置，判断信息域
  2. 编码维度分析：不同能力类型的PCA90维度对比
  3. 隐藏状态能量分布：检测是否存在"视觉专用"和"语言专用"维度
  4. 跨能力vs跨域干扰对比
  5. 归一化密度vs编码效率的定量关系
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
import torch.nn as nn
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    discover_layers,
    free_model,
    load_model_bundle,
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

# 测试用例：覆盖多种能力类型
@dataclass(frozen=True)
class TestCase:
    capability: str
    prompt_a: str
    prompt_b: str
    label: str

CASES = [
    TestCase("disamb", "The river bank was muddy.", "The bank approved the loan.", "消歧"),
    TestCase("syntax", "She quickly ran home.", "Home she ran quickly.", "语法"),
    TestCase("relation", "Paris is the capital of France.", "Berlin is the capital of Germany.", "关系"),
    TestCase("style", "The meeting was extremely productive.", "That get-together was quite fruitful.", "风格"),
    TestCase("spatial", "The cat is under the table.", "The bird is above the tree.", "空间"),
    TestCase("temporal", "Yesterday it rained heavily.", "Tomorrow it will snow.", "时序"),
]


def analyze_model_architecture(model, model_name: str) -> Dict:
    """深度分析模型架构：norm类型/数量/位置 + 多模态检测"""
    print(f"\n{'='*60}")
    print(f"  模型架构分析: {model_name}")
    print(f"{'='*60}")
    
    result = {
        "model_name": model_name,
        "total_params": sum(p.numel() for p in model.parameters()),
        "hidden_dim": None,
        "num_layers": 0,
        "norm_modules": [],
        "norm_types": {},
        "norms_per_layer": [],
        "total_norms": 0,
        "has_vision_encoder": False,
        "has_audio_encoder": False,
        "multimodal_indicators": [],
        "information_domains": ["language"],
    }
    
    # 检测hidden_dim
    for name, param in model.named_parameters():
        if 'embed' in name.lower() and param.dim() == 2:
            result["hidden_dim"] = param.shape[-1]
            break
    if result["hidden_dim"] is None:
        for name, param in model.named_parameters():
            if param.dim() == 2 and param.shape[0] > 100:
                result["hidden_dim"] = param.shape[-1]
                break
    
    # 检测层数
    layers = discover_layers(model)
    result["num_layers"] = len(layers)
    
    # 全面分析norm模块
    norm_info = []
    for name, module in model.named_modules():
        mod_type = type(module).__name__
        if 'norm' in name.lower() or 'norm' in mod_type.lower() or 'layernorm' in mod_type.lower() or 'rmsnorm' in mod_type.lower():
            norm_info.append({
                "name": name,
                "type": mod_type,
                "params": sum(p.numel() for p in module.parameters()),
            })
            result["norm_types"][mod_type] = result["norm_types"].get(mod_type, 0) + 1
    
    result["norm_modules"] = norm_info
    result["total_norms"] = len(norm_info)
    
    # 计算每层norm数量
    if layers:
        for li, layer in enumerate(layers):
            layer_norm_count = 0
            for name, module in layer.named_modules():
                mod_type = type(module).__name__
                if 'norm' in name.lower() or 'norm' in mod_type.lower() or 'layernorm' in mod_type.lower() or 'rmsnorm' in mod_type.lower():
                    layer_norm_count += 1
            result["norms_per_layer"].append(layer_norm_count)
        
        avg_norms = statistics.mean(result["norms_per_layer"]) if result["norms_per_layer"] else 0
        result["avg_norms_per_layer"] = avg_norms
    
    # 多模态检测
    all_names = [n for n, _ in model.named_modules()]
    all_names_str = " ".join(all_names).lower()
    
    multimodal_keywords = {
        "vision": ["vision", "visual", "image", "pixel", "patch", "vit", "siglip", "clip_encoder"],
        "audio": ["audio", "speech", "whisper", "mel", "spectrogram", "sound"],
        "multimodal": ["mm", "multi_modal", "multimodal", "cross_modal", "fusion"],
    }
    
    for domain, keywords in multimodal_keywords.items():
        found = [kw for kw in keywords if kw in all_names_str]
        if found:
            result["multimodal_indicators"].append(f"{domain}: {found}")
            if domain == "vision":
                result["has_vision_encoder"] = True
                result["information_domains"].append("vision")
            elif domain == "audio":
                result["has_audio_encoder"] = True
                result["information_domains"].append("audio")
    
    # 输出
    print(f"\n  基本信息:")
    print(f"    参数量: {result['total_params']:,}")
    print(f"    隐藏维度: {result['hidden_dim']}")
    print(f"    层数: {result['num_layers']}")
    print(f"    信息域: {result['information_domains']}")
    
    print(f"\n  归一化分析:")
    print(f"    总Norm数: {result['total_norms']}")
    print(f"    平均每层: {result.get('avg_norms_per_layer', 'N/A'):.1f}" if isinstance(result.get('avg_norms_per_layer'), (int, float)) else f"    平均每层: N/A")
    
    print(f"\n  Norm类型分布:")
    for ntype, count in sorted(result["norm_types"].items(), key=lambda x: -x[1]):
        print(f"    {ntype}: {count}")
    
    if result["multimodal_indicators"]:
        print(f"\n  多模态指标:")
        for indicator in result["multimodal_indicators"]:
            print(f"    {indicator}")
    else:
        print(f"\n  多模态指标: 未检测到（纯语言模型）")
    
    # 信息域判定
    n_domains = len(result["information_domains"])
    if n_domains == 1:
        result["predicted_strategy"] = "SEPARATED"
        result["predicted_cancellation"] = "30-90%"
        result["predicted_efficiency"] = ">50%"
        print(f"\n  → 预测编码策略: SEPARATED (单信息域)")
    else:
        result["predicted_strategy"] = "DENSE"
        result["predicted_cancellation"] = ">90%"
        result["predicted_efficiency"] = "<30%"
        print(f"\n  → 预测编码策略: DENSE (多信息域: {', '.join(result['information_domains'])})")
    
    return result


def measure_encoding_dimensions(model, tokenizer, cases: List[TestCase]) -> Dict:
    """测量各能力类型的编码维度（PCA90）"""
    print(f"\n  编码维度分析:")
    
    layers = discover_layers(model)
    if not layers:
        return {}
    
    results = {}
    
    for case in cases:
        cap = case.capability
        
        # 提取最后一层的hidden state差异
        inputs_a = tokenizer(case.prompt_a, return_tensors="pt", truncation=True, max_length=64)
        inputs_b = tokenizer(case.prompt_b, return_tensors="pt", truncation=True, max_length=64)
        
        device = next(model.parameters()).device
        
        # 简化：只提取最后一层
        with torch.no_grad():
            try:
                out_a = model(**{k: v.to(device) for k, v in inputs_a.items()}, output_hidden_states=True)
                out_b = model(**{k: v.to(device) for k, v in inputs_b.items()}, output_hidden_states=True)
                
                # 取最后一个token的最后一层hidden state
                ha = out_a.hidden_states[-1][0, -1, :].cpu().float()
                hb = out_b.hidden_states[-1][0, -1, :].cpu().float()
                
                d = ha - hb
                
                # 有效维度分析：对差异向量计算"能量集中度"
                # 使用 p-norm ratio 方法估计有效维度
                d_abs = torch.abs(d)
                sorted_vals, _ = torch.sort(d_abs, descending=True)
                total_energy = float(torch.sum(d ** 2))
                
                if total_energy > 1e-10:
                    # 方法1: 累积能量 → PCA等价
                    cumulative_energy = torch.cumsum(sorted_vals ** 2, dim=0) / total_energy
                    pca90_idx = int((cumulative_energy < 0.9).sum().item()) + 1
                    pca99_idx = int((cumulative_energy < 0.99).sum().item()) + 1
                    
                    # 方法2: 参与编码的维度数（值超过均值一定比例的维度）
                    mean_val = float(torch.mean(d_abs))
                    active_dims = int((d_abs > mean_val * 0.1).sum().item())
                    
                    # 方法3: Rényi熵估计有效维度
                    p = sorted_vals ** 2 / total_energy
                    p = p[p > 1e-10]  # 去掉零
                    entropy = -float(torch.sum(p * torch.log(p + 1e-10)))
                    eff_dim = float(torch.exp(torch.tensor(entropy)))
                else:
                    pca90_idx = pca99_idx = active_dims = 0
                    eff_dim = 0
                    sorted_vals = d_abs
                
                results[cap] = {
                    "pca90": pca90_idx,
                    "pca99": pca99_idx,
                    "active_dims": active_dims,
                    "eff_dim": eff_dim,
                    "norm": float(d.norm()),
                    "max_val": float(sorted_vals[0]),
                    "top1_ratio": float(sorted_vals[0] ** 2 / total_energy) if total_energy > 1e-10 else 0,
                }
                
                print(f"    {cap:>8}: PCA90={pca90_idx:>4}, EffDim={eff_dim:>6.1f}, Active={active_dims:>4}, "
                      f"||d||={d.norm():.2f}, top1={results[cap]['top1_ratio']:.1%}")
                
            except Exception as e:
                print(f"    {cap:>8}: ERROR - {str(e)[:60]}")
        
        torch.cuda.empty_cache()
    
    return results


def measure_hidden_state_energy_distribution(model, tokenizer, cases: List[TestCase]) -> Dict:
    """分析hidden state的能量分布：检测是否存在'能力专用'维度"""
    print(f"\n  隐藏状态能量分布分析:")
    
    results = {}
    directions = {}
    
    for case in cases:
        cap = case.capability
        inputs_a = tokenizer(case.prompt_a, return_tensors="pt", truncation=True, max_length=64)
        inputs_b = tokenizer(case.prompt_b, return_tensors="pt", truncation=True, max_length=64)
        device = next(model.parameters()).device
        
        with torch.no_grad():
            try:
                out_a = model(**{k: v.to(device) for k, v in inputs_a.items()}, output_hidden_states=True)
                out_b = model(**{k: v.to(device) for k, v in inputs_b.items()}, output_hidden_states=True)
                
                ha = out_a.hidden_states[-1][0, -1, :].cpu().float()
                hb = out_b.hidden_states[-1][0, -1, :].cpu().float()
                
                d = ha - hb
                if d.norm() > 1e-6:
                    directions[cap] = d / d.norm()
                else:
                    directions[cap] = d
                    
            except Exception as e:
                pass
        
        torch.cuda.empty_cache()
    
    if len(directions) < 2:
        return results
    
    # 计算方向之间的cos similarity矩阵
    cap_names = sorted(directions.keys())
    cos_matrix = {}
    
    print(f"\n    方向cos similarity矩阵:")
    print(f"    {'':>8}", end="")
    for c in cap_names:
        print(f"  {c[:6]:>6}", end="")
    print()
    
    for ci, c1 in enumerate(cap_names):
        row = []
        for c2 in cap_names:
            cos_val = float(torch.dot(directions[c1], directions[c2]))
            cos_matrix[(c1, c2)] = cos_val
            row.append(cos_val)
        
        print(f"    {c1[:6]:>8}", end="")
        for v in row:
            marker = "★" if abs(v) > 0.5 else " "
            print(f" {marker}{v:>5.3f}", end="")
        print()
    
    # 分析能量集中度
    print(f"\n    能量集中度分析:")
    for cap in cap_names:
        if cap not in directions:
            continue
        d = directions[cap]
        d_abs = torch.abs(d)
        sorted_vals, _ = torch.sort(d_abs, descending=True)
        
        total_energy = torch.sum(d_abs ** 2).item()
        if total_energy > 1e-10:
            # top-1%维度承载多少能量
            top1_pct_count = max(1, len(d) // 100)
            top1_pct_energy = torch.sum(sorted_vals[:top1_pct_count] ** 2).item() / total_energy
            
            # top-10%维度
            top10_pct_count = max(1, len(d) // 10)
            top10_pct_energy = torch.sum(sorted_vals[:top10_pct_count] ** 2).item() / total_energy
            
            print(f"    {cap:>8}: top1%能量={top1_pct_energy:.1%}, top10%能量={top10_pct_energy:.1%}, "
                  f"活跃维度(d>{sorted_vals[max(0,len(sorted_vals)//10)]:.4f})=Top-10%")
            results[cap] = {
                "top1pct_energy": top1_pct_energy,
                "top10pct_energy": top10_pct_energy,
            }
    
    # 离群值检测：是否存在与所有其他方向高度对齐的能力
    print(f"\n    离群值检测（某能力与其他所有能力的平均|cos|）:")
    for c1 in cap_names:
        avg_cos = statistics.mean([abs(cos_matrix[(c1, c2)]) for c2 in cap_names if c2 != c1])
        print(f"    {c1:>8}: avg|cos|={avg_cos:.4f} {'← 异常高!' if avg_cos > 0.3 else ''}")
    
    return results


def measure_norm_density_vs_efficiency(arch_result: Dict, dim_results: Dict) -> Dict:
    """分析归一化密度vs编码效率的定量关系"""
    print(f"\n  归一化密度 vs 编码效率 关系:")
    
    norms_per_layer = arch_result.get("avg_norms_per_layer", 0)
    total_norms = arch_result.get("total_norms", 0)
    n_domains = len(arch_result.get("information_domains", ["language"]))
    
    if dim_results:
        avg_pca90 = statistics.mean([v["pca90"] for v in dim_results.values()])
        avg_eff_dim = statistics.mean([v.get("eff_dim", v["pca90"]) for v in dim_results.values()])
        avg_top1 = statistics.mean([v.get("top1_ratio", 0) for v in dim_results.values()])
        avg_norm = statistics.mean([v["norm"] for v in dim_results.values()])
        
        efficiency = avg_norm / max(avg_eff_dim, 1) if avg_eff_dim > 0 else 0
        
        print(f"    Norm密度(每层): {norms_per_layer:.1f}")
        print(f"    总Norm数: {total_norms}")
        print(f"    信息域数: {n_domains}")
        print(f"    平均EffDim: {avg_eff_dim:.1f}")
        print(f"    平均PCA90: {avg_pca90:.0f}")
        print(f"    平均top1能量比: {avg_top1:.1%}")
        print(f"    编码效率(||d||/EffDim): {efficiency:.4f}")
        
        return {
            "norms_per_layer": norms_per_layer,
            "total_norms": total_norms,
            "n_domains": n_domains,
            "avg_eff_dim": avg_eff_dim,
            "avg_pca90": avg_pca90,
            "avg_top1_energy": avg_top1,
            "efficiency": efficiency,
        }
    
    return {}


def run_model(model_name: str):
    """运行单个模型的完整分析"""
    print(f"\n{'#'*60}")
    print(f"# P26 信息域假说验证: {model_name}")
    print(f"{'#'*60}")
    
    model, tokenizer = load_model_bundle(model_name)
    if model is None:
        print(f"  错误: 无法加载模型 {model_name}")
        return None
    
    try:
        # 实验1: 架构分析
        arch = analyze_model_architecture(model, model_name)
        
        # 实验2: 编码维度分析
        dims = measure_encoding_dimensions(model, tokenizer, CASES)
        
        # 实验3: 能量分布分析
        energy = measure_hidden_state_energy_distribution(model, tokenizer, CASES)
        
        # 实验4: Norm密度vs效率
        relation = measure_norm_density_vs_efficiency(arch, dims)
        
        # 汇总判定
        print(f"\n{'='*60}")
        print(f"  INV-308 判定")
        print(f"{'='*60}")
        
        n_domains = len(arch["information_domains"])
        predicted_strategy = arch["predicted_strategy"]
        
        if dims:
            avg_eff_dim = statistics.mean([v.get("eff_dim", v["pca90"]) for v in dims.values()])
            avg_top1 = statistics.mean([v.get("top1_ratio", 0) for v in dims.values()])
            avg_pca90 = statistics.mean([v["pca90"] for v in dims.values()])
            
            # 有效维度阈值：SEPARATED < 100, DENSE > 100
            actual_separated = avg_eff_dim < 100 and avg_top1 > 0.1
            actual_dense = avg_eff_dim > 100
            
            if n_domains == 1 and actual_separated:
                verdict = f"✅ INV-308确认: 单域→SEPARATED (EffDim={avg_eff_dim:.1f}, PCA90={avg_pca90:.0f}, top1={avg_top1:.1%})"
            elif n_domains == 1 and not actual_separated:
                verdict = f"⚠️ INV-308部分: 单域但EffDim={avg_eff_dim:.1f}偏高"
            elif n_domains > 1 and actual_dense:
                verdict = f"✅ INV-308确认: 多域→DENSE (EffDim={avg_eff_dim:.1f})"
            elif n_domains > 1 and not actual_dense:
                verdict = f"⚠️ INV-308部分: 多域但EffDim={avg_eff_dim:.1f}不高"
            else:
                verdict = f"❓ INV-308不确定"
            
            print(f"  {verdict}")
        else:
            print(f"  ❓ 无法判定（编码维度数据缺失）")
        
        # 保存结果
        output_data = {
            "model": model_name,
            "architecture": arch,
            "dimensions": {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str))} for k, v in dims.items()},
            "norm_efficiency_relation": relation,
        }
        
        output_path = OUTPUT_DIR / f"stage672_{model_name}_{TIMESTAMP}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n  结果已保存: {output_path}")
        
        return output_data
    
    finally:
        free_model(model)


def main():
    if len(sys.argv) < 2:
        print("用法: python stage672_info_domain_hypothesis.py <model_name>")
        print("  model_name: qwen3, deepseek7b, glm4, gemma4")
        return
    
    model_name = sys.argv[1]
    run_model(model_name)


if __name__ == "__main__":
    main()
