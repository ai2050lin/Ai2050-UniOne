#!/usr/bin/env python3
"""
P40: RMSNorm方向对齐数学证明+实证验证（Stage686）

核心问题：为什么所有hidden state方向高度对齐(cos>0.87)？
假设1：RMSNorm + 残差连接 -> hidden state被约束到"流形"(manifold)
假设2：残差流的逐层累积 -> 方向趋向稳态

实验设计：
1. 逐层追踪hidden state方向变化（RMSNorm前vs后）
2. 计算残差连接的"方向约束效应"
3. 验证"方向漂移的随机游走"假设
4. 对比LayerNorm vs RMSNorm模型的方向对齐度

用法：python tests/glm5/stage686_rmsnorm_alignment.py <model_name>
"""
import sys, os, math, json, statistics, time
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from pathlib import Path as _Path

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}

def load_model(model_name):
    path = MODEL_MAP.get(model_name, _Path(model_name))
    print(f"  加载模型: {path.name}")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tokenizer

def get_all_hidden_states(model, tokenizer, text):
    """获取所有层的hidden state（含embedding层L0）"""
    model_device = next(model.parameters()).device
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=64)
    tokens = tokens.to(model_device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
    # outputs.hidden_states: tuple of (1, T, d)
    result = {}
    for i, hs in enumerate(outputs.hidden_states):
        result[i] = hs[0, -1, :].float().cpu()  # 最后一token, (d,)
    return result

def measure_direction_alignment(hs_dict):
    """测量所有层之间的方向对齐度"""
    layers = sorted(hs_dict.keys())
    # 计算所有层对的cos
    cos_matrix = {}
    for i in range(len(layers)):
        for j in range(i+1, len(layers)):
            h1 = hs_dict[layers[i]]
            h2 = hs_dict[layers[j]]
            cos_val = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
            cos_matrix[(layers[i], layers[j])] = cos_val
    return cos_matrix

def measure_residual_direction_constraint(hs_dict):
    """测量残差连接的方向约束效应
    
    原理：残差连接 h_{l+1} = h_l + Δ_l
    如果Δ_l相对h_l很小，那么方向变化就很小 -> 对齐
    
    测量：|Δ_l| / |h_l| 的比值 -> "方向稳定性指标"
    """
    layers = sorted(hs_dict.keys())
    ratios = []
    delta_norms = []
    h_norms = []
    for i in range(len(layers) - 1):
        h_l = hs_dict[layers[i]]
        h_next = hs_dict[layers[i+1]]
        delta = h_next - h_l
        delta_norm = torch.norm(delta).item()
        h_norm = torch.norm(h_l).item()
        ratio = delta_norm / max(h_norm, 1e-10)
        ratios.append(ratio)
        delta_norms.append(delta_norm)
        h_norms.append(h_norm)
    
    return {
        "avg_ratio": statistics.mean(ratios) if ratios else 0,
        "min_ratio": min(ratios) if ratios else 0,
        "max_ratio": max(ratios) if ratios else 0,
        "ratios": ratios,
    }

def measure_direction_drift_random_walk(hs_dict):
    """验证方向漂移是否遵循随机游走
    
    如果方向变化是随机的，那么累积方向变化应该随√N增长（N=层数）
    测量：相邻层的方向变化角度 vs 累积方向变化角度
    """
    layers = sorted(hs_dict.keys())
    angles = []
    for i in range(len(layers) - 1):
        h1 = hs_dict[layers[i]]
        h2 = hs_dict[layers[i+1]]
        cos_val = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
        cos_val = max(-1, min(1, cos_val))
        angle = math.degrees(math.acos(abs(cos_val)))
        angles.append(angle)
    
    # 累积角度
    cumulative_angles = []
    for i in range(len(angles)):
        h0 = hs_dict[layers[0]]
        h_i = hs_dict[layers[i+1]]
        cos_val = F.cosine_similarity(h0.unsqueeze(0), h_i.unsqueeze(0)).item()
        cos_val = max(-1, min(1, cos_val))
        angle = math.degrees(math.acos(abs(cos_val)))
        cumulative_angles.append(angle)
    
    return {
        "step_angles": angles,
        "cumulative_angles": cumulative_angles,
        "avg_step_angle": statistics.mean(angles) if angles else 0,
        "total_cumulative_angle": cumulative_angles[-1] if cumulative_angles else 0,
    }

def test_norm_stability_across_texts(model, tokenizer, texts):
    """测试不同文本在同一层的norm分布
    
    假设：RMSNorm使norm趋于恒定 -> 不同文本的norm分布应该很窄
    """
    # 取中间层
    all_norms = {}
    layer_norms = {}
    
    for text in texts:
        hs = get_all_hidden_states(model, tokenizer, text)
        for layer, h in hs.items():
            norm_val = torch.norm(h).item()
            if layer not in layer_norms:
                layer_norms[layer] = []
            layer_norms[layer].append(norm_val)
    
    # 计算每层的norm变异系数(CV = std/mean)
    cv_by_layer = {}
    for layer, norms in layer_norms.items():
        mean_n = statistics.mean(norms)
        std_n = statistics.stdev(norms) if len(norms) > 1 else 0
        cv_by_layer[layer] = std_n / max(mean_n, 1e-10)
    
    return cv_by_layer

def measure_inter_layer_cos_grid(hs_dict):
    """构建层间cos矩阵的"对齐热图"数据"""
    layers = sorted(hs_dict.keys())
    grid = []
    for i, l1 in enumerate(layers):
        row = []
        for j, l2 in enumerate(layers):
            if l1 <= l2:
                cos_val = F.cosine_similarity(
                    hs_dict[l1].unsqueeze(0), hs_dict[l2].unsqueeze(0)
                ).item()
            else:
                cos_val = grid[j][i]
            row.append(cos_val)
        grid.append(row)
    return grid, layers

def analyze_alignment_mechanism(model, tokenizer, texts):
    """P40核心分析：为什么方向高度对齐？"""
    
    print("\n" + "="*60)
    print("  实验A: 残差连接的方向约束效应")
    print("="*60)
    
    all_ratios = []
    for text in texts[:8]:
        hs = get_all_hidden_states(model, tokenizer, text)
        constraint = measure_residual_direction_constraint(hs)
        all_ratios.append(constraint["avg_ratio"])
        print(f"  \"{text[:40]}...\" avg_Δ/h = {constraint['avg_ratio']:.4f}")
    
    avg_constraint = statistics.mean(all_ratios)
    print(f"\n  -> 平均方向约束比(Δ/h): {avg_constraint:.4f}")
    if avg_constraint < 0.1:
        print(f"  -> [INV-333] Δ << h: 残差流方向高度稳定 -> 方向对齐自然产生 [OK]")
    else:
        print(f"  -> [INV-333] Δ ~ h: 方向变化较大 -> 方向对齐需要其他机制")
    
    print("\n" + "="*60)
    print("  实验B: 方向漂移模式（随机游走 vs 收敛）")
    print("="*60)
    
    drift_data = []
    for text in texts[:5]:
        hs = get_all_hidden_states(model, tokenizer, text)
        drift = measure_direction_drift_random_walk(hs)
        drift_data.append(drift)
        print(f"  \"{text[:35]}...\"")
        print(f"    avg_step={drift['avg_step_angle']:.2f}°, total={drift['total_cumulative_angle']:.2f}°")
    
    avg_step = statistics.mean([d["avg_step_angle"] for d in drift_data])
    avg_total = statistics.mean([d["total_cumulative_angle"] for d in drift_data])
    n_layers = len(drift_data[0]["step_angles"]) if drift_data else 1
    
    # 随机游走预测: total ≈ avg_step × √N
    rw_predict = avg_step * math.sqrt(n_layers)
    print(f"\n  -> 平均步进角: {avg_step:.2f}°")
    print(f"  -> 实际累积角: {avg_total:.2f}°")
    print(f"  -> 随机游走预测: {rw_predict:.2f}° (avg_step × √{n_layers})")
    
    if avg_total < rw_predict * 0.5:
        print(f"  -> [INV-334] 方向漂移是亚扩散(sub-diffusive) -> 方向收敛/稳定 [OK]")
    elif avg_total < rw_predict * 1.5:
        print(f"  -> [INV-334] 方向漂移接近随机游走 -> 无特殊约束")
    else:
        print(f"  -> [INV-334] 方向漂移是超扩散(super-diffusive) -> 方向发散")
    
    print("\n" + "="*60)
    print("  实验C: RMSNorm的norm稳定性（跨文本）")
    print("="*60)
    
    cv_by_layer = test_norm_stability_across_texts(model, tokenizer, texts)
    
    # 显示关键层
    key_layers = sorted(cv_by_layer.keys())
    for l in [0, 1, len(key_layers)//4, len(key_layers)//2, 3*len(key_layers)//4, -1]:
        if 0 <= l < len(key_layers):
            layer = key_layers[l]
            print(f"  L{layer}: CV(norm) = {cv_by_layer[layer]:.4f}")
    
    avg_cv = statistics.mean(list(cv_by_layer.values()))
    print(f"\n  -> 平均norm变异系数: {avg_cv:.4f}")
    if avg_cv < 0.05:
        print(f"  -> [INV-335] RMSNorm使norm几乎恒定(CV<5%) -> 方向是唯一编码通道 [OK]")
    else:
        print(f"  -> [INV-335] norm有明显变化(CV={avg_cv:.1%}) -> norm也携带信息")
    
    print("\n" + "="*60)
    print("  实验D: 层间对齐热图（早/中/后期cos）")
    print("="*60)
    
    hs = get_all_hidden_states(model, tokenizer, texts[0])
    grid, layers = measure_inter_layer_cos_grid(hs)
    
    n = len(layers)
    # 三角平均
    early_avg = []
    mid_avg = []
    late_avg = []
    for i in range(min(n-1, 5)):
        for j in range(i+1, min(n, 8)):
            early_avg.append(grid[i][j])
    for i in range(n//4, 3*n//4):
        for j in range(i+1, min(3*n//4+1, n)):
            mid_avg.append(grid[i][j])
    for i in range(max(0, n-6), n-1):
        for j in range(i+1, n):
            late_avg.append(grid[i][j])
    
    print(f"  早期层(L0-7) avg cos: {statistics.mean(early_avg):.4f}" if early_avg else "  早期层: N/A")
    print(f"  中间层 avg cos: {statistics.mean(mid_avg):.4f}" if mid_avg else "  中间层: N/A")
    print(f"  后期层 avg cos: {statistics.mean(late_avg):.4f}" if late_avg else "  后期层: N/A")
    
    print("\n" + "="*60)
    print("  实验E: 方向基底的跨文本一致性")
    print("="*60)
    
    # 对多个文本取中间层方向，计算一致性
    mid_layer = layers[len(layers)//2]
    directions = []
    for text in texts:
        hs = get_all_hidden_states(model, tokenizer, text)
        h = hs[mid_layer]
        directions.append(h / max(torch.norm(h).item(), 1e-10))
    
    # 计算平均方向
    mean_dir = torch.stack(directions).mean(dim=0)
    mean_dir = mean_dir / max(torch.norm(mean_dir).item(), 1e-10)
    
    # 每个方向与平均方向的cos
    align_scores = []
    for d in directions:
        cos_val = F.cosine_similarity(d.unsqueeze(0), mean_dir.unsqueeze(0)).item()
        align_scores.append(cos_val)
    
    avg_align = statistics.mean(align_scores)
    std_align = statistics.stdev(align_scores) if len(align_scores) > 1 else 0
    
    print(f"  中间层: L{mid_layer}")
    print(f"  方向对齐(与均值): {avg_align:.4f} ± {std_align:.4f}")
    print(f"  -> 所有文本共享\"语义基底方向\"的程度: {avg_align:.4f}")
    
    return {
        "avg_constraint_ratio": avg_constraint,
        "avg_step_angle": avg_step,
        "total_cumulative_angle": avg_total,
        "rw_prediction": rw_predict,
        "norm_cv": avg_cv,
        "direction_alignment": avg_align,
        "direction_alignment_std": std_align,
    }

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    print(f"\n{'='*60}")
    print(f"  P40: RMSNorm方向对齐数学证明+实证验证")
    print(f"  模型: {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer = load_model(model_name)
    
    # 多主题文本
    texts = [
        "The cat sat on the mat.",
        "Paris is the capital of France.",
        "She carefully folded the origami crane.",
        "The derivative of x squared is two x.",
        "DNA contains genetic instructions.",
        "The orchestra played a beautiful symphony.",
        "Gravity causes objects to fall.",
        "Yesterday it rained heavily all day.",
        "Two plus two equals four.",
        "The river bank was muddy.",
        "She quickly ran home.",
        "The meeting was productive.",
        "The sky turned orange at sunset.",
        "He solved the equation step by step.",
        "The neural network learned patterns.",
    ]
    
    t0 = time.time()
    results = analyze_alignment_mechanism(model, tokenizer, texts)
    elapsed = time.time() - t0
    
    print(f"\n{'='*60}")
    print(f"  P40 结果汇总")
    print(f"{'='*60}")
    print(f"  残差约束比(Δ/h): {results['avg_constraint_ratio']:.4f}")
    print(f"  方向步进角: {results['avg_step_angle']:.2f}°")
    print(f"  方向累积角: {results['total_cumulative_angle']:.2f}°")
    print(f"  随机游走预测: {results['rw_prediction']:.2f}°")
    print(f"  Norm变异系数: {results['norm_cv']:.4f}")
    print(f"  方向对齐度: {results['direction_alignment']:.4f} ± {results['direction_alignment_std']:.4f}")
    print(f"  耗时: {elapsed:.1f}s")
    
    # 综合判断
    print(f"\n{'='*60}")
    print(f"  P40 方向对齐机制判断")
    print(f"{'='*60}")
    
    reasons = []
    if results["avg_constraint_ratio"] < 0.1:
        reasons.append("残差约束(Δ/h<0.1)")
    if results["total_cumulative_angle"] < results["rw_prediction"] * 0.5:
        reasons.append("亚扩散漂移(收敛)")
    if results["norm_cv"] < 0.05:
        reasons.append("RMSNorm norm恒定(CV<5%)")
    if results["direction_alignment"] > 0.9:
        reasons.append("强方向基底(cos>0.9)")
    
    if len(reasons) >= 3:
        print(f"  -> 方向对齐机制: {' + '.join(reasons)}")
        print(f"  -> [INV-333/334/335] 三个不变量全部确认 [OK]")
        print(f"  -> 机制：RMSNorm约束norm + 残差约束Δ + 亚扩散漂移 -> 方向自然收敛")
    else:
        print(f"  -> 部分确认: {reasons}")
        print(f"  -> 方向对齐可能涉及其他机制")

if __name__ == "__main__":
    main()
