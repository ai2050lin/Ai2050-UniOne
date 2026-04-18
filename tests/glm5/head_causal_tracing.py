"""
Phase CLXXXVI: 注意力头级别因果追踪 — 从"1维方向"到"注意力路由"
============================================================

核心问题: Phase CLXXXV发现1维极性方向注入无因果效应(斜率<0.05),
但完整残差修补有效(100-300x). 说明极性信息通过注意力头路由传播,
而不是通过1维方向平移.

测试设计:
H1: Head Ablation — 逐个关闭注意力头, 看"not" logit变化
    → 找到对极性最关键的"极性头"
H2: Head Attribution — 从最终位置追踪哪些头将"not"信息路由过来
    → 使用注意力权重分析信息流
H3: Cross-layer Head Path — 追踪极性信息从"not"位置到最终输出的完整路径
    → 构建极性的"电路图"
H4: 极性头 vs 随机头 — 极性头的因果效应是否显著大于随机头?
    → 统计验证

运行方式:
  python tests/glm5/head_causal_tracing.py --model qwen3 --test h1
  python tests/glm5/head_causal_tracing.py --model qwen3 --test h2
  python tests/glm5/head_causal_tracing.py --model qwen3 --test h3
  python tests/glm5/head_causal_tracing.py --model qwen3 --test all
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import gc
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# ============================================================
# 测试集
# ============================================================

POLARITY_PAIRS = [
    ("The cat is here", "The cat is not here", "not"),
    ("The dog is happy", "The dog is not happy", "not"),
    ("The book was found", "The book was not found", "not"),
    ("I like the car", "I do not like the car", "not"),
    ("She knows the answer", "She does not know the answer", "not"),
    ("The house is big", "The house is not big", "not"),
    ("The river flows north", "The river does not flow north", "not"),
    ("He can swim", "He cannot swim", "not"),
    ("The bird will come", "The bird will not come", "not"),
    ("The door was closed", "The door was not closed", "not"),
    ("The phone is working", "The phone is not working", "not"),
    ("The flower has bloomed", "The flower has not bloomed", "not"),
    ("I understand the plan", "I do not understand the plan", "not"),
    ("She likes the movie", "She does not like the movie", "not"),
    ("The bridge is safe", "The bridge is not safe", "not"),
    ("The child was playing", "The child was not playing", "not"),
    ("The star is visible", "The star is not visible", "not"),
    ("The cloud disappeared", "The cloud did not disappear", "not"),
    ("The key works well", "The key does not work well", "not"),
    ("The table holds weight", "The table does not hold weight", "not"),
    ("The food tastes good", "The food does not taste good", "not"),
    ("The light shines bright", "The light does not shine bright", "not"),
    ("The wind blows cold", "The wind does not blow cold", "not"),
    ("The rain falls down", "The rain does not fall down", "not"),
    ("The sun rises early", "The sun does not rise early", "not"),
]


# ============================================================
# 模型加载
# ============================================================

def _load_model(model_name):
    if model_name in ["glm4", "deepseek7b"]:
        return _load_model_4bit(model_name)
    else:
        from model_utils import load_model
        return load_model(model_name)


def _load_model_4bit(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    MODEL_PATHS = {
        "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }

    path = MODEL_PATHS.get(model_name)
    if not path:
        print(f"  ERROR: Unknown model {model_name}")
        return None, None, None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  Loaded {model_name} (4-bit), device={device}, class={type(model).__name__}")
    return model, tokenizer, device


def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"Cannot find layers in {type(model).__name__}")


def _get_n_heads(model, model_name):
    """获取注意力头数量"""
    layers = _get_layers(model)
    layer0 = layers[0]
    sa = layer0.self_attn
    # Qwen3/GLM4/DS7B 都用 q_proj
    if hasattr(sa, 'q_proj'):
        d_model = sa.q_proj.weight.shape[1]
        # 从config获取n_heads
        if hasattr(model, 'config') and hasattr(model.config, 'num_attention_heads'):
            return model.config.num_attention_heads
        if hasattr(model, 'config') and hasattr(model.config, 'num_heads'):
            return model.config.num_heads
        # 从权重推断
        d_q = sa.q_proj.weight.shape[0]
        # GQA: n_kv_heads可能不同
        if hasattr(sa, 'k_proj'):
            d_k = sa.k_proj.weight.shape[0]
            # 如果d_k < d_q, 说明是GQA
            # 假设head_dim = d_k / n_kv_heads
            # n_heads = d_q / head_dim
        # 简单: 从d_model推断
        return d_q // (d_model // d_q) if d_q != d_model else d_q // 64
    return 32  # default


def _get_head_dim(model):
    """获取每个头的维度"""
    layers = _get_layers(model)
    sa = layers[0].self_attn
    if hasattr(sa, 'q_proj'):
        d_model = sa.q_proj.weight.shape[1]
        d_q = sa.q_proj.weight.shape[0]
        n_heads = _get_n_heads(model, 'qwen3')
        return d_q // n_heads
    return 64


def _get_not_token_id(tokenizer):
    """获取'not'的token ID"""
    test_ids = tokenizer.encode("The cat is not here", add_special_tokens=False)
    for tid in test_ids:
        if tokenizer.decode([tid]).strip().lower() == "not":
            return tid
    # fallback
    ids = tokenizer.encode(" not", add_special_tokens=False)
    return ids[0] if ids else None


def _find_token_position(tokenizer, text, target_token):
    """找到target_token在tokenized序列中的位置"""
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = toks.input_ids[0].tolist()
    positions = []
    for i, tid in enumerate(input_ids):
        decoded = tokenizer.decode([tid]).strip().lower()
        if decoded == target_token.lower():
            positions.append((i, tid))
    return positions


# ============================================================
# 核心工具函数: 注意力头级别操作
# ============================================================

def _get_attn_output_by_head(layer, hidden_states, n_heads, head_dim):
    """
    手动计算注意力输出, 按头分组
    
    Args:
        layer: transformer层对象
        hidden_states: [1, seq_len, d_model]
        n_heads: 头数
        head_dim: 每个头的维度
    
    Returns:
        dict: {head_idx: output[1, seq_len, d_model]} 每个头的输出
    """
    sa = layer.self_attn
    
    # 获取位置信息
    seq_len = hidden_states.shape[1]
    
    # LayerNorm
    if hasattr(layer, 'input_layernorm'):
        normed = layer.input_layernorm(hidden_states)
    elif hasattr(layer, 'ln_1'):
        normed = layer.ln_1(hidden_states)
    else:
        normed = hidden_states
    
    # Q, K, V projections
    Q = sa.q_proj(normed)  # [1, seq_len, d_q]
    K = sa.k_proj(normed)
    V = sa.v_proj(normed)
    
    # 处理GQA: K,V可能比Q少头
    d_q = Q.shape[-1]
    d_k = K.shape[-1]
    d_v = V.shape[-1]
    
    n_kv_heads = d_k // head_dim
    
    # Reshape: [1, seq, n_heads*head_dim] -> [1, n_heads, seq, head_dim]
    Q = Q.view(1, seq_len, n_heads, head_dim).transpose(1, 2)
    K = K.view(1, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(1, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    
    # GQA: 如果n_kv_heads < n_heads, 需要重复K, V
    if n_kv_heads < n_heads:
        n_rep = n_heads // n_kv_heads
        K = K.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(1, n_heads, seq_len, head_dim)
        V = V.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(1, n_heads, seq_len, head_dim)
    
    # 注意力计算
    scale = head_dim ** 0.5
    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [1, n_heads, seq, seq]
    
    # Causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool), diagonal=1)
    attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    attn_weights = torch.softmax(attn_weights.float(), dim=-1).to(hidden_states.dtype)
    
    # 注意力输出
    attn_out = torch.matmul(attn_weights, V)  # [1, n_heads, seq, head_dim]
    
    # 按头分组输出
    head_outputs = {}
    for h_idx in range(n_heads):
        # 每个头的输出: [1, seq, head_dim]
        h_out = attn_out[0, h_idx, :, :]  # [seq, head_dim]
        # 通过o_proj映射回d_model
        # 但o_proj是所有头一起的, 无法直接分开
        # 方法: 构造只有1个头输出的tensor, 其他头置零, 然后过o_proj
        full_out = torch.zeros(1, seq_len, n_heads * head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        full_out[0, :, h_idx * head_dim:(h_idx + 1) * head_dim] = h_out
        # Transpose back: [1, seq, n_heads*head_dim]
        full_out_t = full_out  # 已经是正确形状
        # o_proj
        head_final = sa.o_proj(full_out_t)  # [1, seq, d_model]
        head_outputs[h_idx] = head_final
    
    return head_outputs, attn_weights


def _ablate_head_hook(head_idx, n_heads, head_dim):
    """创建一个hook, 将指定头的输出置零
    
    在self_attn的输出上做修改: 将head_idx对应的维度置零
    """
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0].clone()
            # 注意力输出形状: [1, seq, n_heads*head_dim]
            # 将head_idx对应的维度置零
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            h[:, :, start:end] = 0.0
            return (h,) + output[1:]
        else:
            h = output.clone()
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            h[:, :, start:end] = 0.0
            return h
    return hook_fn


# ============================================================
# H1: Head Ablation — 逐头关闭, 找极性头
# ============================================================

def test_h1(model_name, model, tokenizer, device, d_model, n_layers, W_U):
    """H1: Head Ablation — 逐个关闭注意力头, 看'not' logit变化"""
    
    print("\n" + "=" * 70)
    print("H1: Head Ablation — 找极性关键头")
    print("=" * 70)
    
    n_heads = _get_n_heads(model, model_name)
    head_dim = _get_head_dim(model)
    not_tok_id = _get_not_token_id(tokenizer)
    
    print(f"  n_heads={n_heads}, head_dim={head_dim}, not_tok_id={not_tok_id}")
    
    layers = _get_layers(model)
    
    # 采样层
    sample_layers = list(range(0, n_layers, max(1, n_layers // 6))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    results = {
        'n_heads': n_heads,
        'head_dim': head_dim,
        'sample_layers': sample_layers,
        'head_impact': {},  # {(layer, head): avg_delta_not_logit}
    }
    
    # 对每个否定句, 计算基线"not" logit
    neg_texts = [neg for _, neg, _ in POLARITY_PAIRS[:15]]
    
    # 计算基线
    print("\n  [H1.1] 计算基线...")
    baseline_logits = {}
    for neg in neg_texts:
        toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
        baseline_logits[neg] = logits[not_tok_id] if not_tok_id else 0.0
    
    avg_baseline = np.mean(list(baseline_logits.values()))
    print(f"  基线: avg 'not' logit = {avg_baseline:+.3f}")
    
    # 逐层逐头ablation
    print("\n  [H1.2] 逐头Ablation...")
    
    total_tests = len(sample_layers) * n_heads
    done = 0
    
    for li in sample_layers:
        layer_impact = {}
        
        for h_idx in range(n_heads):
            # 对每个否定句, 关闭该头, 计算"not" logit变化
            deltas = []
            
            for neg in neg_texts:
                toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
                
                # 注册hook关闭head
                hook = layers[li].self_attn.register_forward_hook(
                    _ablate_head_hook(h_idx, n_heads, head_dim))
                
                with torch.no_grad():
                    logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
                
                hook.remove()
                
                delta = logits[not_tok_id] - baseline_logits[neg] if not_tok_id else 0.0
                deltas.append(delta)
            
            avg_delta = np.mean(deltas)
            layer_impact[h_idx] = avg_delta
            
            done += 1
            if done % 50 == 0 or abs(avg_delta) > 1.0:
                print(f"    L{li} H{h_idx}: Δnot_logit={avg_delta:+.4f} ({done}/{total_tests})")
        
        results['head_impact'][li] = layer_impact
        
        # 打印该层Top-5头
        sorted_heads = sorted(layer_impact.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  L{li} Top-5 heads:")
        for h_idx, delta in sorted_heads[:5]:
            print(f"    H{h_idx}: Δnot_logit={delta:+.4f}")
    
    # 跨层汇总: 找到对极性最关键的头
    print("\n  [H1.3] 跨层汇总...")
    
    all_head_impacts = []
    for li, layer_impact in results['head_impact'].items():
        for h_idx, delta in layer_impact.items():
            all_head_impacts.append((li, h_idx, delta))
    
    all_head_impacts.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print(f"\n  全局Top-10 极性关键头:")
    for li, h_idx, delta in all_head_impacts[:10]:
        print(f"    L{li} H{h_idx}: Δnot_logit={delta:+.4f}")
    
    # 统计: 极性头 vs 非极性头的差异
    impacts = [abs(x[2]) for x in all_head_impacts]
    top10_avg = np.mean(sorted(impacts, reverse=True)[:10])
    all_avg = np.mean(impacts)
    random_avg = np.mean(sorted(impacts)[:max(1, len(impacts)//2)])
    
    print(f"\n  Top-10头平均|Δ|: {top10_avg:.4f}")
    print(f"  所有头平均|Δ|:   {all_avg:.4f}")
    print(f"  随机头平均|Δ|:   {random_avg:.4f}")
    print(f"  极性头/随机头:   {top10_avg/max(random_avg, 1e-6):.1f}x")
    
    results['top10_avg'] = float(top10_avg)
    results['all_avg'] = float(all_avg)
    results['random_avg'] = float(random_avg)
    results['polarity_head_ratio'] = float(top10_avg / max(random_avg, 1e-6))
    results['top10_heads'] = [(li, h_idx, float(delta)) for li, h_idx, delta in all_head_impacts[:10]]
    
    # 验证: 关闭Top-10头 vs 关闭10个随机头
    print("\n  [H1.4] 验证: Top-10头 vs 随机头...")
    
    # 关闭Top-10头
    top10_not_logits = []
    for neg in neg_texts:
        toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
        hooks = []
        for li, h_idx, _ in all_head_impacts[:10]:
            hooks.append(layers[li].self_attn.register_forward_hook(
                _ablate_head_hook(h_idx, n_heads, head_dim)))
        
        with torch.no_grad():
            logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
        
        for h in hooks:
            h.remove()
        
        top10_not_logits.append(logits[not_tok_id] if not_tok_id else 0.0)
    
    # 关闭10个随机头
    np.random.seed(42)
    random_indices = np.random.choice(len(all_head_impacts), size=10, replace=False)
    random_not_logits = []
    for neg in neg_texts:
        toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
        hooks = []
        for idx in random_indices:
            li, h_idx, _ = all_head_impacts[idx]
            hooks.append(layers[li].self_attn.register_forward_hook(
                _ablate_head_hook(h_idx, n_heads, head_dim)))
        
        with torch.no_grad():
            logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
        
        for h in hooks:
            h.remove()
        
        random_not_logits.append(logits[not_tok_id] if not_tok_id else 0.0)
    
    top10_avg_logit = np.mean(top10_not_logits)
    random_avg_logit = np.mean(random_not_logits)
    
    print(f"  基线 'not' logit:         {avg_baseline:+.3f}")
    print(f"  Top-10头ablated 'not' logit: {top10_avg_logit:+.3f} (Δ={top10_avg_logit-avg_baseline:+.3f})")
    print(f"  随机10头ablated 'not' logit: {random_avg_logit:+.3f} (Δ={random_avg_logit-avg_baseline:+.3f})")
    print(f"  ★ 选择性: Top-10/随机 = {abs(top10_avg_logit-avg_baseline)/max(abs(random_avg_logit-avg_baseline), 1e-6):.1f}x")
    
    results['top10_ablated_logit'] = float(top10_avg_logit)
    results['random10_ablated_logit'] = float(random_avg_logit)
    results['ablation_selectivity'] = float(abs(top10_avg_logit - avg_baseline) / max(abs(random_avg_logit - avg_baseline), 1e-6))
    
    return results


# ============================================================
# H2: Head Attribution — 注意力权重分析信息流
# ============================================================

def test_h2(model_name, model, tokenizer, device, d_model, n_layers, W_U):
    """H2: Head Attribution — 从最终位置追踪信息流"""
    
    print("\n" + "=" * 70)
    print("H2: Head Attribution — 注意力权重信息流分析")
    print("=" * 70)
    
    n_heads = _get_n_heads(model, model_name)
    head_dim = _get_head_dim(model)
    not_tok_id = _get_not_token_id(tokenizer)
    
    print(f"  n_heads={n_heads}, head_dim={head_dim}")
    
    layers = _get_layers(model)
    
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    results = {
        'sample_layers': sample_layers,
        'attn_to_not': {},  # {layer: {head: avg_attn_to_not_from_last}}
        'attn_from_not': {},  # {layer: {head: avg_attn_from_not_to_all}}
    }
    
    neg_texts = [neg for _, neg, _ in POLARITY_PAIRS[:15]]
    
    for li in sample_layers:
        attn_to_not_accum = defaultdict(list)
        attn_from_not_accum = defaultdict(list)
        
        for neg in neg_texts:
            # 找"not"位置
            not_positions = _find_token_position(tokenizer, neg, "not")
            if not not_positions:
                continue
            not_pos = not_positions[0][0]
            
            toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
            seq_len = toks.input_ids.shape[1]
            last_pos = seq_len - 1
            
            # 获取注意力权重
            with torch.no_grad():
                outputs = model(**toks, output_attentions=True)
            
            if outputs.attentions is None:
                continue
            
            attn = outputs.attentions[li]  # [1, n_heads, seq, seq]
            attn_np = attn[0].detach().cpu().float().numpy()  # [n_heads, seq, seq]
            
            for h_idx in range(n_heads):
                # 从last_pos位置看not_pos的注意力权重
                if not_pos < seq_len and last_pos < seq_len:
                    attn_to_not_accum[h_idx].append(float(attn_np[h_idx, last_pos, not_pos]))
                
                # 从not_pos位置看其他位置的注意力权重(总和)
                if not_pos < seq_len:
                    attn_from_not_accum[h_idx].append(float(np.mean(attn_np[h_idx, not_pos, :not_pos+1])))
        
        # 汇总
        attn_to_not = {}
        attn_from_not = {}
        for h_idx in range(n_heads):
            if attn_to_not_accum[h_idx]:
                attn_to_not[h_idx] = np.mean(attn_to_not_accum[h_idx])
            if attn_from_not_accum[h_idx]:
                attn_from_not[h_idx] = np.mean(attn_from_not_accum[h_idx])
        
        results['attn_to_not'][li] = attn_to_not
        results['attn_from_not'][li] = attn_from_not
        
        # 打印该层Top-5
        sorted_to_not = sorted(attn_to_not.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  L{li} — 最后位置→'not'位置注意力 Top-5:")
        for h_idx, attn_val in sorted_to_not[:5]:
            print(f"    H{h_idx}: attn={attn_val:.4f}")
    
    # 跨层汇总
    print("\n  [H2.3] 跨层汇总...")
    
    # 找到对"not"注意力最高的头
    all_attn_to_not = []
    for li, head_attn in results['attn_to_not'].items():
        for h_idx, attn_val in head_attn.items():
            all_attn_to_not.append((li, h_idx, attn_val))
    
    all_attn_to_not.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n  最后位置→'not'注意力 Top-10头:")
    for li, h_idx, attn_val in all_attn_to_not[:10]:
        print(f"    L{li} H{h_idx}: attn={attn_val:.4f}")
    
    results['top_attn_heads'] = [(li, h_idx, float(v)) for li, h_idx, v in all_attn_to_not[:10]]
    
    return results


# ============================================================
# H3: Cross-layer Head Path — 完整极性电路图
# ============================================================

def test_h3(model_name, model, tokenizer, device, d_model, n_layers, W_U):
    """H3: Cross-layer Head Path — 追踪极性信息的完整路径"""
    
    print("\n" + "=" * 70)
    print("H3: Cross-layer Head Path — 极性电路图")
    print("=" * 70)
    
    n_heads = _get_n_heads(model, model_name)
    head_dim = _get_head_dim(model)
    not_tok_id = _get_not_token_id(tokenizer)
    
    # Step 1: 从H1的结果中获取极性关键头(如果有的话, 重新计算)
    # 先用5个样本快速找极性头
    print("\n  [H3.1] 快速定位极性关键层...")
    
    layers = _get_layers(model)
    neg_texts = [neg for _, neg, _ in POLARITY_PAIRS[:10]]
    
    # 逐层ablation(只关闭所有头), 看哪一层最关键
    layer_impacts = {}
    for li in range(n_layers):
        deltas = []
        for neg in neg_texts:
            toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
            
            # 关闭该层所有头
            hooks = []
            for h_idx in range(n_heads):
                hooks.append(layers[li].self_attn.register_forward_hook(
                    _ablate_head_hook(h_idx, n_heads, head_dim)))
            
            with torch.no_grad():
                logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
            
            for h in hooks:
                h.remove()
            
            # 基线
            base_toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
            with torch.no_grad():
                base_logits = model(**base_toks).logits[0, -1, :].detach().cpu().float().numpy()
            
            delta = logits[not_tok_id] - base_logits[not_tok_id] if not_tok_id else 0.0
            deltas.append(delta)
        
        avg_delta = np.mean(deltas)
        layer_impacts[li] = avg_delta
        
        if abs(avg_delta) > 0.5:
            print(f"    L{li}: Δnot_logit={avg_delta:+.4f} ★")
    
    # 找到最关键的3层
    sorted_layers = sorted(layer_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
    top3_layers = [li for li, _ in sorted_layers[:3]]
    print(f"\n  最关键3层: {top3_layers}")
    
    # Step 2: 在关键层中找极性头
    print("\n  [H3.2] 在关键层中找极性头...")
    
    # 计算基线
    baseline_logits = {}
    for neg in neg_texts:
        toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
        baseline_logits[neg] = logits[not_tok_id] if not_tok_id else 0.0
    
    key_heads = {}  # {layer: [(head_idx, delta), ...]}
    
    for li in top3_layers:
        head_impacts = []
        for h_idx in range(n_heads):
            deltas = []
            for neg in neg_texts:
                toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
                hook = layers[li].self_attn.register_forward_hook(
                    _ablate_head_hook(h_idx, n_heads, head_dim))
                
                with torch.no_grad():
                    logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
                
                hook.remove()
                
                delta = logits[not_tok_id] - baseline_logits[neg] if not_tok_id else 0.0
                deltas.append(delta)
            
            avg_delta = np.mean(deltas)
            head_impacts.append((h_idx, avg_delta))
        
        head_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        key_heads[li] = head_impacts[:5]
        
        print(f"\n  L{li} Top-5极性头:")
        for h_idx, delta in head_impacts[:5]:
            print(f"    H{h_idx}: Δ={delta:+.4f}")
    
    # Step 3: 注意力路径追踪
    print("\n  [H3.3] 注意力路径追踪...")
    
    # 对每个关键头, 分析它从哪个位置读取信息
    path_info = {}
    
    for li in top3_layers:
        for h_idx, _ in key_heads[li][:3]:
            # 获取该头在否定句中的注意力模式
            attn_from_not = []
            attn_to_not = []
            
            for neg in neg_texts[:5]:
                not_positions = _find_token_position(tokenizer, neg, "not")
                if not not_positions:
                    continue
                not_pos = not_positions[0][0]
                
                toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
                seq_len = toks.input_ids.shape[1]
                
                with torch.no_grad():
                    outputs = model(**toks, output_attentions=True)
                
                if outputs.attentions is None:
                    continue
                
                attn = outputs.attentions[li][0, h_idx].detach().cpu().float().numpy()
                
                # 从"not"位置看: 它关注哪些位置
                if not_pos < seq_len:
                    attn_from_not.append(attn[not_pos, :not_pos+1])
                
                # 到"not"位置: 哪些位置关注它
                attn_to_not.append(attn[not_pos:, not_pos] if not_pos < seq_len else [])
            
            path_info[(li, h_idx)] = {
                'attn_from_not_avg': np.mean(attn_from_not) if attn_from_not else 0,
                'attn_to_not_avg': np.mean(attn_to_not) if attn_to_not else 0,
            }
            
            print(f"  L{li} H{h_idx}: attn_from_not={path_info[(li, h_idx)]['attn_from_not_avg']:.4f}, "
                  f"attn_to_not={path_info[(li, h_idx)]['attn_to_not_avg']:.4f}")
    
    results = {
        'layer_impacts': {int(k): float(v) for k, v in layer_impacts.items()},
        'top3_layers': top3_layers,
        'key_heads': {str(li): [(int(h), float(d)) for h, d in heads] for li, heads in key_heads.items()},
        'path_info': {f"L{li}_H{h_idx}": v for (li, h_idx), v in path_info.items()},
    }
    
    return results


# ============================================================
# H4: 极性头 vs 随机头 — 统计验证
# ============================================================

def test_h4(model_name, model, tokenizer, device, d_model, n_layers, W_U):
    """H4: 极性头 vs 随机头 — 统计验证"""
    
    print("\n" + "=" * 70)
    print("H4: 极性头 vs 随机头 — 统计验证")
    print("=" * 70)
    
    n_heads = _get_n_heads(model, model_name)
    head_dim = _get_head_dim(model)
    not_tok_id = _get_not_token_id(tokenizer)
    
    layers = _get_layers(model)
    
    # 采样层
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    neg_texts = [neg for _, neg, _ in POLARITY_PAIRS[:15]]
    aff_texts = [aff for aff, _, _ in POLARITY_PAIRS[:15]]
    
    # 计算基线
    print("\n  [H4.1] 计算基线...")
    baseline_neg = {}
    baseline_aff = {}
    for neg in neg_texts:
        toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
        baseline_neg[neg] = logits[not_tok_id] if not_tok_id else 0.0
    
    for aff in aff_texts:
        toks = tokenizer(aff, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
        baseline_aff[aff] = logits[not_tok_id] if not_tok_id else 0.0
    
    avg_neg_baseline = np.mean(list(baseline_neg.values()))
    avg_aff_baseline = np.mean(list(baseline_aff.values()))
    print(f"  否定句 'not' logit: {avg_neg_baseline:+.3f}")
    print(f"  肯定句 'not' logit: {avg_aff_baseline:+.3f}")
    print(f"  差异(否定-肯定):    {avg_neg_baseline - avg_aff_baseline:+.3f}")
    
    # 关键指标: 关闭头后, 否定句和肯定句的"not" logit差异是否缩小?
    # 如果极性头被关闭, 否定句的"not" logit应该下降, 趋近肯定句
    
    print("\n  [H4.2] 逐头Ablation — 极性区分度(否定句-肯定句)...")
    
    # 采样少量层做完整测试
    test_layers = sample_layers[:4] + [n_layers - 1] if len(sample_layers) > 5 else sample_layers
    
    results = {
        'baseline_neg_aff_diff': float(avg_neg_baseline - avg_aff_baseline),
        'head_polarity_scores': {},  # {(layer, head): delta_diff_after_ablation}
    }
    
    for li in test_layers:
        for h_idx in range(n_heads):
            neg_logits_ablated = []
            aff_logits_ablated = []
            
            for neg in neg_texts:
                toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
                hook = layers[li].self_attn.register_forward_hook(
                    _ablate_head_hook(h_idx, n_heads, head_dim))
                
                with torch.no_grad():
                    logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
                
                hook.remove()
                neg_logits_ablated.append(logits[not_tok_id] if not_tok_id else 0.0)
            
            for aff in aff_texts:
                toks = tokenizer(aff, return_tensors="pt", truncation=True, max_length=64).to(device)
                hook = layers[li].self_attn.register_forward_hook(
                    _ablate_head_hook(h_idx, n_heads, head_dim))
                
                with torch.no_grad():
                    logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
                
                hook.remove()
                aff_logits_ablated.append(logits[not_tok_id] if not_tok_id else 0.0)
            
            # 极性区分度变化
            diff_after = np.mean(neg_logits_ablated) - np.mean(aff_logits_ablated)
            diff_before = avg_neg_baseline - avg_aff_baseline
            delta_diff = diff_after - diff_before  # 负值=该头帮助区分极性
            
            results['head_polarity_scores'][(li, h_idx)] = float(delta_diff)
    
    # 排序
    sorted_heads = sorted(results['head_polarity_scores'].items(), key=lambda x: x[1])
    
    print(f"\n  极性区分度下降Top-10头(关闭后极性最难区分):")
    for (li, h_idx), delta in sorted_heads[:10]:
        print(f"    L{li} H{h_idx}: Δpolarity_diff={delta:+.4f}")
    
    # 验证: 关闭Top-5极性头 vs 关闭5个随机头
    print("\n  [H4.3] 批量验证: Top-5极性头 vs 随机5头...")
    
    top5_heads = [(li, h_idx) for (li, h_idx), _ in sorted_heads[:5]]
    np.random.seed(42)
    all_head_keys = list(results['head_polarity_scores'].keys())
    random5_idx = np.random.choice(len(all_head_keys), size=5, replace=False)
    random5_heads = [all_head_keys[i] for i in random5_idx]
    
    # 关闭Top-5极性头
    neg_logits_top5 = []
    aff_logits_top5 = []
    for neg, aff in zip(neg_texts, aff_texts):
        # 否定句
        toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
        hooks = [layers[li].self_attn.register_forward_hook(
            _ablate_head_hook(h_idx, n_heads, head_dim)) for li, h_idx in top5_heads]
        with torch.no_grad():
            logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
        for h in hooks: h.remove()
        neg_logits_top5.append(logits[not_tok_id] if not_tok_id else 0.0)
        
        # 肯定句
        toks = tokenizer(aff, return_tensors="pt", truncation=True, max_length=64).to(device)
        hooks = [layers[li].self_attn.register_forward_hook(
            _ablate_head_hook(h_idx, n_heads, head_dim)) for li, h_idx in top5_heads]
        with torch.no_grad():
            logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
        for h in hooks: h.remove()
        aff_logits_top5.append(logits[not_tok_id] if not_tok_id else 0.0)
    
    # 关闭随机5头
    neg_logits_rand5 = []
    aff_logits_rand5 = []
    for neg, aff in zip(neg_texts, aff_texts):
        toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
        hooks = [layers[li].self_attn.register_forward_hook(
            _ablate_head_hook(h_idx, n_heads, head_dim)) for li, h_idx in random5_heads]
        with torch.no_grad():
            logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
        for h in hooks: h.remove()
        neg_logits_rand5.append(logits[not_tok_id] if not_tok_id else 0.0)
        
        toks = tokenizer(aff, return_tensors="pt", truncation=True, max_length=64).to(device)
        hooks = [layers[li].self_attn.register_forward_hook(
            _ablate_head_hook(h_idx, n_heads, head_dim)) for li, h_idx in random5_heads]
        with torch.no_grad():
            logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
        for h in hooks: h.remove()
        aff_logits_rand5.append(logits[not_tok_id] if not_tok_id else 0.0)
    
    # 极性区分度
    diff_top5 = np.mean(neg_logits_top5) - np.mean(aff_logits_top5)
    diff_rand5 = np.mean(neg_logits_rand5) - np.mean(aff_logits_rand5)
    diff_baseline = avg_neg_baseline - avg_aff_baseline
    
    print(f"\n  基线极性区分度:     {diff_baseline:+.3f}")
    print(f"  Top-5头ablated:     {diff_top5:+.3f} (下降{abs(diff_baseline-diff_top5):.3f})")
    print(f"  随机5头ablated:     {diff_rand5:+.3f} (下降{abs(diff_baseline-diff_rand5):.3f})")
    
    top5_reduction = abs(diff_baseline - diff_top5)
    rand5_reduction = abs(diff_baseline - diff_rand5)
    
    print(f"\n  ★ Top-5极性头区分度下降: {top5_reduction:.3f}")
    print(f"  ★ 随机5头区分度下降:     {rand5_reduction:.3f}")
    print(f"  ★ 选择性: {top5_reduction/max(rand5_reduction, 1e-6):.1f}x")
    
    results['top5_diff'] = float(diff_top5)
    results['random5_diff'] = float(diff_rand5)
    results['top5_reduction'] = float(top5_reduction)
    results['random5_reduction'] = float(rand5_reduction)
    results['polarity_selectivity'] = float(top5_reduction / max(rand5_reduction, 1e-6))
    results['top5_heads'] = [(int(li), int(h_idx)) for li, h_idx in top5_heads]
    results['sorted_heads'] = [((int(li), int(h_idx)), float(d)) for (li, h_idx), d in sorted_heads[:20]]
    
    return results


# ============================================================
# 主入口
# ============================================================

def run_test(model_name, test_name):
    print(f"\n{'='*70}")
    print(f"Phase CLXXXVI: 注意力头级别因果追踪 — {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = _load_model(model_name)
    if model is None:
        return
    
    layers = _get_layers(model)
    n_layers = len(layers)
    d_model = model.get_input_embeddings().weight.shape[1]
    
    n_heads = _get_n_heads(model, model_name)
    head_dim = _get_head_dim(model)
    
    print(f"  Model: {model_name}, n_layers={n_layers}, d_model={d_model}")
    print(f"  n_heads={n_heads}, head_dim={head_dim}")
    
    # 获取W_U
    if hasattr(model, 'lm_head'):
        W_U = model.lm_head.weight.detach().cpu().float().numpy()
    else:
        W_U = None
    
    # 创建结果目录
    result_dir = Path(f"results/head_causal/{model_name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    results = {}
    
    if test_name in ['h1', 'all']:
        r = test_h1(model_name, model, tokenizer, device, d_model, n_layers, W_U)
        results['h1'] = r
        with open(result_dir / "h1_results.json", 'w') as f:
            json.dump(r, f, indent=2, default=str)
        print(f"\n  H1 results saved to {result_dir / 'h1_results.json'}")
    
    if test_name in ['h2', 'all']:
        r = test_h2(model_name, model, tokenizer, device, d_model, n_layers, W_U)
        results['h2'] = r
        with open(result_dir / "h2_results.json", 'w') as f:
            json.dump(r, f, indent=2, default=str)
        print(f"\n  H2 results saved to {result_dir / 'h2_results.json'}")
    
    if test_name in ['h3', 'all']:
        r = test_h3(model_name, model, tokenizer, device, d_model, n_layers, W_U)
        results['h3'] = r
        with open(result_dir / "h3_results.json", 'w') as f:
            json.dump(r, f, indent=2, default=str)
        print(f"\n  H3 results saved to {result_dir / 'h3_results.json'}")
    
    if test_name in ['h4', 'all']:
        r = test_h4(model_name, model, tokenizer, device, d_model, n_layers, W_U)
        results['h4'] = r
        with open(result_dir / "h4_results.json", 'w') as f:
            json.dump(r, f, indent=2, default=str)
        print(f"\n  H4 results saved to {result_dir / 'h4_results.json'}")
    
    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.1f}s")
    
    # 释放模型
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--test", type=str, default="all", choices=["h1", "h2", "h3", "h4", "all"])
    args = parser.parse_args()
    
    run_test(args.model, args.test)
