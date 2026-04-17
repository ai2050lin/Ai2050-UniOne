"""
Phase CLXXIX: 注意力的功能-内容交互机制分析
==========================================
核心目标: 揭示注意力机制中功能层和内容层的交互方式

关键假说: "功能查询内容" — Q在功能空间, K/V在内容空间
→ 注意力机制通过功能方向的Q从内容空间的K/V中检索和组装信息

实验设计:
  P791: Q/K/V与功能/内容空间的对齐度
    - 对每层的W_Q, W_K, W_V做投影分析
    - Q是否对齐功能空间? K/V是否对齐内容空间?
    - 量化: 每个投影矩阵在功能vs内容空间的能量占比

  P792: 注意力头的功能/内容特化
    - 分析每个注意力头在功能/内容空间的对齐度
    - 是否存在"功能头"(Q高功能对齐)和"内容头"(K/V高内容对齐)?
    - 分析头的对齐模式随层的变化

  P793: 功能干预经过注意力层的非线性效应
    - 在残差流中做功能方向干预
    - 对比经过注意力层后的输出变化
    - 功能干预是否触发了内容空间的非线性响应?

  P794: 注意力模式的功能调制
    - 不同功能维度是否影响注意力分布?
    - 时态变化 vs 语义变化 是否导致不同的注意力模式偏移?
    - 功能方向干预是否改变注意力权重?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import gc
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from model_utils import load_model, get_model_info, get_layers, get_layer_weights, release_model


def to_numpy(tensor_or_array):
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.astype(np.float32)
    return tensor_or_array.detach().cpu().float().numpy().astype(np.float32)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================
# 功能句子对 (英文) - 与之前Phase一致
# ============================================================

ENGLISH_FUNCTIONAL_PAIRS = {
    'syntax': [
        ("The cat sits on the mat", "The cats sit on the mat"),
        ("She walks to school", "She walked to school"),
        ("The dog chased the cat", "The cat was chased by the dog"),
        ("He is running fast", "Is he running fast?"),
        ("A bird flies in the sky", "Birds fly in the sky"),
        ("The man reads a book", "The men read books"),
        ("She has eaten dinner", "She had eaten dinner"),
        ("They will go home", "They went home"),
        ("I can see the mountain", "Can I see the mountain?"),
        ("The children play outside", "The child plays outside"),
    ],
    'semantic': [
        ("The cat sat on the mat", "The dog sat on the mat"),
        ("She walked to school", "She drove to school"),
        ("The king ruled the kingdom", "The queen ruled the kingdom"),
        ("He ate an apple", "He ate an orange"),
        ("The sun is bright", "The moon is bright"),
        ("The man is tall", "The woman is tall"),
        ("I love music", "I hate music"),
        ("The car is fast", "The bicycle is fast"),
        ("She bought a house", "She rented a house"),
        ("The doctor cured patients", "The teacher taught students"),
    ],
    'style': [
        ("The cat sat on the mat", "The feline rested upon the rug"),
        ("She walked to school", "She proceeded to the educational institution"),
        ("He is very happy", "He is exceedingly joyful"),
        ("The food was good", "The cuisine was delectable"),
        ("I think this is right", "In my humble opinion, this appears correct"),
        ("The movie was bad", "The cinematic experience was abysmal"),
        ("She said hello", "She articulated a greeting"),
        ("The house is big", "The dwelling is commodious"),
        ("He ran fast", "He sprinted with celerity"),
        ("It was a good day", "It proved to be a splendid day"),
    ],
    'tense': [
        ("I walk to school", "I walked to school"),
        ("She reads a book", "She read a book"),
        ("They play outside", "They played outside"),
        ("He runs every day", "He ran every day"),
        ("We eat dinner together", "We ate dinner together"),
        ("The sun rises at dawn", "The sun rose at dawn"),
        ("She writes in her journal", "She wrote in her journal"),
        ("I understand the problem", "I understood the problem"),
        ("The children sing songs", "The children sang songs"),
        ("He drives to work", "He drove to work"),
    ],
    'polarity': [
        ("She is happy", "She is not happy"),
        ("The movie was good", "The movie was not good"),
        ("He can swim", "He cannot swim"),
        ("I like this song", "I do not like this song"),
        ("The test was easy", "The test was not easy"),
        ("She will come", "She will not come"),
        ("They are rich", "They are not rich"),
        ("We have time", "We do not have time"),
        ("The food is fresh", "The food is not fresh"),
        ("He always arrives early", "He never arrives early"),
    ],
}

# 测试句子 (用于P793/P794)
TEST_SENTENCES = [
    "The cat sat quietly on the old wooden chair",
    "Scientists discovered a new species in the rainforest",
    "She carefully opened the mysterious package",
    "The city skyline glowed against the sunset",
    "He always remembered to call his mother",
    "The ancient temple stood on the hilltop",
    "Music filled the concert hall with beauty",
    "The river flowed gently through the valley",
]


def extract_functional_directions(model, tokenizer, device, model_name, pairs_dict, target_layer=None):
    """提取功能方向，返回正交化的方向矩阵"""
    info = get_model_info(model, model_name)
    d_model = info.d_model
    
    if target_layer is None:
        target_layer = 0
    
    # 获取残差流
    def get_residual(text, layer_idx):
        tokens = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        hs = outputs.hidden_states[layer_idx]
        return to_numpy(hs[0].mean(dim=0))  # [d_model]
    
    # 提取每个维度的差异向量
    directions = {}
    for dim_name, pairs in pairs_dict.items():
        diffs = []
        for s1, s2 in pairs:
            r1 = get_residual(s1, target_layer)
            r2 = get_residual(s2, target_layer)
            diff = r2 - r1
            norm = np.linalg.norm(diff)
            if norm > 1e-8:
                diffs.append(diff / norm)
        if diffs:
            avg_dir = np.mean(diffs, axis=0)
            norm = np.linalg.norm(avg_dir)
            if norm > 1e-8:
                directions[dim_name] = avg_dir / norm
    
    # Gram-Schmidt正交化
    dim_order = list(directions.keys())
    ortho_dirs = []
    ortho_labels = []
    for dim_name in dim_order:
        v = directions[dim_name].copy()
        for u in ortho_dirs:
            v -= np.dot(v, u) * u
        norm = np.linalg.norm(v)
        if norm > 0.01:
            ortho_dirs.append(v / norm)
            ortho_labels.append(dim_name)
    
    if not ortho_dirs:
        return None, None, None, None
    
    V_func = np.array(ortho_dirs)  # [n_func, d_model]
    n_func = V_func.shape[0]
    
    # 构建投影矩阵 (避免创建d_model x d_model矩阵)
    # P_func = V_func^T @ V_func
    # P_nonfunc = I - P_func
    
    return V_func, n_func, d_model, ortho_labels


# ============================================================
# P791: Q/K/V与功能/内容空间的对齐度
# ============================================================

def p791_qkv_alignment(model, tokenizer, device, model_name, V_func, n_func, d_model):
    """分析Q/K/V权重矩阵在功能/内容空间的能量分布"""
    print("\n" + "="*60)
    print("P791: Q/K/V与功能/内容空间的对齐度")
    print("="*60)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = len(layers)
    
    # 采样层
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    results = {}
    
    for layer_idx in sample_layers:
        lw = get_layer_weights(layers[layer_idx], d_model, info.mlp_type)
        
        # W_Q, W_K, W_V, W_O 的形状:
        # W_Q: [n_heads*head_dim, d_model] (可能 > d_model)
        # W_K: [n_kv_heads*head_dim, d_model] (GQA中可能 < d_model)
        # W_V: [n_kv_heads*head_dim, d_model]
        # W_O: [d_model, n_heads*head_dim]
        # V_func: [n_func, d_model]
        # 注意: 只有与d_model维度匹配才能做投影
        
        W_Q = lw.W_q  # [n_heads*head_dim, d_model]
        W_K = lw.W_k  # [n_kv_heads*head_dim, d_model]
        W_V = lw.W_v  # [n_kv_heads*head_dim, d_model]
        W_O = lw.W_o  # [d_model, n_heads*head_dim]
        
        # 计算每个权重矩阵在功能空间的能量占比
        # V_func: [n_func, d_model], 只能投影d_model维
        
        def compute_alignment(W, V_func, n_func, d_model):
            """计算W在功能/内容空间的能量分布"""
            # W的形状可能是 [X, d_model] 或 [d_model, X]
            # 只有d_model维度可以与V_func对齐
            
            # 情况1: W = [X, d_model] → 投影 W @ V_func^T
            if W.shape[1] == d_model:
                W_func = W @ V_func.T  # [X, n_func]
                func_energy = np.sum(W_func ** 2)
                total_energy = np.sum(W ** 2)
            # 情况2: W = [d_model, X] → 投影 V_func @ W
            elif W.shape[0] == d_model:
                W_func = V_func @ W  # [n_func, X]
                func_energy = np.sum(W_func ** 2)
                total_energy = np.sum(W ** 2)
            else:
                # 两个维度都不匹配d_model, 无法投影
                return 0.0, 1.0, float(np.sum(W ** 2))
            
            func_ratio = func_energy / (total_energy + 1e-30)
            content_ratio = 1 - func_ratio
            
            return float(func_ratio), float(content_ratio), float(total_energy)
        
        q_func, q_content, q_total = compute_alignment(W_Q, V_func, n_func, d_model)
        k_func, k_content, k_total = compute_alignment(W_K, V_func, n_func, d_model)
        v_func, v_content, v_total = compute_alignment(W_V, V_func, n_func, d_model)
        o_func, o_content, o_total = compute_alignment(W_O, V_func, n_func, d_model)
        
        print(f"\n  Layer {layer_idx}:")
        print(f"    W_Q: func={q_func:.4f}, content={q_content:.4f}")
        print(f"    W_K: func={k_func:.4f}, content={k_content:.4f}")
        print(f"    W_V: func={v_func:.4f}, content={v_content:.4f}")
        print(f"    W_O: func={o_func:.4f}, content={o_content:.4f}")
        
        results[layer_idx] = {
            'W_Q': {'func_ratio': q_func, 'content_ratio': q_content},
            'W_K': {'func_ratio': k_func, 'content_ratio': k_content},
            'W_V': {'func_ratio': v_func, 'content_ratio': v_content},
            'W_O': {'func_ratio': o_func, 'content_ratio': o_content},
        }
    
    # 汇总趋势
    print("\n  跨层趋势 (功能空间占比):")
    for name in ['W_Q', 'W_K', 'W_V', 'W_O']:
        vals = [results[l][name]['func_ratio'] for l in sample_layers]
        print(f"    {name}: {', '.join([f'{v:.4f}' for v in vals])}")
    
    return results


# ============================================================
# P792: 注意力头的功能/内容特化
# ============================================================

def p792_head_specialization(model, tokenizer, device, model_name, V_func, n_func, d_model):
    """分析每个注意力头在功能/内容空间的对齐度 (兼容GQA)"""
    print("\n" + "="*60)
    print("P792: 注意力头的功能/内容特化")
    print("="*60)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = len(layers)
    
    # 从model config获取头数 (兼容不同模型)
    n_heads = getattr(model.config, 'num_attention_heads', d_model // 128)
    n_kv_heads = getattr(model.config, 'num_key_value_heads', n_heads)
    head_dim = getattr(model.config, 'head_dim', d_model // n_heads)
    n_kv_groups = n_heads // n_kv_heads  # GQA: 每个KV头被多少Q头共享
    
    print(f"  n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, n_kv_groups={n_kv_groups}")
    
    # 采样中间层
    target_layer = n_layers // 4  # 25%深度
    layer = layers[target_layer]
    
    lw = get_layer_weights(layer, d_model, info.mlp_type)
    
    W_Q = lw.W_q  # [n_heads * head_dim, d_model]
    W_K = lw.W_k  # [n_kv_heads * head_dim, d_model]
    W_V = lw.W_v  # [n_kv_heads * head_dim, d_model]
    
    print(f"  W_Q shape: {W_Q.shape}, W_K shape: {W_K.shape}, W_V shape: {W_V.shape}")
    
    results = {
        'n_heads': n_heads,
        'n_kv_heads': n_kv_heads,
        'head_dim': head_dim,
        'n_kv_groups': n_kv_groups,
        'target_layer': target_layer,
        'heads': {}
    }
    
    # 对每个Q头分析
    func_heads = 0
    content_heads = 0
    mixed_heads = 0
    
    for h in range(n_heads):
        start_q = h * head_dim
        end_q = start_q + head_dim
        
        # Q头: [head_dim, d_model]
        q_head = W_Q[start_q:end_q, :]
        
        # 对应的KV头 (GQA: 多个Q头共享一个KV头)
        kv_h = h // n_kv_groups
        start_kv = kv_h * head_dim
        end_kv = start_kv + head_dim
        
        k_head = W_K[start_kv:end_kv, :]
        v_head = W_V[start_kv:end_kv, :]
        
        # 计算功能对齐
        q_func_energy = np.sum((q_head @ V_func.T) ** 2)
        q_total_energy = np.sum(q_head ** 2)
        q_func_ratio = q_func_energy / (q_total_energy + 1e-30)
        
        k_func_energy = np.sum((k_head @ V_func.T) ** 2)
        k_total_energy = np.sum(k_head ** 2)
        k_func_ratio = k_func_energy / (k_total_energy + 1e-30)
        
        v_func_energy = np.sum((v_head @ V_func.T) ** 2)
        v_total_energy = np.sum(v_head ** 2)
        v_func_ratio = v_func_energy / (v_total_energy + 1e-30)
        
        # 分类头
        if q_func_ratio > 0.05:
            head_type = "functional"
            func_heads += 1
        elif k_func_ratio < 0.02 and v_func_ratio < 0.02:
            head_type = "content"
            content_heads += 1
        else:
            head_type = "mixed"
            mixed_heads += 1
        
        results['heads'][h] = {
            'q_func_ratio': float(q_func_ratio),
            'k_func_ratio': float(k_func_ratio),
            'v_func_ratio': float(v_func_ratio),
            'kv_head_idx': kv_h,
            'type': head_type,
        }
    
    print(f"\n  Layer {target_layer} 头特化:")
    print(f"    功能头 (Q高功能对齐): {func_heads}/{n_heads}")
    print(f"    内容头 (K/V低功能对齐): {content_heads}/{n_heads}")
    print(f"    混合头: {mixed_heads}/{n_heads}")
    
    # 打印top功能头
    sorted_by_q_func = sorted(results['heads'].items(), 
                               key=lambda x: x[1]['q_func_ratio'], reverse=True)
    print(f"\n  Top 5 功能对齐头 (Q):")
    for h, data in sorted_by_q_func[:5]:
        print(f"    Head {h}: Q_func={data['q_func_ratio']:.4f}, "
              f"K_func={data['k_func_ratio']:.4f}, V_func={data['v_func_ratio']:.4f}")
    
    # 跨层分析 (采样3个层)
    layer_results = {}
    for layer_idx in [n_layers // 4, n_layers // 2, 3 * n_layers // 4]:
        layer = layers[layer_idx]
        lw = get_layer_weights(layer, d_model, info.mlp_type)
        
        W_Q = lw.W_q
        W_K = lw.W_k
        W_V = lw.W_v
        
        func_h = 0
        content_h = 0
        mixed_h = 0
        
        for h in range(n_heads):
            start_q = h * head_dim
            end_q = start_q + head_dim
            
            q_head = W_Q[start_q:end_q, :]
            q_func_energy = np.sum((q_head @ V_func.T) ** 2)
            q_total_energy = np.sum(q_head ** 2)
            q_func_ratio = q_func_energy / (q_total_energy + 1e-30)
            
            kv_h = h // n_kv_groups
            start_kv = kv_h * head_dim
            end_kv = start_kv + head_dim
            
            k_head = W_K[start_kv:end_kv, :]
            k_func_energy = np.sum((k_head @ V_func.T) ** 2)
            k_total_energy = np.sum(k_head ** 2)
            k_func_ratio = k_func_energy / (k_total_energy + 1e-30)
            
            v_head = W_V[start_kv:end_kv, :]
            v_func_energy = np.sum((v_head @ V_func.T) ** 2)
            v_total_energy = np.sum(v_head ** 2)
            v_func_ratio = v_func_energy / (v_total_energy + 1e-30)
            
            if q_func_ratio > 0.05:
                func_h += 1
            elif k_func_ratio < 0.02 and v_func_ratio < 0.02:
                content_h += 1
            else:
                mixed_h += 1
        
        layer_results[layer_idx] = {
            'functional': func_h,
            'content': content_h,
            'mixed': mixed_h,
        }
        print(f"\n  Layer {layer_idx}: func={func_h}, content={content_h}, mixed={mixed_h}")
    
    results['cross_layer'] = layer_results
    
    return results


# ============================================================
# P793: 功能干预经过注意力层的非线性效应
# ============================================================

def p793_functional_intervention_attention(model, tokenizer, device, model_name, V_func, n_func, d_model, ortho_labels):
    """在残差流中做功能方向干预，追踪经过注意力层后的变化"""
    print("\n" + "="*60)
    print("P793: 功能干预经过注意力层的非线性效应")
    print("="*60)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = len(layers)
    
    # 采样层
    target_layer = n_layers // 2  # 中间层
    
    # 测试句子
    test_text = TEST_SENTENCES[0]
    
    # 获取残差流
    tokens = tokenizer(test_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    
    # 中间层残差流
    residual = outputs.hidden_states[target_layer]  # [1, seq_len, d_model]
    seq_len = residual.shape[1]
    
    results = {}
    
    print(f"\n  测试文本: '{test_text}'")
    print(f"  目标层: {target_layer}, seq_len={seq_len}")
    
    # 对每个功能方向做干预
    for i, label in enumerate(ortho_labels):
        func_dir = V_func[i]  # [d_model]
        
        # 干预: 在最后一个token的位置添加功能方向
        scale = 2.0  # 干预强度
        
        # 原始残差流
        orig_residual = residual.clone()
        
        # 干预后的残差流
        intervened_residual = orig_residual.clone()
        intervened_residual[0, -1, :] += torch.tensor(scale * func_dir, 
                                                        dtype=residual.dtype, device=device)
        
        # 计算差异 (在残差流层面)
        diff_residual = to_numpy(intervened_residual[0, -1, :] - orig_residual[0, -1, :])
        
        # 差异应该主要是功能方向
        func_component = np.dot(diff_residual, func_dir) * func_dir
        content_component = diff_residual - func_component
        
        func_norm = np.linalg.norm(func_component)
        content_norm = np.linalg.norm(content_component)
        
        print(f"\n  {label} 干预 (scale={scale}):")
        print(f"    残差流层面: 功能分量={func_norm:.4f}, 内容分量={content_norm:.4f}")
        
        # 传播到下一层看非线性效应
        if target_layer + 1 < n_layers:
            next_layer = layers[target_layer]
            lw = get_layer_weights(next_layer, d_model, info.mlp_type)
            W_O = lw.W_o  # [d_model, n_heads*head_dim] 或 [d_model, d_model]
            W_O_np = to_numpy(W_O)
            
            # 注意力输出 = softmax(QK^T/√d) @ V @ W_O
            # 简化分析: W_O的输出空间中功能vs内容的分布
            # W_O: [d_model, n_heads*head_dim]
            # V_func: [n_func, d_model]
            # V_func @ W_O: [n_func, n_heads*head_dim] → 功能方向在W_O列空间中的投影
            
            func_proj = V_func @ W_O  # [n_func, n_heads*head_dim]
            func_energy = np.sum(func_proj ** 2)
            total_energy = np.sum(W_O_np ** 2)
            
            func_ratio = func_energy / (total_energy + 1e-30)
            content_ratio = 1 - func_ratio
            
            # 泄漏比: W_O输出中非功能/功能的比例
            leak_ratio = content_ratio / (func_ratio + 1e-30)
            
            print(f"    W_O输出分布: 功能={func_ratio:.4f}, 内容={content_ratio:.4f}")
            print(f"      功能→内容泄漏比: {leak_ratio:.4f}")
            
            results[label] = {
                'residual_level': {
                    'func_norm': float(func_norm),
                    'content_norm': float(content_norm),
                },
                'after_W_O': {
                    'func_ratio': float(func_ratio),
                    'content_ratio': float(content_ratio),
                    'leak_ratio': float(leak_ratio),
                }
            }
    
    # FFN同样分析
    print("\n  --- FFN变换分析 ---")
    for i, label in enumerate(ortho_labels):
        func_dir = V_func[i]
        
        next_layer = layers[target_layer]
        lw = get_layer_weights(next_layer, d_model, info.mlp_type)
        
        if lw.W_down is None:
            print(f"  {label}: W_down is None, skipping FFN analysis")
            continue
        W_down = to_numpy(lw.W_down)  # [d_model, intermediate]
        intermediate_size = W_down.shape[1]
        
        if lw.W_gate_up is not None:
            # merged gate_up: [2*intermediate, d_model]
            W_gate_up = to_numpy(lw.W_gate_up)
            # 注意: 合并方式可能不同, 需要按intermediate_size分割
            W_up = W_gate_up[:intermediate_size, :]  # [intermediate, d_model]
            W_gate = W_gate_up[intermediate_size:, :]  # [intermediate, d_model]
        elif lw.W_up is not None:
            W_up = to_numpy(lw.W_up)
            W_gate = to_numpy(lw.W_gate) if lw.W_gate is not None else W_up
        else:
            continue
        
        # FFN: x -> W_down @ (silu(W_gate @ x) * (W_up @ x))
        # 功能方向经过FFN
        up_out = W_up @ func_dir  # [intermediate]
        gate_out = W_gate @ func_dir  # [intermediate]
        # SiLU激活: silu(x) = x * sigmoid(x) ≈ x * (1/(1+exp(-x)))
        # 近似: 对于大值, sigmoid(x)≈1; 对于小值, sigmoid(x)≈0.5
        sigmoid_gate = 1.0 / (1.0 + np.exp(-gate_out))
        activated = up_out * sigmoid_gate  # [intermediate] - SiLU(W_gate @ x) * (W_up @ x) 的简化
        
        # W_down: [d_model, intermediate], 正确计算: W_down @ activated
        down_out = W_down @ activated  # [d_model]
        
        down_norm = np.linalg.norm(down_out)
        if down_norm > 1e-8:
            down_in_func = np.dot(down_out, V_func.T)
            down_func_energy = np.sum(down_in_func ** 2)
            down_content_energy = down_norm ** 2 - down_func_energy
            leak_ratio_ffn = down_content_energy / (down_func_energy + 1e-30)
        else:
            down_func_energy = 0
            down_content_energy = 0
            leak_ratio_ffn = 0
        
        print(f"\n  {label} FFN变换:")
        print(f"    输出norm={down_norm:.4f}")
        print(f"    功能空间能量: {down_func_energy:.4f}")
        print(f"    内容空间能量: {down_content_energy:.4f}")
        print(f"    功能→内容泄漏比: {leak_ratio_ffn:.4f}")
        
        if label in results:
            results[label]['after_FFN'] = {
                'total_norm': float(down_norm),
                'func_energy': float(down_func_energy),
                'content_energy': float(down_content_energy),
                'leak_ratio': float(leak_ratio_ffn),
            }
    
    return results


# ============================================================
# P794: 注意力模式的功能调制
# ============================================================

def p794_attention_pattern_modulation(model, tokenizer, device, model_name, V_func, n_func, d_model, ortho_labels):
    """分析功能维度对注意力模式的影响"""
    print("\n" + "="*60)
    print("P794: 注意力模式的功能调制")
    print("="*60)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = len(layers)
    
    # 采样层
    target_layer = n_layers // 2
    
    results = {}
    
    # 对每个功能维度，构造一对句子
    test_pairs = {}
    for dim_name in ortho_labels:
        if dim_name in ENGLISH_FUNCTIONAL_PAIRS:
            test_pairs[dim_name] = ENGLISH_FUNCTIONAL_PAIRS[dim_name][0]
    
    print(f"\n  目标层: {target_layer}")
    
    # 获取注意力模式
    for dim_name, (s1, s2) in test_pairs.items():
        # 获取两个句子的注意力权重
        for sent_label, text in [('base', s1), ('modified', s2)]:
            tokens = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**tokens, output_attentions=True)
            
            # 检查是否返回了attention weights
            if outputs.attentions is None or target_layer >= len(outputs.attentions):
                print(f"  {dim_name}_{sent_label}: attention weights unavailable, skipping")
                continue
            
            # 目标层的注意力
            attn_weights = outputs.attentions[target_layer]  # [1, n_heads, seq_len, seq_len]
            attn_np = to_numpy(attn_weights[0])  # [n_heads, seq_len, seq_len]
            
            # 统计注意力模式
            n_heads = attn_np.shape[0]
            seq_len = attn_np.shape[1]
            
            # 注意力集中度 (平均top-1注意力权重)
            top1_weights = []
            for h in range(n_heads):
                for q_pos in range(seq_len):
                    top1_weights.append(float(np.max(attn_np[h, q_pos, :])))
            avg_top1 = np.mean(top1_weights)
            
            # 注意力分散度 (熵)
            entropies = []
            for h in range(n_heads):
                for q_pos in range(seq_len):
                    p = attn_np[h, q_pos, :] + 1e-30
                    entropies.append(float(-np.sum(p * np.log(p))))
            avg_entropy = np.mean(entropies)
            
            # 对角注意力比例 (self-attention)
            diag_attn = 0
            total_attn = 0
            for h in range(n_heads):
                for q_pos in range(seq_len):
                    diag_attn += attn_np[h, q_pos, q_pos]
                    total_attn += 1
            diag_ratio = diag_attn / (total_attn + 1e-30)
            
            key = f"{dim_name}_{sent_label}"
            results[key] = {
                'avg_top1': float(avg_top1),
                'avg_entropy': float(avg_entropy),
                'diag_ratio': float(diag_ratio),
                'n_heads': n_heads,
                'seq_len': seq_len,
            }
            
            print(f"  {key}: top1={avg_top1:.4f}, entropy={avg_entropy:.4f}, "
                  f"diag={diag_ratio:.4f}")
        
        # 计算base vs modified的差异
        base_key = f"{dim_name}_base"
        mod_key = f"{dim_name}_modified"
        if base_key in results and mod_key in results:
            top1_diff = results[mod_key]['avg_top1'] - results[base_key]['avg_top1']
            entropy_diff = results[mod_key]['avg_entropy'] - results[base_key]['avg_entropy']
            diag_diff = results[mod_key]['diag_ratio'] - results[base_key]['diag_ratio']
            results[f"{dim_name}_diff"] = {
                'top1_diff': float(top1_diff),
                'entropy_diff': float(entropy_diff),
                'diag_diff': float(diag_diff),
            }
            print(f"  {dim_name} diff: top1={top1_diff:+.4f}, "
                  f"entropy={entropy_diff:+.4f}, diag={diag_diff:+.4f}")
    
    # 不同功能维度的注意力调制模式对比
    print("\n  各功能维度的注意力调制强度:")
    for dim_name in ortho_labels:
        diff_key = f"{dim_name}_diff"
        if diff_key in results:
            d = results[diff_key]
            modulation_strength = abs(d['top1_diff']) + abs(d['entropy_diff']) + abs(d['diag_diff'])
            print(f"    {dim_name}: total_modulation={modulation_strength:.4f}")
            results[diff_key]['total_modulation'] = float(modulation_strength)
    
    return results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase CLXXIX: Attention Functional-Content Interaction")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["glm4", "qwen3", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    print(f"\n{'#'*70}")
    print(f"# Phase CLXXIX: Attention Functional-Content Interaction")
    print(f"# Model: {model_name}")
    print(f"{'#'*70}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    
    print(f"\nModel: {info.name}, Layers={info.n_layers}, d_model={info.d_model}")
    
    # 提取功能方向 (Layer 0)
    print("\n提取功能方向 (Layer 0)...")
    V_func, n_func, d_model, ortho_labels = extract_functional_directions(
        model, tokenizer, device, model_name, ENGLISH_FUNCTIONAL_PAIRS, target_layer=0
    )
    
    if V_func is None:
        print("ERROR: 无法提取功能方向")
        release_model(model)
        return
    
    print(f"  功能方向数: {n_func}, 维度: {ortho_labels}")
    
    # P791: Q/K/V对齐度
    p791_results = p791_qkv_alignment(model, tokenizer, device, model_name, V_func, n_func, d_model)
    
    # P792: 注意力头特化
    p792_results = p792_head_specialization(model, tokenizer, device, model_name, V_func, n_func, d_model)
    
    # P793: 功能干预经过注意力层
    p793_results = p793_functional_intervention_attention(
        model, tokenizer, device, model_name, V_func, n_func, d_model, ortho_labels
    )
    
    # P794: 注意力模式调制
    p794_results = p794_attention_pattern_modulation(
        model, tokenizer, device, model_name, V_func, n_func, d_model, ortho_labels
    )
    
    # 保存结果
    all_results = {
        'model': model_name,
        'model_info': {
            'n_layers': info.n_layers,
            'd_model': info.d_model,
            'vocab_size': info.vocab_size,
        },
        'timestamp': datetime.now().isoformat(),
        'functional_directions': {
            'n_func': n_func,
            'labels': ortho_labels,
        },
        'p791_qkv_alignment': p791_results,
        'p792_head_specialization': p792_results,
        'p793_intervention': p793_results,
        'p794_attention_modulation': p794_results,
    }
    
    output_dir = Path("results/phase_clxxix")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_results.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")
    
    # 释放模型
    release_model(model)
    print("\nPhase CLXXIX 完成!")


if __name__ == "__main__":
    main()
