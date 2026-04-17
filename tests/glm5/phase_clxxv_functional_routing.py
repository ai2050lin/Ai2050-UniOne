"""
Phase CLXXV: 功能路由追踪 — 残差流中的信息选择性传递
======================================================
核心洞察(来自CLXXIV):
  - W_U的SVD分解是"统计最优"而非"功能最优"
  - SV4是跨模型的"语法/功能词枢纽" (GLM4: 10.48x, DS7B: 6.56x)
  - SparsePCA/ICA/NMF等静态分解无法真正解纠缠
  - 功能对齐重建在中等k值略微优势但不显著

新思路: 不再静态分解W_U, 而是动态追踪残差流中的信息路由
  - 每层的残差流 = W_embed + Sum(attention_heads) + Sum(ffn_outputs)
  - 关键问题: 不同功能(语法/语义/风格)的信息是在哪一层、哪个组件中被注入?
  - 如果能追踪到"功能注入点", 就能理解语言背后的计算图

实验设计:
  P751: 残差流的功能梯度
    - 对比不同功能类型句子的残差流逐层变化
    - 计算功能方向(语法方向、语义方向等)在残差流中的投影变化
    - 假说: 语法信息在中间层集中注入, 语义在早期层, 风格在后期层

  P752: 注意力头的功能贡献分解
    - 对每个注意力头, 计算其输出在7个功能方向上的投影
    - 找出每个功能维度的"核心头"和"抑制头"
    - 假说: 少数头承担了特定功能的注入

  P753: FFN的功能角色分析
    - 对每个FFN层, 分析其输出在功能方向上的贡献
    - FFN的输出 = 残差流增量, 增量中各功能维度的占比
    - 假说: FFN主要负责语义扩展, 注意力负责语法路由

  P754: 功能因果干预
    - 在特定层、特定方向上做有向扰动
    - 测量扰动对输出token的功能属性影响
    - 假说: 扰动语法方向会改变句法但不改变语义, 反之亦然
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

from model_utils import load_model, get_model_info, get_layers


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
# 功能方向定义
# ============================================================

# 句子对: 每对只在某个功能维度上不同
FUNCTIONAL_PAIRS = {
    'syntax': [
        ("The cat sits on the mat", "The cats sit on the mat"),         # 单复数
        ("She walks to school", "She walked to school"),               # 时态
        ("The dog chased the cat", "The cat was chased by the dog"),   # 主被动
        ("He is running fast", "Is he running fast?"),                 # 陈述vs疑问
    ],
    'semantic': [
        ("The cat sat on the mat", "The dog sat on the mat"),          # 语义替换
        ("She drank cold water", "She drank hot tea"),                # 语义+属性
        ("The car drove fast", "The bird flew high"),                  # 语义域
        ("He read a book about history", "He read a book about math"), # 话题
    ],
    'style': [
        ("The cat sat on the mat", "The feline settled upon the rug"), # 正式vs非正式
        ("She went to the store", "She proceeded to the establishment"),# 简单vs复杂
        ("It is very cold", "It is exceedingly frigid"),               # 口语vs书面
        ("The dog ran quickly", "The canine hastened with alacrity"),   # 日常vs文学
    ],
    'tense': [
        ("She walks to school every day", "She walked to school every day"),  # 现在vs过去
        ("He will finish the work", "He finished the work"),                   # 将来vs过去
        ("They are playing outside", "They were playing outside"),             # 现在进行vs过去进行
    ],
    'polarity': [
        ("This is a good idea", "This is not a good idea"),           # 肯定vs否定
        ("She likes the movie", "She dislikes the movie"),           # 正vs负
        ("He always comes on time", "He never comes on time"),       # always vs never
    ],
}

# 中文句子对
FUNCTIONAL_PAIRS_CN = {
    'syntax': [
        ("猫坐在垫子上", "猫们坐在垫子上"),              # 单复数
        ("她走路去学校", "她走了路去学校"),              # 时态
        ("狗追猫", "猫被狗追"),                          # 主被动
    ],
    'semantic': [
        ("猫坐在垫子上", "狗坐在垫子上"),                # 语义替换
        ("她喝了冷水", "她喝了热水"),                    # 属性
        ("他读了一本关于历史的书", "他读了一本关于数学的书"), # 话题
    ],
    'style': [
        ("猫坐在垫子上", "猫咪安坐于软垫之上"),          # 口语vs书面
        ("她去商店了", "她前往商铺了"),                    # 简单vs正式
    ],
    'polarity': [
        ("这是个好主意", "这不是个好主意"),                # 肯定vs否定
        ("她喜欢这部电影", "她不喜欢这部电影"),            # 正vs负
    ],
}


# ============================================================
# P751: 残差流的功能梯度
# ============================================================

def _collect_residual_stream(model, tokenizer, device, sentence):
    """对单个句子收集每层输出的最后一个token残差流"""
    ids = tokenizer.encode(sentence, return_tensors='pt').to(device)
    layers_out = {}

    def make_hook(storage, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[layer_idx] = to_numpy(output[0])
            else:
                storage[layer_idx] = to_numpy(output)
        return hook

    hooks = []
    for i, layer in enumerate(get_layers(model)):
        h = layer.register_forward_hook(make_hook(layers_out, i))
        hooks.append(h)

    with torch.no_grad():
        _ = model(ids)

    for h in hooks:
        h.remove()

    # 提取每个层最后一个token的表示
    residual = {}
    for layer_idx in sorted(layers_out.keys()):
        arr = layers_out[layer_idx]
        if arr.ndim == 3:
            residual[layer_idx] = arr[0, -1, :]  # [batch, seq, d_model] -> [d_model]
        elif arr.ndim == 2:
            residual[layer_idx] = arr[-1, :]      # [seq, d_model] -> [d_model]
        else:
            residual[layer_idx] = arr             # [d_model]
    return residual


def P751_residual_functional_gradient(model, tokenizer, device, model_name, results):
    """
    追踪残差流中功能方向的逐层投影变化
    核心思路: 对比同一功能维度的句子对, 差异向量就是"功能方向"
    """
    print("\n--- P751: 残差流的功能梯度 ---")

    n_vocab, d_model = model.lm_head.weight.shape
    n_layers = len(get_layers(model))

    # 收集所有句子对的逐层残差流
    all_pairs_data = {}

    for func_dim, pairs in FUNCTIONAL_PAIRS.items():
        print(f"\n  功能维度: {func_dim} ({len(pairs)} 对)")
        dim_data = []

        for s1, s2 in pairs:
            # 对每个句子独立收集残差流
            residual1 = _collect_residual_stream(model, tokenizer, device, s1)
            residual2 = _collect_residual_stream(model, tokenizer, device, s2)

            if not residual1 or not residual2:
                print(f"    警告: 未获取到层输出")
                continue

            dim_data.append({
                's1': s1, 's2': s2,
                'residual1': residual1,
                'residual2': residual2,
            })

        all_pairs_data[func_dim] = dim_data

    # 分析功能方向的逐层变化
    print("\n  === 功能方向逐层分析 ===")

    functional_gradient = {}

    for func_dim, dim_data in all_pairs_data.items():
        if not dim_data:
            continue

        # 对所有句子对, 计算功能方向(差值向量)在每层的范数和方向
        layer_diff_norms = defaultdict(list)
        layer_cos_sim = defaultdict(list)

        for pair in dim_data:
            r1 = pair['residual1']
            r2 = pair['residual2']

            # 找共同层
            common_layers = sorted(set(r1.keys()) & set(r2.keys()))

            for layer_idx in common_layers:
                diff = r1[layer_idx] - r2[layer_idx]
                diff_norm = np.linalg.norm(diff)
                layer_diff_norms[layer_idx].append(diff_norm)

                # 同层内r1和r2的余弦相似度(衡量功能差异有多大)
                cos = np.dot(r1[layer_idx], r2[layer_idx]) / (
                    np.linalg.norm(r1[layer_idx]) * np.linalg.norm(r2[layer_idx]) + 1e-30
                )
                layer_cos_sim[layer_idx].append(cos)

        # 计算平均
        avg_diff_norms = {}
        avg_cos_sim = {}
        for layer_idx in sorted(layer_diff_norms.keys()):
            avg_diff_norms[layer_idx] = float(np.mean(layer_diff_norms[layer_idx]))
            avg_cos_sim[layer_idx] = float(np.mean(layer_cos_sim[layer_idx]))

        functional_gradient[func_dim] = {
            'avg_diff_norm': avg_diff_norms,
            'avg_cos_sim': avg_cos_sim,
            'n_pairs': len(dim_data),
        }

        # 打印
        print(f"\n  {func_dim}:")
        layers = sorted(avg_diff_norms.keys())
        for li in layers[:5]:
            print(f"    Layer {li}: diff_norm={avg_diff_norms[li]:.4f}, cos_sim={avg_cos_sim[li]:.4f}")
        if len(layers) > 5:
            print(f"    ... ({len(layers)} layers total)")
        # 找最大差异层
        if avg_diff_norms:
            max_layer = max(avg_diff_norms, key=avg_diff_norms.get)
            min_cos_layer = min(avg_cos_sim, key=avg_cos_sim.get)
            print(f"    最大差异层(norm): {max_layer}, 最小余弦层: {min_cos_layer}")

    # 跨维度对比: 不同功能维度在哪一层产生最大分歧
    print("\n  === 跨维度: 功能分歧的峰值层 ===")
    peak_divergence = {}
    for func_dim, fg in functional_gradient.items():
        diff_norms = fg['avg_diff_norm']
        cos_sims = fg['avg_cos_sim']
        if diff_norms:
            peak_norm_layer = max(diff_norms, key=diff_norms.get)
            peak_cos_layer = min(cos_sims, key=cos_sims.get)
            peak_divergence[func_dim] = {
                'peak_norm_layer': peak_norm_layer,
                'peak_norm_value': diff_norms[peak_norm_layer],
                'min_cos_layer': peak_cos_layer,
                'min_cos_value': cos_sims[peak_cos_layer],
            }
            print(f"    {func_dim}: 峰值差异层={peak_norm_layer} (norm={diff_norms[peak_norm_layer]:.4f}), "
                  f"最低余弦层={peak_cos_layer} (cos={cos_sims[peak_cos_layer]:.4f})")

    # 计算跨维度的功能方向正交性
    print("\n  === 跨维度功能方向的正交性 ===")
    # 对每一层, 计算不同功能维度的差异方向之间的余弦相似度
    cross_dim_orthogonality = {}

    # 找所有层的交集
    all_layers = set()
    for func_dim, dim_data in all_pairs_data.items():
        if dim_data:
            for pair in dim_data:
                all_layers.update(pair['residual1'].keys())
    all_layers = sorted(all_layers)

    for layer_idx in all_layers:
        dim_directions = {}
        for func_dim, dim_data in all_pairs_data.items():
            # 平均差异方向
            diffs = []
            for pair in dim_data:
                if layer_idx in pair['residual1'] and layer_idx in pair['residual2']:
                    diff = pair['residual1'][layer_idx] - pair['residual2'][layer_idx]
                    diffs.append(diff)
            if diffs:
                avg_diff = np.mean(diffs, axis=0)
                norm = np.linalg.norm(avg_diff)
                if norm > 1e-10:
                    dim_directions[func_dim] = avg_diff / norm

        # 计算两两余弦相似度
        dim_names = sorted(dim_directions.keys())
        if len(dim_names) >= 2:
            cos_matrix = {}
            for i, d1 in enumerate(dim_names):
                for j, d2 in enumerate(dim_names):
                    if i < j:
                        cos_val = float(np.dot(dim_directions[d1], dim_directions[d2]))
                        cos_matrix[f"{d1}_vs_{d2}"] = cos_val
            cross_dim_orthogonality[str(layer_idx)] = cos_matrix

    # 打印几个关键层
    key_layers = all_layers[:3] + all_layers[len(all_layers)//2:len(all_layers)//2+1] + all_layers[-3:]
    key_layers = sorted(set(key_layers))
    for layer_idx in key_layers:
        if str(layer_idx) in cross_dim_orthogonality:
            cos_m = cross_dim_orthogonality[str(layer_idx)]
            print(f"    Layer {layer_idx}:")
            for pair_key, cos_val in sorted(cos_m.items()):
                print(f"      {pair_key}: cos={cos_val:.4f}")

    results["p751_functional_gradient"] = {
        "functional_gradient": functional_gradient,
        "peak_divergence": peak_divergence,
        "cross_dim_orthogonality": cross_dim_orthogonality,
        "n_layers": n_layers,
        "d_model": d_model,
    }

    return results, all_pairs_data


# ============================================================
# P752: 注意力头的功能贡献分解
# ============================================================

def P752_attention_head_functional_contribution(model, tokenizer, device, model_name, results, all_pairs_data):
    """
    分析每个注意力头的输出在功能方向上的投影
    """
    print("\n--- P752: 注意力头的功能贡献分解 ---")

    n_layers = len(get_layers(model))

    # 获取模型配置中的头数
    if model_name == 'glm4':
        n_heads = 32  # GLM-4-9B
        head_dim = 128
    elif model_name == 'qwen3':
        n_heads = 20  # Qwen3-4B  
        head_dim = 128
    elif model_name == 'deepseek7b':
        n_heads = 28  # DeepSeek-7B
        head_dim = 128
    else:
        n_heads = 12
        head_dim = 64

    # 选取一组测试句子
    test_sentences = [
        "The cat sat on the mat",
        "She walked to the store yesterday",
        "He carefully opened the old door",
        "They will finish the project soon",
        "The weather was very cold and dark",
    ]

    # 为每个功能维度计算一个"功能方向"(从CLXXIV的结果中获取, 或在此重新计算)
    # 简化: 使用all_pairs_data中的差异方向作为功能方向
    func_directions = {}
    for func_dim, dim_data in all_pairs_data.items():
        diffs = []
        for pair in dim_data:
            # 使用中间层的差异方向
            r1 = pair['residual1']
            r2 = pair['residual2']
            common_layers = sorted(set(r1.keys()) & set(r2.keys()))
            if common_layers:
                mid_layer = common_layers[len(common_layers)//2]
                diff = r1[mid_layer] - r2[mid_layer]
                diffs.append(diff)
        if diffs:
            avg_diff = np.mean(diffs, axis=0)
            norm = np.linalg.norm(avg_diff)
            if norm > 1e-10:
                func_directions[func_dim] = avg_diff / norm

    print(f"  功能方向维度: {list(func_directions.keys())}")

    # 对每个句子, 收集每层每个头的输出
    head_contributions = defaultdict(lambda: defaultdict(dict))
    # head_contributions[func_dim][layer_idx][head_idx] = avg_projection

    for sent in test_sentences:
        ids = tokenizer.encode(sent, return_tensors='pt').to(device)

        # 获取每层的注意力输出
        # 使用hook获取attn_output (attn后的残差连接增量)
        attn_outputs = {}

        def make_attn_hook(layer_idx):
            def hook(module, input, output):
                # output[0]是attn output (after projection)
                # 我们需要的是attn对残差流的贡献
                if isinstance(output, tuple) and len(output) >= 1:
                    attn_out = output[0]  # [batch, seq, d_model]
                    attn_outputs[layer_idx] = to_numpy(attn_out)
            return hook

        hooks = []
        for i, layer in enumerate(get_layers(model)):
            # Hook到self_attention的output
            if hasattr(layer, 'self_attention'):
                h = layer.self_attention.register_forward_hook(make_attn_hook(i))
            elif hasattr(layer, 'attention'):
                h = layer.attention.register_forward_hook(make_attn_hook(i))
            else:
                # 尝试第一个子模块
                children = list(layer.named_children())
                if children:
                    h = children[0][1].register_forward_hook(make_attn_hook(i))
                else:
                    continue
            hooks.append(h)

        with torch.no_grad():
            _ = model(ids)

        for h in hooks:
            h.remove()

        # 分析每层注意力输出在功能方向上的投影
        for layer_idx, attn_out in attn_outputs.items():
            # attn_out: [batch, seq, d_model], 取最后一个token
            last_token_attn = attn_out[0, -1, :]  # [d_model]

            # 分解到每个head
            # 注意力输出 = Concat(head_1, ..., head_n) @ W_O
            # 但W_O是混合的, 无法直接分解
            # 近似: 将attn_output投影到功能方向上

            for func_dim, direction in func_directions.items():
                proj = np.dot(last_token_attn, direction)
                if layer_idx not in head_contributions[func_dim]:
                    head_contributions[func_dim][layer_idx] = {'projections': []}
                head_contributions[func_dim][layer_idx]['projections'].append(float(proj))

    # 计算每层的平均功能投影
    print("\n  === 每层注意力输出的功能投影 ===")
    layer_func_profile = {}

    for func_dim in sorted(head_contributions.keys()):
        print(f"\n  {func_dim}:")
        for layer_idx in sorted(head_contributions[func_dim].keys()):
            projs = head_contributions[func_dim][layer_idx]['projections']
            avg_proj = float(np.mean(projs))
            std_proj = float(np.std(projs))
            head_contributions[func_dim][layer_idx]['avg_projection'] = avg_proj
            head_contributions[func_dim][layer_idx]['std_projection'] = std_proj

            if layer_idx not in layer_func_profile:
                layer_func_profile[layer_idx] = {}
            layer_func_profile[layer_idx][func_dim] = avg_proj

        # 找峰值层
        layer_avgs = {li: head_contributions[func_dim][li]['avg_projection']
                      for li in head_contributions[func_dim]}
        if layer_avgs:
            peak_layer = max(layer_avgs, key=lambda x: abs(layer_avgs[x]))
            print(f"    峰值层: {peak_layer} (avg_proj={layer_avgs[peak_layer]:.4f})")

    # 每层的功能指纹
    print("\n  === 每层的功能指纹(前5层+后5层+中间) ===")
    all_layer_indices = sorted(layer_func_profile.keys())
    key_layers = (all_layer_indices[:5] + 
                  all_layer_indices[len(all_layer_indices)//2:len(all_layer_indices)//2+3] + 
                  all_layer_indices[-5:])
    key_layers = sorted(set(key_layers))

    for li in key_layers:
        if li in layer_func_profile:
            profile = layer_func_profile[li]
            sorted_dims = sorted(profile.items(), key=lambda x: abs(x[1]), reverse=True)
            top3 = sorted_dims[:3]
            dim_str = ", ".join([f"{d}={v:.4f}" for d, v in top3])
            print(f"    Layer {li}: {dim_str}")

    results["p752_attn_functional"] = {
        "head_contributions": {dim: {str(k): v for k, v in layers.items()}
                               for dim, layers in head_contributions.items()},
        "layer_func_profile": {str(k): v for k, v in layer_func_profile.items()},
        "n_heads": n_heads,
        "head_dim": head_dim,
    }

    return results


# ============================================================
# P753: FFN的功能角色分析
# ============================================================

def P753_ffn_functional_role(model, tokenizer, device, model_name, results, all_pairs_data):
    """
    分析FFN输出在功能方向上的贡献
    FFN输出 = 残差流增量, 分析增量中各功能维度的占比
    """
    print("\n--- P753: FFN的功能角色分析 ---")

    # 使用与P751相同的功能方向
    func_directions = {}
    for func_dim, dim_data in all_pairs_data.items():
        diffs = []
        for pair in dim_data:
            r1 = pair['residual1']
            r2 = pair['residual2']
            common_layers = sorted(set(r1.keys()) & set(r2.keys()))
            if common_layers:
                mid_layer = common_layers[len(common_layers)//2]
                diff = r1[mid_layer] - r2[mid_layer]
                diffs.append(diff)
        if diffs:
            avg_diff = np.mean(diffs, axis=0)
            norm = np.linalg.norm(avg_diff)
            if norm > 1e-10:
                func_directions[func_dim] = avg_diff / norm

    # 测试句子
    test_sentences = [
        "The cat sat on the mat",
        "She walked to the store yesterday",
        "He carefully opened the old door",
        "They will finish the project soon",
        "The weather was very cold and dark",
    ]

    # 收集每层FFN输出
    ffn_outputs = {}
    mlp_outputs = {}

    for sent in test_sentences:
        ids = tokenizer.encode(sent, return_tensors='pt').to(device)

        layer_ffn = {}

        def make_ffn_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                layer_ffn[layer_idx] = to_numpy(out)
            return hook

        hooks = []
        for i, layer in enumerate(get_layers(model)):
            # FFN/MLP模块
            if hasattr(layer, 'mlp'):
                h = layer.mlp.register_forward_hook(make_ffn_hook(i))
            elif hasattr(layer, 'feed_forward'):
                h = layer.feed_forward.register_forward_hook(make_ffn_hook(i))
            elif hasattr(layer, 'ffn'):
                h = layer.ffn.register_forward_hook(make_ffn_hook(i))
            else:
                # 尝试找到MLP子模块
                children = list(layer.named_children())
                for name, child in children:
                    if 'mlp' in name.lower() or 'ffn' in name.lower() or 'feed_forward' in name.lower():
                        h = child.register_forward_hook(make_ffn_hook(i))
                        hooks.append(h)
                        break
                continue
            hooks.append(h)

        with torch.no_grad():
            _ = model(ids)

        for h in hooks:
            h.remove()

        # 分析FFN输出在功能方向上的投影
        for layer_idx, ffn_out in layer_ffn.items():
            # ffn_out: [batch, seq, d_model], 取最后一个token
            last_token_ffn = ffn_out[0, -1, :]

            for func_dim, direction in func_directions.items():
                proj = float(np.dot(last_token_ffn, direction))
                if layer_idx not in mlp_outputs:
                    mlp_outputs[layer_idx] = defaultdict(list)
                mlp_outputs[layer_idx][func_dim].append(proj)

    # 汇总
    print("\n  === FFN输出的功能投影 (逐层平均) ===")
    ffn_functional_profile = {}

    for layer_idx in sorted(mlp_outputs.keys()):
        profile = {}
        for func_dim, projs in mlp_outputs[layer_idx].items():
            profile[func_dim] = float(np.mean(projs))
        ffn_functional_profile[layer_idx] = profile

        sorted_dims = sorted(profile.items(), key=lambda x: abs(x[1]), reverse=True)
        top3 = sorted_dims[:3]
        dim_str = ", ".join([f"{d}={v:.4f}" for d, v in top3])

        # 只打印关键层
        all_layers = sorted(mlp_outputs.keys())
        if (layer_idx in all_layers[:3] or 
            layer_idx in all_layers[len(all_layers)//2:len(all_layers)//2+2] or
            layer_idx in all_layers[-3:]):
            print(f"    Layer {layer_idx}: {dim_str}")

    # 找每个功能维度的FFN峰值层
    print("\n  === 各功能维度的FFN峰值层 ===")
    ffn_peak_layers = {}
    for func_dim in func_directions.keys():
        layer_projs = {}
        for layer_idx, profile in ffn_functional_profile.items():
            if func_dim in profile:
                layer_projs[layer_idx] = profile[func_dim]
        if layer_projs:
            peak_layer = max(layer_projs, key=lambda x: abs(layer_projs[x]))
            ffn_peak_layers[func_dim] = {
                'peak_layer': peak_layer,
                'peak_value': layer_projs[peak_layer],
            }
            print(f"    {func_dim}: 峰值层={peak_layer} (proj={layer_projs[peak_layer]:.4f})")

    results["p753_ffn_functional"] = {
        "ffn_functional_profile": {str(k): v for k, v in ffn_functional_profile.items()},
        "ffn_peak_layers": ffn_peak_layers,
    }

    return results


# ============================================================
# P754: 功能因果干预
# ============================================================

def P754_functional_causal_intervention(model, tokenizer, device, model_name, results, all_pairs_data):
    """
    在特定层、特定功能方向上做有向扰动, 测量对输出的影响
    """
    print("\n--- P754: 功能因果干预 ---")

    n_vocab, d_model = model.lm_head.weight.shape
    n_layers = len(get_layers(model))

    # 功能方向
    func_directions = {}
    for func_dim, dim_data in all_pairs_data.items():
        diffs = []
        for pair in dim_data:
            r1 = pair['residual1']
            r2 = pair['residual2']
            common_layers = sorted(set(r1.keys()) & set(r2.keys()))
            if common_layers:
                mid_layer = common_layers[len(common_layers)//2]
                diff = r1[mid_layer] - r2[mid_layer]
                diffs.append(diff)
        if diffs:
            avg_diff = np.mean(diffs, axis=0)
            norm = np.linalg.norm(avg_diff)
            if norm > 1e-10:
                func_directions[func_dim] = avg_diff / norm

    # 测试句子和干预设置
    test_sentence = "The cat sat on the mat and"
    intervention_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    intervention_scales = [0.5, 1.0, 2.0]  # 扰动强度(残差流范数的比例)

    # 先获取baseline logits
    ids = tokenizer.encode(test_sentence, return_tensors='pt').to(device)
    with torch.no_grad():
        baseline_output = model(ids)
        baseline_logits = to_numpy(baseline_output.logits[0, -1, :])
        baseline_top10 = set(np.argsort(baseline_logits)[-10:][::-1])

    print(f"  Baseline: {test_sentence}")
    print(f"  Baseline top-5 tokens: {tokenizer.batch_decode(torch.tensor(list(baseline_top10))[:5])}")

    # 在每层做功能方向干预
    intervention_results = {}

    for func_dim, direction in func_directions.items():
        print(f"\n  功能干预: {func_dim}")
        dim_results = {}

        for layer_idx in intervention_layers:
            layer_results = {}

            for scale in intervention_scales:
                # 修改该层的残差流: 添加功能方向的扰动
                def make_intervention_hook(direction_np, scale_val):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            hidden = output[0]
                        else:
                            hidden = output

                        # 计算残差流的范数来确定扰动大小
                        batch_size, seq_len, hidden_dim = hidden.shape
                        residual_norm = torch.norm(hidden[0, -1, :]).item()
                        perturbation_magnitude = scale_val * residual_norm

                        # 在最后一个token上添加扰动
                        direction_t = torch.tensor(direction_np, dtype=hidden.dtype, device=hidden.device)
                        perturbation = direction_t.unsqueeze(0).unsqueeze(0) * perturbation_magnitude
                        hidden_modified = hidden.clone()
                        hidden_modified[0, -1, :] += perturbation[0, 0, :]

                        if isinstance(output, tuple):
                            return (hidden_modified,) + output[1:]
                        return hidden_modified
                    return hook

                # 注册hook
                layers = get_layers(model)
                if layer_idx < len(layers):
                    hook = layers[layer_idx].register_forward_hook(
                        make_intervention_hook(direction, scale)
                    )

                    # 前向传播
                    with torch.no_grad():
                        modified_output = model(ids)
                        modified_logits = to_numpy(modified_output.logits[0, -1, :])

                    hook.remove()

                    # 计算影响
                    modified_top10 = set(np.argsort(modified_logits)[-10:][::-1])
                    top10_overlap = len(baseline_top10 & modified_top10) / 10.0

                    logit_diff = modified_logits - baseline_logits
                    logit_l2 = float(np.linalg.norm(logit_diff))
                    logit_cos = float(np.dot(baseline_logits, modified_logits) / (
                        np.linalg.norm(baseline_logits) * np.linalg.norm(modified_logits) + 1e-30
                    ))

                    # 获取top-5变化最大的token
                    top_changes = np.argsort(np.abs(logit_diff))[-5:][::-1]
                    change_tokens = []
                    for tid in top_changes:
                        token_str = tokenizer.decode([tid])
                        change_tokens.append({
                            'token': token_str,
                            'baseline_logit': float(baseline_logits[tid]),
                            'modified_logit': float(modified_logits[tid]),
                            'change': float(logit_diff[tid]),
                        })

                    layer_results[str(scale)] = {
                        'top10_overlap': top10_overlap,
                        'logit_l2': logit_l2,
                        'logit_cos': logit_cos,
                        'top_changes': change_tokens,
                    }

            dim_results[str(layer_idx)] = layer_results

            # 打印关键信息
            for scale in intervention_scales:
                if str(scale) in layer_results:
                    r = layer_results[str(scale)]
                    print(f"    Layer {layer_idx}, scale={scale}: "
                          f"overlap={r['top10_overlap']:.2f}, "
                          f"logit_cos={r['logit_cos']:.4f}, "
                          f"L2={r['logit_l2']:.4f}")

        intervention_results[func_dim] = dim_results

    # 找每个功能维度最敏感的干预层
    print("\n  === 功能干预的敏感层 ===")
    sensitive_layers = {}
    for func_dim, dim_results in intervention_results.items():
        layer_sensitivity = {}
        for layer_str, scales in dim_results.items():
            if '1.0' in scales:
                layer_sensitivity[int(layer_str)] = scales['1.0']['logit_l2']
        if layer_sensitivity:
            most_sensitive = max(layer_sensitivity, key=layer_sensitivity.get)
            sensitive_layers[func_dim] = {
                'most_sensitive_layer': most_sensitive,
                'sensitivity': layer_sensitivity[most_sensitive],
                'all_sensitivities': layer_sensitivity,
            }
            print(f"    {func_dim}: 最敏感层={most_sensitive} (L2={layer_sensitivity[most_sensitive]:.4f})")

    results["p754_intervention"] = {
        "intervention_results": intervention_results,
        "sensitive_layers": sensitive_layers,
        "test_sentence": test_sentence,
        "intervention_scales": intervention_scales,
    }

    return results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "deepseek7b", "glm4"])
    args = parser.parse_args()

    model_name = args.model
    print(f"Phase CLXXV: 功能路由追踪 — 残差流中的信息选择性传递 -- {model_name}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载模型
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)

    results = {"model": model_name, "timestamp": datetime.now().isoformat()}

    # P751: 残差流的功能梯度
    results, all_pairs_data = P751_residual_functional_gradient(
        model, tokenizer, device, model_name, results
    )

    # P752: 注意力头的功能贡献
    results = P752_attention_head_functional_contribution(
        model, tokenizer, device, model_name, results, all_pairs_data
    )

    # P753: FFN的功能角色
    results = P753_ffn_functional_role(
        model, tokenizer, device, model_name, results, all_pairs_data
    )

    # P754: 功能因果干预
    results = P754_functional_causal_intervention(
        model, tokenizer, device, model_name, results, all_pairs_data
    )

    # 释放模型
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 保存
    out_dir = Path(f"results/phase_clxxv")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\nResults saved to {out_file}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
