#!/usr/bin/env python3
"""
Phase CLVIII: 残差流信息分解 — "苹果的信息在哪里?"
===================================================

核心目标: 突破CLVII的瓶颈——编码不在G项内部,而在G项+残差流的交互中

关键思路:
  CLVII发现: G项在logit空间概念区分度极弱(同家族vs跨家族cos差<0.12)
  → G项本身不携带足够的语义区分信息,它只是调制已有信息
  → 需要分析完整的 h_final = h_0 + Σ(G_i + A_i) 的信息流

实验设计:
  P692: h_final的logit逐层分解 — logit(apple) = W_U[apple]·h_final
    逐项分解: = W_U[apple]·h_0 + Σ W_U[apple]·G_i + Σ W_U[apple]·A_i
    哪些层/组件贡献了apple的logit?

  P693: 语义信息的层间传播路径 — 追踪"苹果"语义从嵌入到输出的完整路径
    在每一层, h的W_U[apple]投影如何变化? 哪些层是关键转折点?

  P694: 信息瓶颈分析 — 哪一层是语义信息的关键瓶颈?
    如果消融某一层的输出, logit(apple)如何变化?
    找到信息流的"必经之路"

实验模型: qwen3 -> deepseek7b -> glm4 (串行, 避免OOM)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import json
import time
import argparse
import gc
from datetime import datetime
from pathlib import Path
from scipy import stats
from scipy.sparse.linalg import svds
from model_utils import load_model, get_model_info, get_layer_weights, release_model, MODEL_CONFIGS

import functools
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("d:/develop/TransformerLens-main/results/phase_clviii")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("d:/develop/TransformerLens-main/tests/glm5_temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ============ 测试词汇 ============
TEST_WORDS = {
    "apple":    {"cat": "fruit", "pos": "noun",    "freq": "high"},
    "banana":   {"cat": "fruit", "pos": "noun",    "freq": "high"},
    "orange":   {"cat": "fruit", "pos": "noun",    "freq": "high"},
    "cat":      {"cat": "animal", "pos": "noun",   "freq": "high"},
    "dog":      {"cat": "animal", "pos": "noun",   "freq": "high"},
    "car":      {"cat": "vehicle", "pos": "noun",  "freq": "high"},
    "run":      {"cat": "action", "pos": "verb",   "freq": "high"},
    "red":      {"cat": "color", "pos": "adjective", "freq": "high"},
    "the":      {"cat": "function", "pos": "determiner", "freq": "very_high"},
}


def safe_sigmoid(x):
    """安全 sigmoid, 避免溢出"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def get_layers(model):
    """获取模型的 transformer 层"""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    return []


# ================================================================
# P692: h_final 的 logit 逐层分解
# ================================================================
def p692_logit_layer_decomposition(model, tokenizer, model_info, W_embed, W_U, device):
    """
    P692: h_final 的 logit 逐层分解

    核心问题: logit(apple) = W_U[apple] · h_final 中,
    各层贡献了多少?

    数学:
      h_final = h_0 + Σ_{l=0}^{L-1} (G_l + A_l)
      logit(apple) = W_U[apple] · h_final
                   = W_U[apple] · h_0 + Σ_l [W_U[apple] · G_l + W_U[apple] · A_l]

    算法:
    1. 前向传播, 收集各层 hidden_states
    2. 计算每层的 delta = h_{l+1} - h_l = G_l + A_l
    3. 用 W_U[word_id] · delta 得到每层对 logit 的贡献
    4. 区分 G 项贡献 vs A 项贡献

    关键发现: 哪些层贡献了大部分 logit? G 还是 A 更重要?
    """
    print("\n" + "="*70)
    print("P692: h_final 的 logit 逐层分解")
    print("="*70)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    d_mlp = model_info.intermediate_size
    mlp_type = model_info.mlp_type
    layers = get_layers(model)

    results = {}

    for word in ["apple", "banana", "cat", "run", "the"]:
        print(f"\n  --- {word} ---")
        t0 = time.time()

        # 获取 word_id
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if not word_ids:
            continue
        word_id = word_ids[0]
        W_U_word = W_U[word_id]  # [d_model]

        # 前向传播, 收集所有层的 hidden_states
        text = f"The word is {word}."
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # hidden_states: [n_layers+1, seq_len, d_model]
        # hidden_states[0] = embedding输出, hidden_states[l+1] = 第l层输出
        all_h = []
        for li in range(len(outputs.hidden_states)):
            h = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
            all_h.append(h)

        # 计算最终 logit
        h_final = all_h[-1]
        final_logit = float(W_U_word @ h_final)

        # 逐层分解
        # h_l = all_h[l+1] (第l层输出, 0-indexed)
        # h_0 = all_h[1] (embedding+L0的输出)
        # delta_l = all_h[l+1] - all_h[l] = G_l + A_l

        layer_contributions = []
        cumulative_logit = 0.0

        # embedding 贡献
        h_embed = all_h[0]  # embedding 输出
        embed_logit = float(W_U_word @ h_embed)
        cumulative_logit = embed_logit

        layer_contributions.append({
            "component": "embedding",
            "logit_contribution": embed_logit,
            "cumulative_logit": cumulative_logit,
            "h_norm": float(np.linalg.norm(h_embed)),
        })

        # 逐层贡献
        for li in range(n_layers):
            h_before = all_h[li]      # 第li层输入
            h_after = all_h[li + 1]   # 第li层输出
            delta = h_after - h_before  # G_l + A_l

            delta_logit = float(W_U_word @ delta)
            cumulative_logit += delta_logit

            # 分离 G 和 A 的贡献 (使用权重计算)
            lw = get_layer_weights(layers[li], d_model, mlp_type)
            W_gate = lw.W_gate
            W_up = lw.W_up
            W_down = lw.W_down

            if W_gate is not None and W_up is not None and W_down is not None:
                gate_val = safe_sigmoid(W_gate @ h_before)
                up_val = W_up @ h_before
                post_val = gate_val * up_val
                G_exact = W_down @ post_val
                A_approx = delta - G_exact

                G_logit = float(W_U_word @ G_exact)
                A_logit = float(W_U_word @ A_approx)
            else:
                G_logit = 0.0
                A_logit = delta_logit

            layer_contributions.append({
                "component": f"L{li}",
                "logit_contribution": delta_logit,
                "G_logit_contribution": G_logit,
                "A_logit_contribution": A_logit,
                "cumulative_logit": cumulative_logit,
                "delta_norm": float(np.linalg.norm(delta)),
                "G_norm": float(np.linalg.norm(G_exact)) if W_gate is not None else 0,
                "A_norm": float(np.linalg.norm(A_approx)) if W_gate is not None else 0,
            })

            del lw, W_gate, W_up, W_down

        # 检查分解精度
        decomposition_error = abs(cumulative_logit - final_logit)
        total_G_logit = sum(lc.get("G_logit_contribution", 0) for lc in layer_contributions[1:])
        total_A_logit = sum(lc.get("A_logit_contribution", 0) for lc in layer_contributions[1:])
        total_delta_logit = sum(lc.get("logit_contribution", 0) for lc in layer_contributions[1:])

        # 找到贡献最大的层 (正贡献和负贡献)
        contributions_list = [(lc["component"], lc["logit_contribution"]) for lc in layer_contributions]
        contributions_sorted = sorted(contributions_list, key=lambda x: abs(x[1]), reverse=True)

        G_contributions_list = [(lc["component"], lc.get("G_logit_contribution", 0)) for lc in layer_contributions]
        A_contributions_list = [(lc["component"], lc.get("A_logit_contribution", 0)) for lc in layer_contributions]

        results[word] = {
            "final_logit": final_logit,
            "decomposition_cumulative": cumulative_logit,
            "decomposition_error": decomposition_error,
            "total_G_logit": total_G_logit,
            "total_A_logit": total_A_logit,
            "total_delta_logit": total_delta_logit,
            "embed_logit": embed_logit,
            "top3_positive": sorted(contributions_list, key=lambda x: x[1], reverse=True)[:3],
            "top3_negative": sorted(contributions_list, key=lambda x: x[1])[:3],
            "top3_by_abs": contributions_sorted[:3],
            "G_vs_A_ratio": abs(total_G_logit) / max(abs(total_A_logit), 1e-10),
            "layer_contributions": layer_contributions,
        }

        elapsed = time.time() - t0
        print(f"  {word}: final_logit={final_logit:.4f}, embed={embed_logit:.4f}, "
              f"ΣG={total_G_logit:.4f}, ΣA={total_A_logit:.4f}, "
              f"G/A_ratio={abs(total_G_logit)/max(abs(total_A_logit),1e-10):.2f}, "
              f"top3_abs={[f'{c}:{v:.4f}' for c,v in contributions_sorted[:3]]}, "
              f"elapsed={elapsed:.1f}s")

        del outputs, all_h
        gc.collect()

    # 跨词比较: 不同词的logit分解模式是否不同?
    if len(results) >= 3:
        cross_comparison = {}
        for component_type in ["embedding", "L0", "L1", f"L{n_layers//2}", f"L{n_layers-2}", f"L{n_layers-1}"]:
            word_contribs = {}
            for word, wr in results.items():
                for lc in wr["layer_contributions"]:
                    if lc["component"] == component_type:
                        word_contribs[word] = lc.get("logit_contribution", 0)
                        break
            if word_contribs:
                vals = list(word_contribs.values())
                cross_comparison[component_type] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "range": float(max(vals) - min(vals)),
                    "values": word_contribs,
                }

        results["_cross_comparison"] = cross_comparison

    return results


# ================================================================
# P693: 语义信息的层间传播路径
# ================================================================
def p693_semantic_propagation_path(model, tokenizer, model_info, W_embed, W_U, device):
    """
    P693: 语义信息的层间传播路径

    核心问题: "苹果"语义从嵌入到输出的完整路径是什么?

    算法:
    1. 在每一层, 计算 h 的 W_U 投影:
       proj_l[apple] = W_U[apple] · h_l (logit空间的投影)
    2. 追踪 proj_l 如何随层变化:
       - 哪些层 proj 大幅增加? (注入语义)
       - 哪些层 proj 大幅减少? (抑制语义)
       - 哪些层 proj 稳定? (保持语义)
    3. 计算语义传播的"速度场":
       v_l = proj_{l+1} - proj_l = 每层注入/移除的语义信息量
    4. 语义选择比:
       对 apple, proj[apple] / proj[banana] 的层间变化
       → 模型何时开始区分 apple vs banana?

    关键发现: 语义区分在哪些层发生?
    """
    print("\n" + "="*70)
    print("P693: 语义信息的层间传播路径")
    print("="*70)

    n_layers = model_info.n_layers
    d_model = model_info.d_model

    results = {}

    # 选择5个词进行追踪
    trace_words = ["apple", "banana", "cat", "red", "the"]
    word_ids = {}
    for w in trace_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            word_ids[w] = ids[0]

    for word in trace_words:
        if word not in word_ids:
            continue
        print(f"\n  --- {word} ---")
        t0 = time.time()

        text = f"The word is {word}."
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # 逐层追踪 W_U 投影
        layer_projections = {}  # word_id -> [n_layers+1]
        all_word_projections = {w: [] for w in word_ids}

        for li in range(len(outputs.hidden_states)):
            h = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()

            # 计算对所有追踪词的 W_U 投影
            for w, wid in word_ids.items():
                proj = float(W_U[wid] @ h)
                all_word_projections[w].append(proj)

        # 分析传播路径
        apple_proj = np.array(all_word_projections[word])

        # 1) 语义注入/抑制速度
        velocities = np.diff(apple_proj)  # [n_layers]

        # 2) 关键转折层: |velocity| 最大的层
        key_injection_layers = np.argsort(np.abs(velocities))[::-1][:5]

        # 3) 语义选择比: apple vs banana
        if "banana" in all_word_projections:
            banana_proj = np.array(all_word_projections["banana"])
            selection_ratio = apple_proj / np.maximum(np.abs(banana_proj), 1e-10)
            selection_change = np.diff(selection_ratio)

            # 找到选择比变化最大的层 (区分 apple vs banana 的关键层)
            key_discrimination_layers = np.argsort(np.abs(selection_change))[::-1][:5]
        else:
            selection_ratio = np.zeros_like(apple_proj)
            selection_change = np.zeros_like(apple_proj)
            key_discrimination_layers = np.array([])

        # 4) 语义保真度: h_l 在 W_U[apple] 方向上的投影比
        h_norms = []
        for li in range(len(outputs.hidden_states)):
            h = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
            h_norms.append(float(np.linalg.norm(h)))

        fidelity = apple_proj / np.maximum(np.array(h_norms), 1e-10)

        # 5) 水果 vs 非水果的区分路径
        fruit_words = [w for w in ["apple", "banana", "orange"] if w in word_ids]
        nonfruit_words = [w for w in ["cat", "dog", "car", "red", "the"] if w in word_ids]

        fruit_mean_proj = np.mean([np.array(all_word_projections[w]) for w in fruit_words], axis=0) if fruit_words else np.zeros(n_layers+1)
        nonfruit_mean_proj = np.mean([np.array(all_word_projections[w]) for w in nonfruit_words], axis=0) if nonfruit_words else np.zeros(n_layers+1)
        fruit_nonfruit_gap = fruit_mean_proj - nonfruit_mean_proj

        # 找到 gap 最大和变化最大的层
        gap_velocity = np.diff(fruit_nonfruit_gap)

        results[word] = {
            "word": word,
            "n_layers": n_layers,
            # 逐层投影 (精简版: 只保留关键层)
            "projections_key_layers": {
                f"L{li}": float(apple_proj[li])
                for li in [0, 1, 2, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1, n_layers]
                if li < len(apple_proj)
            },
            "projections_full": [float(v) for v in apple_proj],
            # 关键注入层
            "top5_injection_layers": [
                {"layer": int(l), "velocity": float(velocities[l])}
                for l in key_injection_layers if l < len(velocities)
            ],
            # 语义选择比
            "selection_ratio_key_layers": {
                f"L{li}": float(selection_ratio[li])
                for li in [0, 1, n_layers//2, n_layers-1, n_layers]
                if li < len(selection_ratio)
            },
            "top5_discrimination_layers": [
                {"layer": int(l), "selection_change": float(selection_change[l])}
                for l in key_discrimination_layers if l < len(selection_change)
            ],
            # 水果-非水果区分
            "fruit_nonfruit_gap_final": float(fruit_nonfruit_gap[-1]),
            "top5_gap_change_layers": [
                {"layer": int(l), "gap_velocity": float(gap_velocity[l])}
                for l in np.argsort(np.abs(gap_velocity))[::-1][:5]
                if l < len(gap_velocity)
            ],
        }

        elapsed = time.time() - t0
        print(f"  {word}: final_proj={apple_proj[-1]:.4f}, embed_proj={apple_proj[0]:.4f}, "
              f"top_injection_L{key_injection_layers[0] if len(key_injection_layers)>0 else '?'}"
              f"(v={velocities[key_injection_layers[0]]:.4f})" if len(key_injection_layers) > 0 else "",
              f"fruit_nonfruit_gap={fruit_nonfruit_gap[-1]:.4f}, "
              f"elapsed={elapsed:.1f}s")

        del outputs
        gc.collect()

    # 跨词语义路径比较
    if len(results) >= 3:
        path_comparison = {}
        for li in range(n_layers + 1):
            projs = {w: results[w]["projections_full"][li] for w in results if "projections_full" in results[w] and li < len(results[w]["projections_full"])}
            if projs:
                vals = list(projs.values())
                path_comparison[f"L{li}"] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "range": float(max(vals) - min(vals)),
                }

        # 找到方差最大的层 (最关键的语义分化层)
        high_variance_layers = sorted(
            [(k, v["std"]) for k, v in path_comparison.items()],
            key=lambda x: x[1], reverse=True
        )[:5]

        results["_path_comparison"] = {
            "high_variance_layers": high_variance_layers,
            "layer_stats": path_comparison,
        }

    return results


# ================================================================
# P694: 信息瓶颈分析
# ================================================================
def p694_information_bottleneck(model, tokenizer, model_info, W_embed, W_U, device):
    """
    P694: 信息瓶颈分析

    核心问题: 哪一层是语义信息的关键瓶颈?

    如果消融某一层的输出, logit(apple) 如何变化?
    找到信息流的"必经之路"

    算法:
    1. 基线: 正常前向传播, 得到 logit(apple)
    2. 消融实验: 对每一层 l, 将 h_{l+1} 替换为 h_l (跳过第l层)
       → logit_ablated_l = W_U[apple] · h_l (而不是 h_{l+1})
       → delta_logit_l = logit_ablated_l - logit_baseline
    3. 如果 |delta_logit_l| 很大: 第l层是关键瓶颈
    4. 更精确: 只消融 G_l 或只消融 A_l, 看哪个更关键

    注意: 这里不用真正的hook干预(太重), 而是用数学近似:
      skip_layer_l ≈ h_final - (G_l + A_l) = h_final - delta_l
      logit_skip_l ≈ logit_final - W_U[word_id] · delta_l

    这个近似是精确的(线性分解), 不需要重新前向传播

    关键发现: 哪些层是不可跳过的? G 还是 A 是瓶颈?
    """
    print("\n" + "="*70)
    print("P694: 信息瓶颈分析")
    print("="*70)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    d_mlp = model_info.intermediate_size
    mlp_type = model_info.mlp_type
    layers = get_layers(model)

    results = {}

    for word in ["apple", "banana", "cat", "the"]:
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if not word_ids:
            continue
        word_id = word_ids[0]
        W_U_word = W_U[word_id]

        print(f"\n  --- {word} ---")
        t0 = time.time()

        # 前向传播
        text = f"The word is {word}."
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # 收集各层 h 和 delta
        all_h = []
        for li in range(len(outputs.hidden_states)):
            h = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
            all_h.append(h)

        h_final = all_h[-1]
        baseline_logit = float(W_U_word @ h_final)

        # 逐层消融分析
        layer_bottleneck = []

        for li in range(n_layers):
            h_before = all_h[li]
            h_after = all_h[li + 1]
            delta = h_after - h_before

            # 消融整层的效果
            skip_logit = baseline_logit - float(W_U_word @ delta)
            skip_delta = skip_logit - baseline_logit

            # 分离 G 和 A 的瓶颈贡献
            lw = get_layer_weights(layers[li], d_model, mlp_type)
            W_gate = lw.W_gate
            W_up = lw.W_up
            W_down = lw.W_down

            if W_gate is not None and W_up is not None and W_down is not None:
                gate_val = safe_sigmoid(W_gate @ h_before)
                up_val = W_up @ h_before
                post_val = gate_val * up_val
                G_exact = W_down @ post_val
                A_approx = delta - G_exact

                G_skip_logit = baseline_logit - float(W_U_word @ G_exact)
                A_skip_logit = baseline_logit - float(W_U_word @ A_approx)

                G_bottleneck = float(W_U_word @ G_exact)  # 消融G的logit变化
                A_bottleneck = float(W_U_word @ A_approx)  # 消融A的logit变化
            else:
                G_bottleneck = 0.0
                A_bottleneck = 0.0

            layer_bottleneck.append({
                "layer": li,
                "skip_logit_delta": float(skip_delta),
                "skip_logit_delta_pct": float(skip_delta / max(abs(baseline_logit), 1e-10)),
                "G_bottleneck": float(G_bottleneck),
                "A_bottleneck": float(A_bottleneck),
                "G_bottleneck_pct": float(G_bottleneck / max(abs(skip_delta), 1e-10)),
                "A_bottleneck_pct": float(A_bottleneck / max(abs(skip_delta), 1e-10)),
                "delta_norm": float(np.linalg.norm(delta)),
            })

            del lw, W_gate, W_up, W_down

        # 找到最关键的瓶颈层
        sorted_by_abs_delta = sorted(layer_bottleneck, key=lambda x: abs(x["skip_logit_delta"]), reverse=True)
        top5_bottleneck = sorted_by_abs_delta[:5]

        # G vs A 的整体瓶颈贡献
        total_G_bottleneck = sum(abs(lb["G_bottleneck"]) for lb in layer_bottleneck)
        total_A_bottleneck = sum(abs(lb["A_bottleneck"]) for lb in layer_bottleneck)

        # 正贡献层 vs 负贡献层
        positive_layers = [lb for lb in layer_bottleneck if lb["skip_logit_delta"] > 0]
        negative_layers = [lb for lb in layer_bottleneck if lb["skip_logit_delta"] < 0]

        results[word] = {
            "baseline_logit": baseline_logit,
            "top5_bottleneck_layers": top5_bottleneck,
            "total_G_bottleneck_abs": total_G_bottleneck,
            "total_A_bottleneck_abs": total_A_bottleneck,
            "G_A_bottleneck_ratio": total_G_bottleneck / max(total_A_bottleneck, 1e-10),
            "n_positive_layers": len(positive_layers),
            "n_negative_layers": len(negative_layers),
            "sum_positive_skip_delta": sum(lb["skip_logit_delta"] for lb in positive_layers),
            "sum_negative_skip_delta": sum(lb["skip_logit_delta"] for lb in negative_layers),
            # 逐层瓶颈数据
            "layer_bottleneck_full": layer_bottleneck,
        }

        elapsed = time.time() - t0
        print(f"  {word}: baseline={baseline_logit:.4f}, "
              f"top_bottleneck_L{top5_bottleneck[0]['layer']}(Δ={top5_bottleneck[0]['skip_logit_delta']:.4f}), "
              f"G/A_bottleneck={total_G_bottleneck/max(total_A_bottleneck,1e-10):.2f}, "
              f"n_pos={len(positive_layers)}, n_neg={len(negative_layers)}, "
              f"elapsed={elapsed:.1f}s")

        del outputs, all_h
        gc.collect()

    # 跨词瓶颈比较
    if len(results) >= 3:
        # 哪些层是所有词的共同瓶颈?
        common_bottleneck_layers = {}
        for li in range(n_layers):
            deltas = []
            for word, wr in results.items():
                for lb in wr["layer_bottleneck_full"]:
                    if lb["layer"] == li:
                        deltas.append(lb["skip_logit_delta"])
                        break
            if deltas:
                common_bottleneck_layers[f"L{li}"] = {
                    "mean_abs_delta": float(np.mean(np.abs(deltas))),
                    "std_delta": float(np.std(deltas)),
                    "all_same_sign": all(d >= 0 for d in deltas) or all(d <= 0 for d in deltas),
                }

        # 排序找共同瓶颈
        sorted_common = sorted(
            [(k, v["mean_abs_delta"]) for k, v in common_bottleneck_layers.items()],
            key=lambda x: x[1], reverse=True
        )

        results["_common_bottleneck"] = {
            "top5_common": sorted_common[:5],
            "layer_stats": common_bottleneck_layers,
        }

    return results


# ================================================================
# 主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    model_name = args.model
    print("="*70)
    print(f"Phase CLVIII: 残差流信息分解 — {model_name}")
    print("="*70)

    # 加载模型
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)

    print(f"Model: {model_info.model_class}")
    print(f"Layers: {model_info.n_layers}, d_model: {model_info.d_model}, "
          f"d_mlp: {model_info.intermediate_size}, vocab: {model_info.vocab_size}")

    # 获取 W_embed 和 W_U
    W_embed = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    lm_head = model.get_output_embeddings()
    if lm_head is not None:
        W_U = lm_head.weight.detach().float().cpu().numpy()
    else:
        W_U = W_embed.copy()

    print(f"W_embed: {W_embed.shape}, W_U: {W_U.shape}")

    all_results = {
        "model": model_name,
        "model_info": {
            "class": model_info.model_class,
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
            "d_mlp": model_info.intermediate_size,
            "vocab_size": model_info.vocab_size,
            "mlp_type": model_info.mlp_type,
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # P692: h_final 的 logit 逐层分解
    all_results["P692"] = p692_logit_layer_decomposition(model, tokenizer, model_info, W_embed, W_U, device)
    gc.collect()

    # P693: 语义信息的层间传播路径
    all_results["P693"] = p693_semantic_propagation_path(model, tokenizer, model_info, W_embed, W_U, device)
    gc.collect()

    # P694: 信息瓶颈分析
    all_results["P694"] = p694_information_bottleneck(model, tokenizer, model_info, W_embed, W_U, device)
    gc.collect()

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUT_DIR / f"phase_clviii_{model_name}_{timestamp}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {out_file}")

    # 打印关键发现摘要
    print("\n" + "="*70)
    print("关键发现摘要")
    print("="*70)
    print(f"\n模型: {model_name}")

    # P692 摘要
    if "P692" in all_results:
        print("\n--- P692: logit 逐层分解 ---")
        for word in ["apple", "banana", "cat", "the"]:
            if word in all_results["P692"]:
                wr = all_results["P692"][word]
                print(f"  {word}: final={wr.get('final_logit', 0):.4f}, "
                      f"embed={wr.get('embed_logit', 0):.4f}, "
                      f"ΣG={wr.get('total_G_logit', 0):.4f}, ΣA={wr.get('total_A_logit', 0):.4f}, "
                      f"G/A_ratio={wr.get('G_vs_A_ratio', 0):.2f}")

    # P693 摘要
    if "P693" in all_results:
        print("\n--- P693: 语义传播路径 ---")
        for word in ["apple", "banana", "cat"]:
            if word in all_results["P693"]:
                wr = all_results["P693"][word]
                top_inj = wr.get("top5_injection_layers", [])
                top_disc = wr.get("top5_discrimination_layers", [])
                print(f"  {word}: final_proj={wr.get('projections_full', [0])[-1]:.4f}, "
                      f"top_injection={top_inj[:2] if top_inj else 'N/A'}, "
                      f"top_discrim={top_disc[:2] if top_disc else 'N/A'}")

    # P694 摘要
    if "P694" in all_results:
        print("\n--- P694: 信息瓶颈 ---")
        for word in ["apple", "banana", "cat", "the"]:
            if word in all_results["P694"]:
                wr = all_results["P694"][word]
                top5 = wr.get("top5_bottleneck_layers", [])
                print(f"  {word}: baseline={wr.get('baseline_logit', 0):.4f}, "
                      f"top_bottleneck_L{top5[0]['layer'] if top5 else '?'}"
                      f"(Δ={top5[0]['skip_logit_delta']:.4f})" if top5 else "",
                      f"G/A_bottleneck={wr.get('G_A_bottleneck_ratio', 0):.2f}")

    # 释放模型
    release_model(model)
    print("[model_utils] GPU memory released")


if __name__ == "__main__":
    main()
